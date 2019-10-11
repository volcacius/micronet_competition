"""
Source: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv

MIT License

Copyright (c) 2019 Alessandro Pappalardo
Copyright (c) 2018 Oleg Sémery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__all__ = ['proxylessnas']

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.nn import QuantReLU, QuantConv2d, QuantHardTanh, QuantAvgPool2d, QuantLinear
from brevitas.quant_tensor import pack_quant_tensor

from .compute_micronet_cost import *

MIN_SCALING_VALUE = 2e-16
RELU_MAX_VAL = 6.0


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 quant_type,
                 weight_bit_width,
                 act_bit_width,
                 act_scaling_per_channel,
                 weight_scaling_impl_type,
                 bias,
                 compute_micronet_cost,
                 dilation=1,
                 groups=1,
                 bn_eps=1e-5,
                 shared_act=None):
        super(ConvBlock, self).__init__()
        self.compute_micronet_cost = compute_micronet_cost

        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            weight_quant_type=quant_type,
            weight_bit_width=weight_bit_width,
            weight_scaling_impl_type=weight_scaling_impl_type,
            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
            weight_narrow_range=True,
            weight_scaling_stats_op=StatsOp.MAX,
            weight_scaling_min_val=MIN_SCALING_VALUE,
            compute_output_bit_width=True, # Compute the number of bits in the output accumulator
            return_quant_tensor=True, # Return a quantized tensor that represents the quantized accumulator
            weight_scaling_per_output_channel=True)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps)
        if shared_act is None and quant_type == QuantType.FP:
            self.activ = nn.ReLU6()
        elif shared_act is None and quant_type == QuantType.INT:
            self.activ = QuantReLU(
                quant_type=quant_type,
                bit_width=act_bit_width,
                max_val=RELU_MAX_VAL,
                scaling_per_channel=act_scaling_per_channel,
                scaling_impl_type=ScalingImplType.PARAMETER,
                scaling_min_val=MIN_SCALING_VALUE,
                restrict_scaling_type=RestrictValueType.LOG_FP,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                return_quant_tensor=True)
        elif shared_act is not None:
            self.activ = shared_act
        else:
            raise Exception("Activ non recognized.")

    def forward(self, x):
        quant_input, input_scale, input_bit_width = x
        x = self.conv(x)
        quant_acc, acc_scale, acc_bit_width = x
        if self.compute_micronet_cost:
            # Add cost of conv's quantized weights, muls and adds.
            update_conv_or_linear_cost(self.conv, input_bit_width, acc_bit_width, quant_acc)
        # Pass the accumulator through batch norm. BN is going to be merged into the thresholds that implements
        # the activation function
        x = self.bn(quant_acc)
        # Pass the normalized accumulator through the quantized activation
        x = self.activ(x)
        quant_out, out_scale, out_bit_width = x
        if self.compute_micronet_cost:
            update_act_cost(acc_bit_width, quant_acc, out_bit_width)
        return x


class ProxylessBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion,
                 bit_width,
                 depthwise_bit_width,
                 quant_type,
                 weight_scaling_impl_type,
                 shared_act,
                 compute_micronet_cost):
        super(ProxylessBlock, self).__init__()
        self.compute_micronet_cost = compute_micronet_cost
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        if self.use_bc:
            self.bc_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bn_eps=bn_eps,
                act_scaling_per_channel=True,
                quant_type=quant_type,
                weight_bit_width=bit_width,
                bias=False,
                weight_scaling_impl_type=weight_scaling_impl_type,
                act_bit_width=depthwise_bit_width,
                compute_micronet_cost=compute_micronet_cost)

        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=mid_channels,
            bn_eps=bn_eps,
            act_scaling_per_channel=False,
            quant_type=quant_type,
            weight_bit_width=depthwise_bit_width,
            act_bit_width=bit_width,
            weight_scaling_impl_type=weight_scaling_impl_type,
            bias=False,
            compute_micronet_cost=compute_micronet_cost)
        self.pw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=bn_eps,
            weight_bit_width=bit_width,
            quant_type=quant_type,
            shared_act=shared_act,
            bias=False,
            act_bit_width=None,
            weight_scaling_impl_type=weight_scaling_impl_type,
            act_scaling_per_channel=None,
            compute_micronet_cost=compute_micronet_cost)

    def forward(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion,
                 residual,
                 shortcut,
                 bit_width,
                 depthwise_bit_width,
                 quant_type,
                 weight_scaling_impl_type,
                 shared_act,
                 compute_micronet_cost):
        super(ProxylessUnit, self).__init__()
        assert (residual or shortcut)
        assert(shared_act is not None)
        self.compute_micronet_cost = compute_micronet_cost
        self.residual = residual
        self.shortcut = shortcut

        if self.residual:
            self.body = ProxylessBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bn_eps=bn_eps,
                expansion=expansion,
                bit_width=bit_width,
                depthwise_bit_width=depthwise_bit_width,
                quant_type=quant_type,
                weight_scaling_impl_type=weight_scaling_impl_type,
                shared_act=shared_act,
                compute_micronet_cost=compute_micronet_cost)
            self.shared_act = shared_act

    def forward(self, x):
        if not self.residual:
            return x
        if not self.shortcut:
            x = self.body(x)
            return x
        identity = x
        x = self.body(x)

        # Take the bit width of the rhs tensor in the elemwise add
        quant_rhs, rhs_scale, rhs_bit_width = x
        # Take the bit width of the lhs tensor in the elemwise add
        quant_lhs, lhs_scale, lhs_bit_width = x
        x = identity + x
        quant_acc, acc_scale, acc_bit_width = x
        if self.compute_micronet_cost:
            # Compute the cost of the elemwise add
            update_elemwise_add_cost(quant_acc, lhs_bit_width, rhs_bit_width)
        x = self.shared_act(x)
        quant_out, out_scale, out_bit_width = x
        if self.compute_micronet_cost:
            # Compute the cost of the quantized activation
            update_act_cost(acc_bit_width, quant_acc, out_bit_width)
        return x


class ProxylessNAS(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 residuals,
                 shortcuts,
                 kernel_sizes,
                 expansions,
                 quant_type,
                 bit_width,
                 depthwise_bit_width,
                 first_layer_bit_width,
                 hard_tanh_threshold,
                 dropout_rate,
                 dropout_steps,
                 weight_scaling_impl_type,
                 compute_micronet_cost,
                 input_bit_width=8,
                 bn_eps=1e-3,
                 in_channels=3,
                 num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.compute_micronet_cost = compute_micronet_cost
        self.input_bit_width = torch.tensor(input_bit_width).float().cuda()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.dropout_steps = dropout_steps

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            bn_eps=bn_eps,
            act_scaling_per_channel=False,
            weight_scaling_impl_type=weight_scaling_impl_type,
            bias=False,
            quant_type=quant_type,
            act_bit_width=bit_width,
            weight_bit_width=first_layer_bit_width,
            compute_micronet_cost=compute_micronet_cost))
        in_channels = init_block_channels
        shared_act = None
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            residuals_per_stage = residuals[i]
            shortcuts_per_stage = shortcuts[i]
            kernel_sizes_per_stage = kernel_sizes[i]
            expansions_per_stage = expansions[i]
            for j, out_channels in enumerate(channels_per_stage):
                residual = (residuals_per_stage[j] == 1)
                shortcut = (shortcuts_per_stage[j] == 1)
                kernel_size = kernel_sizes_per_stage[j]
                expansion = expansions_per_stage[j]
                stride = 2 if (j == 0) and (i != 0) else 1

                if not shortcut:
                    shared_act = QuantHardTanh(
                        bit_width=bit_width,
                        quant_type=quant_type,
                        scaling_per_channel=False,
                        scaling_impl_type=ScalingImplType.PARAMETER,
                        scaling_min_val=MIN_SCALING_VALUE,
                        max_val=hard_tanh_threshold,
                        min_val=-hard_tanh_threshold,
                        restrict_scaling_type=RestrictValueType.LOG_FP,
                        return_quant_tensor=True)

                stage.add_module("unit{}".format(j + 1), ProxylessUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bn_eps=bn_eps,
                    expansion=expansion,
                    residual=residual,
                    shortcut=shortcut,
                    bit_width=bit_width,
                    depthwise_bit_width=depthwise_bit_width,
                    quant_type=quant_type,
                    weight_scaling_impl_type=weight_scaling_impl_type,
                    shared_act=shared_act,
                    compute_micronet_cost=compute_micronet_cost))
                in_channels = out_channels

            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", ConvBlock(
                in_channels=in_channels,
                out_channels=final_block_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bn_eps=bn_eps,
                act_scaling_per_channel=False,
                quant_type=quant_type,
                act_bit_width=bit_width,
                weight_bit_width=bit_width,
                weight_scaling_impl_type=weight_scaling_impl_type,
                bias=False,
                compute_micronet_cost=compute_micronet_cost))
        in_channels = final_block_channels
        self.final_pool = QuantAvgPool2d(
            kernel_size=7,
            stride=1,
            quant_type=quant_type,
            min_overall_bit_width=bit_width,
            max_overall_bit_width=bit_width)

        self.output = QuantLinear(
            in_features=in_channels,
            out_features=num_classes,
            bias=True,
            bias_quant_type=quant_type,
            compute_output_bit_width=quant_type == QuantType.INT,
            compute_output_scale=quant_type == QuantType.INT,
            weight_bit_width=bit_width,
            weight_quant_type=quant_type,
            weight_scaling_min_val=MIN_SCALING_VALUE,
            weight_scaling_per_output_channel=False,
            weight_scaling_stats_op=StatsOp.MAX,
            weight_narrow_range=True,
            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
            weight_scaling_impl_type=weight_scaling_impl_type,
            return_quant_tensor=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def dropout_classifier(self, x, scale, bit_width):
        if self.training and self.dropout_steps > 0:
            out_list = []
            for i in range(self.dropout_steps):
                out = F.dropout(x, p=self.dropout_rate)
                out = self.output(pack_quant_tensor(out, scale, bit_width))
                quant_out, out_scale, out_bit_width = out
                out_list.append(quant_out)
            return tuple(out_list)
        else:
            out = self.output(pack_quant_tensor(x, scale, bit_width))
            quant_out, out_scale, out_bit_width = out
            if self.compute_micronet_cost:
                update_conv_or_linear_cost(self.output, bit_width, out_bit_width, quant_out)
            return quant_out,

    def forward(self, x):
        x = pack_quant_tensor(x, None, self.input_bit_width)
        x = self.features(x)
        quant_input, input_scale, input_bit_width = x
        x, scale, bit_width = self.final_pool(x)
        if self.compute_micronet_cost:
            update_avg_pool_cost(self.final_pool, x, input_bit_width)
        x = x.view(x.size(0), -1)
        x = self.dropout_classifier(x, scale, bit_width)
        return x


def proxylessnas(**kwargs):

    model_name = kwargs['model_name']

    if model_name == 'proxylessnas_mobile14':
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [40, 40, 40, 40], [56, 56, 56, 56], [112, 112, 112, 112, 136, 136, 136, 136],
                    [256, 256, 256, 256, 448]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 48
        final_block_channels = 1792
    else:
        raise Exception("Model not recognized: {}.".format(model_name))

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expansions=expansions,
        compute_micronet_cost=kwargs['compute_micronet_cost'],
        quant_type=kwargs['quant_type'],
        bit_width=kwargs['bit_width'],
        first_layer_bit_width=kwargs['first_layer_bit_width'],
        hard_tanh_threshold=kwargs['hard_tanh_threshold'],
        weight_scaling_impl_type=kwargs['weight_scaling_impl_type'],
        dropout_rate=kwargs['dropout_rate'],
        dropout_steps=kwargs['dropout_steps'],
        depthwise_bit_width=kwargs['depthwise_bit_width'])

    return net
