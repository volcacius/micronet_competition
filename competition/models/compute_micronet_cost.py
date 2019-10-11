import torch
from operator import mul
from functools import reduce

from brevitas.nn import QuantConv2d, QuantLinear

import competition.cost_vars as cost_vars


def update_conv_or_linear_cost(mod, input_bit_width, acc_bit_width, quant_acc):

    with torch.no_grad():

        # Compute the quantized weights with their scale and bit width
        quant_weight, weight_scale, weight_bit_width = mod.weight_quant(mod.weight)
        # Get the number of weights
        num_weights = reduce(mul, quant_weight.size(), 1)
        # Add the memory cost to the global tracker
        cost_vars.micronet_memory_cost += num_weights * weight_bit_width.item()

        # Get the number of output elements, excluding batch size
        num_output_elems = reduce(mul, quant_acc.size()[1:], 1)
        # Get the total number of muls (adds) as the product between the number of muls (adds) per output element
        # and the number of output elems. Brevitas' per_elem_ops attribute put muls and adds toghether, so we divide by
        # two first.
        num_muls = (mod.per_elem_ops / 2) * num_output_elems
        num_adds = (mod.per_elem_ops / 2) * num_output_elems
        # Get the bit width considered for muls
        muls_bit_width = max(weight_bit_width.item(), input_bit_width.item())
        # Add the muls cost to the global tracker
        cost_vars.micronet_math_cost += muls_bit_width * num_muls
        # Add the adds cost to the global tracker. Bit width considered is the size of the accumulator. This assume
        # a naive implementation without an adder tree.
        cost_vars.micronet_math_cost += acc_bit_width * num_adds

        # Add the cost of quantized biases
        if isinstance(mod, QuantConv2d):
            assert(mod.bias is None)
        if isinstance(mod, QuantLinear):
            # Bias is quantized to the number of bits of the accumulator
            num_biases = reduce(mul, mod.bias.size(), 1)
            # Add the memory cost of biases to the global tracker
            cost_vars.micronet_memory_cost += num_biases * acc_bit_width
            # Account for the number of adds performed when adding biases. This amount to performing an elemwise add on
            # each element in the accumulator
            cost_vars.micronet_math_cost += num_output_elems * acc_bit_width


def update_act_cost(acc_bit_width, quant_acc, out_bit_width):

    with torch.no_grad():

        # Get the number of channels going into the activation.
        channels = quant_acc.size()[1]
        # Compute the memory cost of the threshold. The size of the thresholds in terms of bit-width is determined by
        # the number of bits in the accumulator that is compared against. Each channel has its own set of thresholds.
        # Each set of thresholds contains as many thresholds as the levels of quantization in the output.
        thresholds_per_channel = 2 ** out_bit_width - 1
        cost_vars.micronet_memory_cost += channels * acc_bit_width * thresholds_per_channel

        # A comparison between the value of the accumulator and the threshold is computed as a subtraction. At worst
        # each element in the accumulator is compared with each threshold for its channel. This assumes a naive
        # implementation without a comparison tree
        num_output_elems = reduce(mul, quant_acc.size()[1:], 1)
        cost_vars.micronet_math_cost += num_output_elems * acc_bit_width * thresholds_per_channel


def update_elemwise_add_cost(quant_acc, lhs_bit_width, rhs_bit_width):

    with torch.no_grad():

        #Get the number of output elem
        num_output_elems = reduce(mul, quant_acc.size()[1:], 1)
        # Get the bit width considered for elemwise adds
        elemwise_adds_bit_width = max(lhs_bit_width.item(), rhs_bit_width.item())
        # Update the global tracker
        cost_vars.micronet_math_cost += num_output_elems * elemwise_adds_bit_width


def update_avg_pool_cost(mod, quant_out, input_bit_width):

    with torch.no_grad():

        # Quant avg pool is implemented as a sum and and a truncation
        # Per rules truncation is not considered so only the accumulation cost is taken into account
        # Get the size of the accumulator
        acc_bit_width = mod.max_output_bit_width(input_bit_width).item()
        # Number of adds per output element is given by the kernel size
        per_elem_adds = mod.kernel_size * mod.kernel_size
        # Get the number of output elements, excluding batch size
        num_output_elems = reduce(mul, quant_out.size()[1:], 1)
        # Compute total number of adds and update global tracker
        cost_vars.micronet_math_cost += per_elem_adds * acc_bit_width * num_output_elems
