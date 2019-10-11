#
# Source: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/RN50v1.5/image_classification/utils.py
#

import os
import numpy as np
import time
import torch
import shutil
import torch.distributed as dist
from enum import Enum


class AutoName(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
         return name

    def __str__(self):
        return self.value


def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='./', backup_filename=None):
    if (not (torch.distributed.is_available() and torch.distributed.is_initialized())) \
            or torch.distributed.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function


def accuracy(output, target, mix_target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if mix_target:
        targets, shuffled_targets, lam = target
        target = torch.argmax(targets, dim=1)
        shuffled_target = torch.argmax(shuffled_targets, dim=1)
        correct = lam * pred.eq(target.view(1, -1).expand_as(pred)).float() + (1 - lam) * pred.eq(shuffled_target.view(1, -1).expand_as(pred)).float()
    else:
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    batch_size = target.size(0)
    res_percent = []
    res_absolute = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res_absolute.append(correct_k)
        res_percent.append(correct_k.mul(100.0 / batch_size))
    return res_percent, res_absolute


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt

def to_python_float(x):
    return x.item()