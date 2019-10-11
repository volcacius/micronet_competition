import torch
import numpy as np

from competition.utils import AutoName
from enum import auto


class MixType(AutoName):
    CUTMIX = auto()
    MIXUP = auto()


#
# Adapted from: https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
#
def cutmix(alpha, input, targets):

    indices = torch.randperm(input.size(0))
    shuffled_data = input[indices, :]
    shuffled_targets = targets[indices, :]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = input.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    input[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    return input, (targets, shuffled_targets, lam)


def mixup(alpha, data, target):
    with torch.no_grad():
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)
        perm = torch.randperm(bs).cuda()
        md = c * data + (1 - c) * data[perm, :]
        return md, (target, target[perm, :], c)


class MixWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, mix_type, mix_prob):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.mix_prob = mix_prob
        if mix_type == MixType.CUTMIX:
            self.mix_impl = cutmix
        elif mix_type == MixType.MIXUP:
            self.mix_impl = mixup
        else:
            raise Exception("Mix type not recognized.")

    def mix_loader(self, loader):
        for input, target in loader:
            if torch.rand(1) < self.mix_prob:
                i, t = self.mix_impl(self.alpha, input, target)
            else:
                i, t = input, (target, target, 1.0)
            yield i, t

    def __iter__(self):
        return self.mix_loader(self.dataloader)