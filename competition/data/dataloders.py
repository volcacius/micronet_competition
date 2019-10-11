import os

import torch
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .autoaugment import ImageNetPolicy

IMAGENET_NUM_CLASSES = 1000
IMAGENET_INPUT_SIZE = 224


def imagenet_transform_val(input_size):
    transforms_list = [transforms.Resize(input_size + 32), transforms.CenterCrop(input_size)]
    return transforms.Compose(transforms_list)


def imagenet_transform_train(input_size):
    transforms_list = [transforms.RandomResizedCrop(input_size), ImageNetPolicy()]
    return transforms.Compose(transforms_list)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot, norm_input):
        if norm_input:
            mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
            std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        else:
            mean = torch.tensor([128.0, 128.0, 128.0]).cuda().view(1,3,1,1)
            std = torch.tensor([128.0, 128.0, 128.0]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)

                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, one_hot, norm_input):
        self.dataloader = dataloader
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.norm_input = norm_input

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.num_classes, self.one_hot, self.norm_input)


def train_dataloader(train_dataset,
                     batch_size,
                     workers,
                     num_classes,
                     one_hot,
                     collate,
                     worker_init_fn,
                     norm_input):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True,
                              sampler=train_sampler,
                              collate_fn=collate,
                              drop_last=True)
    return PrefetchedWrapper(train_loader, num_classes, one_hot, norm_input), len(train_loader)


def val_dataloader(val_dataset,
                   batch_size,
                   workers,
                   num_classes,
                   one_hot,
                   collate,
                   worker_init_fn,
                   norm_input):
    val_loader = DataLoader(val_dataset,
                            sampler=None,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            worker_init_fn=worker_init_fn,
                            pin_memory=True,
                            collate_fn=collate)
    return PrefetchedWrapper(val_loader, num_classes, one_hot, norm_input), len(val_loader)


def imagenet_train_loader(data_path,
                          batch_size,
                          num_classes,
                          one_hot,
                          input_size,
                          workers,
                          worker_init_fn,
                          norm_input):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(traindir, imagenet_transform_train(input_size))
    return train_dataloader(train_dataset,
                            batch_size,
                            workers,
                            num_classes,
                            one_hot,
                            fast_collate,
                            worker_init_fn,
                            norm_input)


def imagenet_val_loader(data_path,
                        batch_size,
                        num_classes,
                        one_hot,
                        input_size,
                        workers,
                        worker_init_fn,
                        norm_input):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(valdir, imagenet_transform_val(input_size))
    return val_dataloader(val_dataset,
                          batch_size,
                          workers,
                          num_classes,
                          one_hot,
                          fast_collate,
                          worker_init_fn,
                          norm_input)
