import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from apex.parallel import DistributedDataParallel as DDP

from . import logger as log
from . import utils
from .utils import to_python_float


class ModelAndLoss(nn.Module):
    def __init__(self, model, loss, epochs, cuda=True):
        super(ModelAndLoss, self).__init__()
        if cuda:
            model = model.cuda()

        self.cuda = cuda
        criterion = loss()
        self.epochs = epochs
        if cuda:
            criterion = criterion.cuda()
        self.model = model
        self.loss = criterion

    def forward(self, data, target, mix_target):
        if mix_target:
            targets, shuffled_targets, lam = target
            target = lam * targets + (1 - lam) * shuffled_targets
        output = self.model(data)
        loss = sum((self.loss(o, target) for o in output)) # supports multisample dropout
        loss = loss / len(output)
        output = sum(output) / len(output)
        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)
            if self.cuda:
                self.model = self.model.cuda()


def get_optimizer(model,
                  lr,
                  momentum,
                  weight_decay,
                  bn_no_wd,
                  nesterov=False,
                  state=None):
    if bn_no_wd:
        parameters =list(model.named_parameters())
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        parameters_to_pass = [{'params': bn_params, 'weight_decay': 0},
                              {'params': rest_params, 'weight_decay': weight_decay}]
    else:
        parameters_to_pass = model.parameters()
    optimizer = torch.optim.SGD(parameters_to_pass,
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=nesterov)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr', log.IterationMeter(), log_level=1)
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1-(e/es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier)/es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(model_and_loss,
                   optimizer,
                   mix_target,
                   batch_size_multiplier=1):
    def _step(input, target, optimizer_step = True):
        loss, output = model_and_loss(input, target, mix_target)
        correct_percent, correct_absolute = utils.accuracy(output.data, target, mix_target, topk=(1, 5))
        prec1, prec5 = correct_percent

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        loss.backward()

        if optimizer_step:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad /= batch_size_multiplier

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def train(train_loader,
          model_and_loss,
          optimizer,
          lr_scheduler,
          logger,
          epoch,
          mix_target,
          prof=-1,
          batch_size_multiplier=1,
          register_metrics=True):

    if register_metrics and logger is not None:
        logger.register_metric('train.top1', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.top5', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.loss', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.compute_ips', log.AverageMeter(), log_level=1)
        logger.register_metric('train.total_ips', log.AverageMeter(), log_level=1)
        logger.register_metric('train.data_time', log.AverageMeter(), log_level=1)
        logger.register_metric('train.compute_time', log.AverageMeter(), log_level=1)

    step = get_train_step(model_and_loss=model_and_loss,
                          optimizer=optimizer,
                          mix_target=mix_target,
                          batch_size_multiplier=batch_size_multiplier)

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        if prof > 0:
            if i >= prof:
                break

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step = optimizer_step)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric('train.top1', to_python_float(prec1))
            logger.log_metric('train.top5', to_python_float(prec5))
            logger.log_metric('train.loss', to_python_float(loss))
            logger.log_metric('train.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('train.total_ips', calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)

        end = time.time()


def get_val_step(model_and_loss):

    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var, mix_target=False)

        correct_percent, correct_absolute = utils.accuracy(output.data, target, topk=(1, 5), mix_target=False)
        prec1, prec5 = correct_percent
        abs_prec1, abs_prec5 = correct_absolute

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5, abs_prec1, abs_prec5

    return _step


def validate(val_loader, model_and_loss, logger, prof=-1, register_metrics=True):

    if register_metrics and logger is not None:
        logger.register_metric('val.top1',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.top5',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.abs_top1',     log.SumMeter(), log_level = 0)
        logger.register_metric('val.abs_top5',     log.SumMeter(), log_level = 0)
        logger.register_metric('val.loss',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.compute_ips',  log.AverageMeter(), log_level = 1)
        logger.register_metric('val.total_ips',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.data_time',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.compute_time', log.AverageMeter(), log_level = 1)

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        if prof > 0:
            if i > prof:
                break

        loss, prec1, prec5, abs_prec1, abs_prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)

        if logger is not None:
            logger.log_metric('val.top1', to_python_float(prec1), bs)
            logger.log_metric('val.top5', to_python_float(prec5), bs)
            logger.log_metric('val.abs_top1', to_python_float(abs_prec1))
            logger.log_metric('val.abs_top5', to_python_float(abs_prec5))
            logger.log_metric('val.loss', to_python_float(loss), bs)
            logger.log_metric('val.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('val.total_ips', calc_ips(bs, it_time))
            logger.log_metric('val.data_time', data_time)
            logger.log_metric('val.compute_time', it_time - data_time)

        end = time.time()

    return top1.get_val()


def calc_ips(batch_size, time):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    tbs = world_size * batch_size
    return tbs/time


def train_loop(model_and_loss,
               optimizer,
               lr_scheduler,
               train_loader,
               val_loader,
               epochs,
               logger,
               should_backup_checkpoint,
               train_mix_target,
               batch_size_multiplier=1,
               best_prec1=0,
               start_epoch=0,
               prof=-1,
               skip_training=False,
               skip_validation=False,
               save_checkpoints=True,
               checkpoint_dir='./'):

    prec1 = -1

    epoch_iter = range(start_epoch, epochs)
    if logger is not None:
        epoch_iter = logger.epoch_generator_wrapper(epoch_iter)
    for epoch in epoch_iter:
        if not skip_training:
            train(train_loader=train_loader,
                  model_and_loss=model_and_loss,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  logger=logger,
                  epoch=epoch,
                  mix_target=train_mix_target,
                  prof = prof,
                  register_metrics=epoch==start_epoch,
                  batch_size_multiplier=batch_size_multiplier)

        if not skip_validation:
            prec1 = validate(val_loader=val_loader,
                             model_and_loss=model_and_loss,
                             logger=logger,
                             prof=prof,
                             register_metrics=epoch==start_epoch)

        if save_checkpoints and (not (torch.distributed.is_available() and torch.distributed.is_initialized())
                                 or torch.distributed.get_rank() == 0):
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if should_backup_checkpoint(epoch):
                backup_filename = 'checkpoint-{}.pth.tar'.format(epoch + 1)
            else:
                backup_filename = None

            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_and_loss.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_dir=checkpoint_dir, backup_filename=backup_filename)
