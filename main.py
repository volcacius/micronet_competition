import argparse
import random

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType

from competition.loss.smoothing import *
from competition.data.dataloders import *
from competition.data.mix import *
from competition.training import *
from competition.utils import *
from competition.models.proxylessnas import *

import competition.cost_vars as cost_vars


def add_parser_arguments(parser):

    parser.add_argument('--data', type=str, help='path to dataset')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256) per gpu')
    parser.add_argument('--optimizer-batch-size', default=-1, type=int,
                        metavar='N', help='size of a total batch size, for simulating bigger batches')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-schedule', default='cosine', type=str, metavar='SCHEDULE',
                        choices=['step','linear','cosine'])
    parser.add_argument("--milestones", type=str, default='15,30,45', help="Scheduler milestones")
    parser.add_argument('--warmup', default=0, type=int,
                        metavar='E', help='number of warmup epochs')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        metavar='S', help='label smoothing')
    parser.add_argument('--mix-rate', default=0.0, type=float,
                        metavar='ALPHA', help='mixup/cutmix beta')
    parser.add_argument('--mix-prob', default=0.5, type=float, help='Probability of applying cutmix/mixup')
    parser.add_argument('--mix-type', default=MixType.CUTMIX, type=MixType)
    parser.add_argument('--dropout-steps', type=int, default=0)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--bn-no-wd', action='store_true', help='No weight decay on BN parameters')
    parser.add_argument('--norm-input', action='store_true', help='Apply dataset mean/var normalization')
    parser.add_argument('--finetune', action='store_true', help='Retrain a restored model')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov momentum, default: false)')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--prof', type=int, default=-1,
                        help='Run only N iterations')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--seed', default=123456, type=int,
                        help='random seed used for np and pytorch')
    parser.add_argument('--gather-checkpoints', action='store_true',
                        help='Gather checkpoints throughout the training')
    parser.add_argument('--report-file', default='experiment_report.json', type=str,
                        help='file in which to store JSON experiment report')
    parser.add_argument('--final-weights', default='model.pth.tar', type=str,
                        help='file in which to store final model weights')
    parser.add_argument('--evaluate', action='store_true', help='evaluate checkpoint/model')
    parser.add_argument('--compute-micronet-cost', action='store_true', help='Compute micronet cost')
    parser.add_argument('--training-only', action='store_true', help='do not evaluate')
    parser.add_argument('--no-checkpoints', action='store_false', dest='save_checkpoints')
    parser.add_argument('--workspace', type=str, default='./proxylessnas_4b_5b')
    parser.add_argument('--bit-width', type=int, default=4,
                        help='Per layer bit-width, unless specified otherwise')
    parser.add_argument('--depthwise-bit-width', type=int, default=5,
                        help='Input and weights bit-width for depthwise convs')
    parser.add_argument('--first-layer-bit-width', type=int, default=8,
                        help='Bit-width for the weights of the first layer')
    parser.add_argument('--hard-tanh-threshold', type=int, default=10,
                        help='Bit-width for the weights of the first layer')
    parser.add_argument('--quant-type', type=QuantType, default=QuantType.INT,
                        help='Type of quantization')
    parser.add_argument('--weight-scaling-impl-type', type=ScalingImplType, default=ScalingImplType.STATS,
                        help='Type of weight scaling implementation')


def main(args):
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print("Warning: simulated batch size {} is not divisible by actual batch size {}"
                  .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size/ tbs)
        print("BSM: {}".format(batch_size_multiplier))

    # optionally resume from a checkpoint
    checkpoint_epoch = -1
    if args.resume:
        if os.path.isfile(args.resume) and args.resume.endswith('.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            checkpoint_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isfile(args.resume) and args.resume.endswith('.pth'):
            model_state = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    if args.finetune or args.evaluate or args.compute_micronet_cost:
        optimizer_state = None
        best_prec1 = 0
        args.start_epoch = 0

    if args.evaluate:
        args.epochs = 1

    if args.start_epoch == -1 and checkpoint_epoch == -1:
        args.start_epoch = 0
    elif args.start_epoch == -1 and checkpoint_epoch != -1:
        args.start_epoch = checkpoint_epoch

    num_classes = IMAGENET_NUM_CLASSES
    input_size = IMAGENET_INPUT_SIZE
    train_loader, val_loader = imagenet_train_loader, imagenet_val_loader

    train_mix_target = args.mix_rate > 0.0 and args.mix_prob > 0.0
    train_loader, train_loader_len = train_loader(data_path=args.data,
                                                  batch_size=args.batch_size,
                                                  num_classes=num_classes,
                                                  one_hot=train_mix_target,
                                                  workers=args.workers,
                                                  input_size=input_size,
                                                  worker_init_fn=_worker_init_fn,
                                                  norm_input=args.norm_input)
    val_loader, val_loader_len = val_loader(data_path=args.data,
                                            batch_size=args.batch_size,
                                            num_classes=num_classes,
                                            one_hot=False,
                                            workers=args.workers,
                                            input_size=input_size,
                                            worker_init_fn=_worker_init_fn,
                                            norm_input=args.norm_input)

    if train_mix_target:
        train_loader = MixWrapper(args.mix_rate, num_classes, train_loader,
                                  mix_type=args.mix_type,
                                  mix_prob=args.mix_prob)

    if not torch.distributed.is_available() \
            or not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0:
        logger = log.Logger(
                args.print_freq,
                [
                    log.JsonBackend(os.path.join(args.workspace, args.report_file), log_level=1),
                    log.StdOut1LBackend(train_loader_len, val_loader_len, args.epochs, args.start_epoch, log_level=0),
                ], start_epoch=args.start_epoch)

        for k, v in args.__dict__.items():
            logger.log_run_tag(k, v)
    else:
        logger = None

    loss = nn.CrossEntropyLoss
    if args.mix_rate > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    model = proxylessnas(model_name='proxylessnas_mobile14',
                         num_classes=num_classes,
                         dropout_rate=args.dropout_rate,
                         dropout_steps=args.dropout_steps,
                         quant_type=args.quant_type,
                         first_layer_bit_width=args.first_layer_bit_width,
                         bit_width=args.bit_width,
                         depthwise_bit_width=args.depthwise_bit_width,
                         weight_scaling_impl_type=args.weight_scaling_impl_type,
                         hard_tanh_threshold=args.hard_tanh_threshold,
                         compute_micronet_cost=args.compute_micronet_cost,
                         workspace=args.workspace)

    if os.path.isfile(args.resume) and args.resume.endswith('.pth'):
        model.load_state_dict(model_state)

    model_and_loss = ModelAndLoss(model, loss,
                                  cuda=True,
                                  epochs=args.epochs)

    optimizer = get_optimizer(model=model_and_loss.model,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              bn_no_wd=args.bn_no_wd,
                              nesterov=args.nesterov,
                              state=optimizer_state)

    if args.lr_schedule == 'step':
        milestones = [int(i) for i in args.milestones.split(',')]
        lr_policy = lr_step_policy(args.lr, milestones, 0.1, args.warmup, logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)
    else:
        raise Exception("LR policy not recognized: {}".format(args.lr_schedule))

    if args.distributed:
        model_and_loss.distributed()

    if os.path.isfile(args.resume) and not args.resume.endswith('.pth'):
        model_and_loss.load_model_state(model_state)

    if args.compute_micronet_cost:
        inp = torch.rand(size=(1, 3, IMAGENET_INPUT_SIZE, IMAGENET_INPUT_SIZE)).cuda()
        model_and_loss.eval()
        model_and_loss.model(inp)
        cost = cost_vars.micronet_memory_cost / cost_vars.MICRONET_MEMORY_REF
        cost += cost_vars.micronet_math_cost / cost_vars.MICRONET_MATH_REF
        print("Micronet cost: {}".format(cost))

    else:
        train_loop(model_and_loss=model_and_loss,
                   optimizer=optimizer,
                   lr_scheduler=lr_policy,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   logger=logger,
                   should_backup_checkpoint=should_backup_checkpoint(args),
                   batch_size_multiplier=batch_size_multiplier,
                   train_mix_target=train_mix_target,
                   start_epoch=args.start_epoch,
                   best_prec1=best_prec1,
                   prof=args.prof,
                   skip_training=args.evaluate,
                   skip_validation=args.training_only,
                   save_checkpoints=args.save_checkpoints and not args.evaluate,
                   checkpoint_dir=args.workspace)

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
            or torch.distributed.get_rank() == 0:
        logger.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    main(args)
