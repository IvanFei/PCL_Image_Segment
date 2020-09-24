import math
import torch

from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer
    """

    def poly_schd(epoch):
        return math.pow(1 - epoch / args.num_epochs, args.poly_exp)

    param_groups = net.parameters()

    if args.optim.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                              momentum=args.momentum, nesterov=False)

    elif args.optim.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError

    if args.lr_schedule == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                        gamma=args.gamma, last_epoch=-1)

    elif args.lr_schedule == "multi_step":
        if isinstance(args.milestones, str):
            args.milestones = eval(args.milestones)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                             gamma=args.gamma, last_epoch=args.last_epoch)

    elif args.lr_schedule == "reduce_lr_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10, threshold=0.001,
                                                   threshold_mode="rel", cooldown=0, min_lr=0)

    elif args.lr_schedule == "poly":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    else:
        raise NotImplementedError

    return optimizer, scheduler

