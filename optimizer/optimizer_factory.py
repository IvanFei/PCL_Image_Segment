import math
import torch

from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

from config import cfg
from utils.utils import get_logger


logger = get_logger()


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
        logger.info(f"[*] Using The SGD Optimizer with lr {args.lr} and weight decay {args.weight_decay} "
                    f"and momentum {args.momentum}.")

    elif args.optim.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        logger.info(f"[*] Using The Adam Optimizer with lr {args.lr} and weight decay {args.weight_decay}")
    else:
        raise NotImplementedError

    if args.lr_schedule == "step":
        # step_size 30 gamma 0.2
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                        gamma=args.gamma, last_epoch=-1)
        logger.info(f"[*] Using `Step` LR Scheduler with step size {args.step_size} and gamma {args.gamma}")

    elif args.lr_schedule == "multi_step":
        if isinstance(args.milestones, str):
            args.milestones = eval(args.milestones)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                             gamma=args.gamma, last_epoch=args.last_epoch)
        logger.info(f"[*] Using `Multi Step` LR Scheduler with milestones {args.milestones} and gamma {args.gamma}")

    elif args.lr_schedule == "reduce_lr_on_plateau":
        patience, threshold = 8, 0.0005
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=patience, threshold=threshold,
                                                   threshold_mode="rel", cooldown=0, min_lr=0)
        logger.info(f"[*] Using `Reduce Lr On Plateau` LR Scheduler with patience {8} and threshold {threshold}")

    elif args.lr_schedule == "poly":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
        logger.info(f"[*] Using `Poly` LR Scheduler with poly {args.poly_exp}")

    else:
        raise NotImplementedError

    return optimizer, scheduler

