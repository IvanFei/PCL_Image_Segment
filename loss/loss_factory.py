import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from config import cfg
from loss.rmi import RMILoss


def get_loss(args, cuda=False):
    """
    Get the criterion based on loss function
    """
    if args.loss_type == "ce":
        criterion = CrossEntropyLoss2d(
            ignore_index=cfg.DATASET.IGNORE_LABEL
        )
    elif args.loss_type == "weight_ce":
        criterion = ImageBasedCrossEntropyLoss2d(num_classes=cfg.DATASET.NUM_CLASSES,
                                                 ignore_index=cfg.DATASET.IGNORE_LABEL)
    elif args.loss_type == "rmi":
        criterion = RMILoss(num_classes=cfg.DATASET.NUM_CLASSES,
                            ignore_index=cfg.DATASET.IGNORE_LABEL)
    else:
        raise NotImplementedError("[*] loss {} is not implement.".format(args.loss_type))

    criterion_val = CrossEntropyLoss2d(weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL)

    if cuda:
        criterion = criterion.cuda()
        criterion_val = criterion_val.cuda()

    return criterion, criterion_val


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entropy NLL Loss
    """
    def __init__(self, weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL, reduction="mean"):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets, **kwargs):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """
    def __init__(self, num_classes, weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL, upper_bound=1.0,
                 norm=False, batch_weights=False):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = num_classes
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = batch_weights
        self.nll_loss = nn.NLLLoss(weight, reduction="mean", ignore_index=ignore_index)

    def calculate_weigths(self, targets):
        bins = torch.histc(targets, bins=self.num_classes, min=0.0, max=self.num_classes)
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound * (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound * (1. - hist_norm)) + 1.0

        return hist

    def forward(self, inputs, targets, **kwargs):

        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                if self.fp16:
                    weights = weights.half()
                self.nll_loss.weight = weights

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0), )
        return loss


if __name__ == '__main__':
    import easydict as edict
    args = edict.EasyDict({"loss_type": "ce"})
    criterion, criterion_val = get_loss(args=args)
    inputs = torch.sigmoid(torch.randn([1, 2, 5, 5]))
    targets = torch.ones([1, 5, 5]).long()
    print(inputs, targets)
    loss = criterion(inputs, targets)
    print(loss)




