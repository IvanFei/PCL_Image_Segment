import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from config import cfg
from loss.rmi import RMILoss
from utils.utils import get_class_weight


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
    elif args.loss_type == "sce":
        weights = get_class_weight()
        weights = torch.from_numpy(weights).float()
        if cuda:
            weights = weights.cuda()
        criterion = SymmetricCrossEntropyLoss2d(weights=weights, ignore_index=cfg.DATASET.IGNORE_LABEL)
    elif args.loss_type == "focal":
        criterion = FocalCrossEntropy(gamma=2.0, alpha=0.2, smooth=0.1)
    elif args.loss_type == "ohem":
        criterion = OHEMCrossEntropy(selected_ratio=0.5)
    elif args.loss_type == "combo":
        weights = get_class_weight()
        criterion = MultiComboLoss(num_classes=cfg.DATASET.NUM_CLASSES,
                                   alpha=0.5, ce_ratio=0.5, weights=weights,
                                   ignore_label=cfg.DATASET.IGNORE_LABEL)
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


class SymmetricCrossEntropyLoss2d(nn.Module):
    """
    Symmetric Cross Entropy Loss
    """
    def __init__(self, alpha=0.5, beta=0.6, weights=None, ignore_index=cfg.DATASET.IGNORE_LABEL):
        super(SymmetricCrossEntropyLoss2d, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    def forward(self, inputs, targets, **kwargs):

        if targets.dim() >= 2:
            targets = targets.squeeze().long()
        else:
            targets = targets.long()

        # CCE
        ce = self.cross_entropy(inputs, targets)

        # RCE
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        if not pred.is_cuda:
            pred = pred.to(self.device)
        label_one_hot = F.one_hot(targets, cfg.DATASET.NUM_CLASSES).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        return loss


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


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, smooth=0.1, weight=None, reduction="mean", **kwargs):
        super(FocalCrossEntropy, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        B, C, H, W = inputs.size()
        epsilon = 1e-7
        logit = F.softmax(inputs, dim=1)

        if logit.dim() > 2:
            # N, C, d1, d2 -> N, C, m
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        targets = targets.view(-1, 1)
        idx = targets.cpu().long()
        target_onehot = torch.zeros([targets.size(0), C]).scatter_(1, idx, 1)
        if target_onehot.device != inputs.device:
            target_onehot = target_onehot.to(inputs.device)

        if self.smooth:
            target_onehot = torch.clamp(target_onehot, self.smooth / (C - 1), 1. - self.smooth)

        pt = (target_onehot * logit).sum(1) + epsilon
        logpt = pt.log()

        loss = -1 * self.alpha * torch.pow((1 - pt), self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class OHEMCrossEntropy(nn.Module):
    def __init__(self, selected_ratio=0.5, **kwargs):
        super(OHEMCrossEntropy, self).__init__()
        self.selected_ratio = selected_ratio

        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        pixel_num = inputs.size(2) * inputs.size(3)
        n_selected = int(self.selected_ratio * batch_size * pixel_num)

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.permute(0, 2, 1).contiguous()
            inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, 1).squeeze(1)

        ce_loss = self.cross_entropy(inputs, targets)
        vals, _ = ce_loss.topk(n_selected, 0, True, True)
        th = vals[-1]
        selected_mask = ce_loss >= th
        loss_weight = selected_mask.float()
        loss = (ce_loss * loss_weight).sum() / loss_weight.sum()

        return loss


class BinaryComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, weight=None, size_average=True, **kwargs):
        super(BinaryComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1.0):
        # flatten label and prediction tensors
        eps = 1e-7
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (self.alpha * ((targets * torch.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)

        return combo


class MultiComboLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, ce_ratio=0.5, weight=None, 
                 ignore_label=cfg.DATASET.IGNORE_LABEL, **kwargs):
        super(MultiComboLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_label = ignore_label
        self.combo_loss = BinaryComboLoss(alpha=alpha, ce_ratio=ce_ratio)

    def forward(self, inputs, targets):
        total_loss = 0.0
        inputs = F.softmax(inputs, dim=1)
        for c in range(self.num_classes):
            loss = self.combo_loss(inputs[:, c:c+1].contiguous(), (targets == c).float())
            if self.weight is not None:
                loss *= self.weight[c]

            total_loss += loss

        return total_loss


if __name__ == '__main__':
    import easydict as edict
    args = edict.EasyDict({"loss_type": "combo"})
    criterion, criterion_val = get_loss(args=args)
    inputs = torch.sigmoid(torch.randn([1, 2, 5, 5]))
    targets = torch.ones([1, 5, 5]).long()
    print(inputs, targets)
    loss = criterion(inputs, targets)
    print(loss)




