import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def update_bn(loader, model, cuda=False):
    if not check_bn(model):
        return

    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    if cuda:
        model = model.cuda()

    n = 0
    for data_batch in tqdm(loader, desc="update bn"):
        images, targets = data_batch[0], data_batch[1]
        if cuda:
            images = images.cuda(async=True)
            targets = targets.cuda(async=True)
        b = images.size(0)

        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum

        inputs = {"images": images, "gts": targets}
        model(inputs)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))














