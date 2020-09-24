import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch


Norm2d = nn.BatchNorm2d


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def scale_as(x, y):
    """
    Scale x to the same size as y
    """
    y_size = y.size(2), y.size(3)

    x_scaled = F.interpolate(x, size=y_size, mode="bilinear", align_corners=False)

    return x_scaled


def fmt_scale(prefix, scale):
    """
        format scale name

        :prefix: a string that is the beginning of the field name
        :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """
    scale_str = str(float(scale))
    scale_str.replace(".", "")
    return f"{prefix}_{scale_str}x"



