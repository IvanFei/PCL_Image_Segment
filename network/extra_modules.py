import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# import network
from network.SEresnext import se_resnext50_32x4d, se_resnext101_32x4d
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from network.xception import xception71
from network.mobilenetv2 import mobilenetv2
from network.hrnetv2 import hrnetv2
from network.pnasnet import PNASNet5Large, PNASNet5Small, pnasnet5large, pnasnet5small
from network.efficientnet_base import EfficientNet, EncoderMixin


Norm2d = nn.BatchNorm2d


class get_resnet(nn.Module):
    def __init__(self, trunk_name, output_stride=8, pretrained=True):
        super(get_resnet, self).__init__()

        if trunk_name == "resnet18":
            resnet = resnet18(pretrained=pretrained)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == "resnet34":
            resnet = resnet34(pretrained=pretrained)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == "resnet50":
            resnet = resnet50(pretrained=pretrained)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == "resnet101":
            resnet = resnet101(pretrained=pretrained)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == "resnet152":
            resnet = resnet152(pretrained=pretrained)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == "se_resnet50":
            resnet = se_resnext50_32x4d(pretrained=pretrained)
        elif trunk_name == "se_resnet101":
            resnet = se_resnext101_32x4d(pretrained=pretrained)
        else:
            raise KeyError("[*] Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise KeyError("[*] Unsupported output_stride {}".format(output_stride))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        s2_features = x
        x = self.layer2(x)
        s4_features = x
        x = self.layer3(x)
        x = self.layer4(x)
        return s2_features, s4_features, x


def get_trunk(trunk_name, output_stride=8, pretrained=True):

    if trunk_name == "xception71":
        backbone = xception71(output_stride=output_stride, BatchNorm=Norm2d,
                              pretrained=pretrained)

        s2_ch, s4_ch = 64, 128
        high_level_ch = 2048
    elif trunk_name == "resnet50" or trunk_name == "resnet101" or trunk_name == "resnet152" \
        or trunk_name == "resnet18" or trunk_name == "resnet34":
        backbone = get_resnet(trunk_name, output_stride=output_stride, pretrained=pretrained)

        s2_ch, s4_ch = 256, -1
        high_level_ch = 2048
    elif trunk_name == "hrnetv2":
        backbone = hrnetv2(pretrained=pretrained)
        high_level_ch = backbone.high_level_ch
        s2_ch, s4_ch = -1, -1
    else:
        raise KeyError("[*] Unknown backbone {}".format(trunk_name))

    return backbone, s2_ch, s4_ch, high_level_ch


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
        1x1 x depth
        3x3 x depth dilation 6
        3x3 x depth dilation 12
        3x3 x depth dilation 18
        image pooling
        concatenate all together
        Final 1x1 conv
    """

    RATES = (6, 12, 18)

    def __init__(self, in_dim, reduction_dim=256, output_stride=16):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            self.rates = [2 * r for r in self.RATES]
        elif output_stride == 16:
            self.rates = self.RATES
        else:
            raise KeyError("[*] Output stride of {} not supported.".format(output_stride))

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True))
        )
        # other rates
        for r in self.rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.features = nn.ModuleList(self.features)

        # image level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, size=x_size[2:], mode="bilinear", align_corners=False)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        return out


def dpc_conv(in_dim, reduction_dim, dilation, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=dilation,
                  padding=dilation, bias=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True)
    )


class DPC(nn.Module):
    """
    From: Searching for Efficient Multi-scale architectures for dense
    prediction. NAS search
    """

    RATES = [(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)]

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            self.rates = [(2 * r[0], 2 * r[1]) for r in self.RATES]
        elif output_stride == 16:
            self.rates = self.RATES
        else:
            raise KeyError("[*] Output stride of {} not supported.".format(output_stride))

        self.a = dpc_conv(in_dim, reduction_dim, self.rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, self.rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, self.rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, self.rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, self.rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch, output_stride=output_stride)

    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, norm_layer=Norm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


def BNReLU(ch):
    return nn.Sequential(
        Norm2d(ch),
        nn.ReLU(inplace=True)
    )


def old_make_attn_head(in_ch, bot_ch, out_ch):
    attn = nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=out_ch, bias=False),
        nn.Sigmoid()
    )

    return attn


def mask_attn_head(in_ch, out_ch):
    bot_ch = 256
    od = OrderedDict([("conv0", nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False)),
                      ("bn0", Norm2d(bot_ch)),
                      ("re0", nn.ReLU(inplace=True))])

    od["conv1"] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False)
    od["bn1"] = Norm2d(bot_ch)
    od["re1"] = nn.ReLU(inplace=True)

    od["conv2"] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od["sig"] = nn.Sigmoid()

    attn_head = nn.Sequential(od)

    return attn_head


if __name__ == '__main__':
    backbone, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name="hrnetv2")
    print(backbone.state_dict().keys())
    print(s2_ch, s4_ch, high_level_ch)



