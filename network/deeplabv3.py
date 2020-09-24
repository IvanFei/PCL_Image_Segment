import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from network.extra_modules import get_trunk, get_aspp
from network.utils import initialize_weights


Norm2d = nn.BatchNorm2d


class DeepV3Plus(nn.Module):
    def __init__(self, num_classes, trunk="resnet101", criterion=None,
                 use_dpc=False, init_all=False):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8, dpc=use_dpc)

        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, inputs):
        """
        Args:
            inputs: dict, e.g. {'images': ..., 'gts': ...}
        Returns:
            dict: {'pred': ...}
        """
        assert "images" in inputs
        x = inputs['images']

        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = F.interpolate(conv_aspp, size=s2_features.size()[2:], mode="bilinear", align_corners=False)
        cat_s4 = torch.cat([conv_s2, conv_aspp], 1)
        final = self.final(cat_s4)
        out = F.interpolate(final, size=x_size[2:], mode="bilinear", align_corners=False)

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)
        else:
            return {'pred': out}


def DeepV3PlusR50(num_classes, criterion=None):

    return DeepV3Plus(num_classes, trunk="resnet50", criterion=criterion)


def DeepV3PlusR101(num_classes, criterion=None):

    return DeepV3Plus(num_classes, trunk="resnet101", criterion=criterion)


def DeepV3PlusR152(num_classes, criterion=None):

    return DeepV3Plus(num_classes, trunk="resnet152", criterion=criterion)


def DeepV3PlusX71(num_classes, criterion=None):

    return DeepV3Plus(num_classes, trunk="xception71", criterion=criterion)


if __name__ == '__main__':

    model = DeepV3PlusR50(num_classes=80)

    print(model.state_dict().keys())





