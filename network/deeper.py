import torch
from torch import nn
import torch.nn.functional as F
from network.extra_modules import get_aspp, get_trunk, ConvBnRelu


class DeeperS8(nn.Module):
    """
    Panoptic DeepLab-style semantic segmentation network
    stride8 only
    """
    def __init__(self, num_classes, trunk="xception71", criterion=None):
        super(DeeperS8, self).__init__()

        self.criterion = criterion
        self.backbone, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk, output_stride=8)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8)

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.conv_up2 = ConvBnRelu(256 + 64, 256, kernel_size=5, padding=2)
        self.conv_up3 = ConvBnRelu(256 + 32, 256, kernel_size=5, padding=2)
        self.conv_up5 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['images']
        else:
            x = inputs

        s2_features, s4_features, final_features = self.backbone(x)
        s2_features = self.convs2(s2_features)
        s4_features = self.convs4(s4_features)
        aspp = self.aspp(final_features)
        x = self.conv_up1(aspp)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s4_features], 1)
        x = self.conv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s2_features], 1)
        x = self.conv_up3(x)
        x = self.conv_up5(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(x, gts)
        else:
            return {"pred": x}
            # return x


def DeeperX71(num_classes, criterion=None):

    return DeeperS8(num_classes, criterion=criterion, trunk="xception71")


def DeeperRX101(num_classes, criterion=None):
    pass


if __name__ == '__main__':
    model = DeeperS8(num_classes=80)

    print(model.state_dict().keys())



