import torch
from torch import nn
import torch.nn.functional as F

from network.extra_modules import get_aspp, get_trunk, ConvBnRelu, mask_attn_head
from network.utils import scale_as, fmt_scale


class DeeperS8(nn.Module):
    """
    Panoptic DeepLab-style semantic segmentation network
    stride8 only
    """
    def __init__(self, num_classes, trunk="xception71", criterion=None, dpc=False):
        super(DeeperS8, self).__init__()

        self.criterion = criterion
        self.backbone, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk, output_stride=8)
        # TODO: DPC modules
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8, dpc=dpc)

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


class MscaleDeeperS8(nn.Module):
    """Deeper S8 + Mscale and aux classifier
       stride 8 only.
    """

    MSCALE_LO_SCALE = 0.875
    SUPERVISED_MSCALE_WT = 0.05
    N_SCALES = [0.875, 1.0, 1.125]

    def __init__(self, num_classes, trunk="xception71", criterion=None, dpc=False):
        super(MscaleDeeperS8, self).__init__()

        self.criterion = criterion
        self.backbone, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk, output_stride=8)

        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8, dpc=dpc)

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        # Change to DeeperS8
        # self.conv_up1 = nn.Conv2d(aspp_out_ch, 256, kernel_size=3, padding=1, bias=False)
        # self.conv_up2 = ConvBnRelu(256 + 64, 256, kernel_size=3, padding=1)
        # self.conv_up3 = ConvBnRelu(256 + 32, 256, kernel_size=3, padding=1)
        # self.conv_up5 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.conv_up2 = ConvBnRelu(256 + 64, 256, kernel_size=5, padding=2)
        self.conv_up3 = ConvBnRelu(256 + 32, 256, kernel_size=5, padding=2)
        self.conv_up5 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        self.scale_attn = mask_attn_head(in_ch=256, out_ch=1, bot_ch=128)

    def _fwd(self, x):
        x_size = x.size()[2:]

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
        cls_feat = self.conv_up3(x)
        x = self.conv_up5(cls_feat)
        cls_out = F.interpolate(x, size=x_size, mode="bilinear", align_corners=False)

        attn = self.scale_attn(cls_feat)

        attn = F.interpolate(attn, size=x_size, mode="bilinear", align_corners=False)

        return {"cls_out": cls_out,
                "logit_attn": attn}

    def nscale_forward(self, inputs, scales):
        assert "images" in inputs
        x_1x = inputs["images"]

        assert 1.0 in scales, "expected 1.0 to be the target scale."
        scales = sorted(scales, reverse=True)

        pred, output_dict = None, {}

        for s in scales:
            x = F.interpolate(x_1x, scale_factor=s, mode="bilinear", align_corners=False)
            outs = self._fwd(x)
            cls_out, attn_out = outs["cls_out"], outs["logit_attn"]
            output_dict[fmt_scale("pred", s)] = cls_out
            if s != 2:
                output_dict[fmt_scale("attn", s)] = attn_out

            if pred is None:
                pred = cls_out
            elif s >= 1.0:
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
            else:
                cls_out = attn_out * cls_out
                cls_out = scale_as(cls_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            loss = self.criterion(pred, gts)
            return loss
        else:
            output_dict["pred"] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        assert "images" in inputs
        x_1x = inputs["images"]

        x_lo = F.interpolate(x_1x, scale_factor=self.MSCALE_LO_SCALE, mode="bilinear", align_corners=False)
        lo_outs = self._fwd(x_lo)

        pred_05x, logit_attn = lo_outs["cls_out"], lo_outs["logit_attn"]
        p_lo, attn_05x = pred_05x, logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs["cls_out"]
        p_1x = pred_10x

        p_lo = logit_attn * p_lo
        p_lo = scale_as(p_lo, p_1x)
        logit_attn = scale_as(logit_attn, p_1x)

        # combine with hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        if self.training:
            gts = inputs["gts"]
            loss = self.criterion(joint_pred, gts)

            if self.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts)
                loss_hi = self.criterion(pred_10x, gts)
                loss += self.SUPERVISED_MSCALE_WT * loss_lo + self.SUPERVISED_MSCALE_WT * loss_hi

            return loss
        else:
            output_dict = {
                "pred": joint_pred,
                "pred_05x": pred_05x,
                "pred_10x": pred_10x,
                "attn_05x": attn_05x
            }
            return output_dict

    def forward(self, inputs):

        if self.N_SCALES and not self.training:
            return self.nscale_forward(inputs, self.N_SCALES)

        return self.two_scale_forward(inputs)

    def load_pretrained_weight(self, weight_pth="/nfs/users/huangfeifei/PCL_Image_Segment/final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth"):
        weight = torch.load(weight_pth)
        state_dict = weight["state_dict"]
        new_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k in new_state_dict:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)


def DeeperX71(num_classes, criterion=None):

    return DeeperS8(num_classes, criterion=criterion, trunk="xception71")


def DeeperX71_DPC(num_classes, criterion=None):

    return DeeperS8(num_classes, criterion=criterion, trunk="xception71", dpc=True)


def DeeperRX101(num_classes, criterion=None):
    pass


def DeeperX71_ASPP_Mscale(num_classes, criterion=None):

    return MscaleDeeperS8(num_classes, criterion=criterion, trunk="xception71")


def DeeperX71_DPC_Mscale(num_classes, criterion=None):

    return MscaleDeeperS8(num_classes, criterion=criterion, trunk="xception71", dpc=True)


if __name__ == '__main__':
    model = DeeperX71_ASPP_Mscale(num_classes=15)

    print(model.state_dict().keys())

    weight_pth = "/nfs/users/huangfeifei/PCL_Image_Segment/final_logs/DeeperX71_ASPP_CE_Adam/model-step-300999.pth"
    state_dict = torch.load(weight_pth)["state_dict"]
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)


