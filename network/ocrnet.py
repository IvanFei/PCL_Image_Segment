import torch
import torch.nn as nn
import torch.nn.functional as F

from network.utils import initialize_weights, scale_as, fmt_scale
from network.extra_modules import get_trunk, get_aspp, get_resnet, BNReLU, mask_attn_head
from network.ocr_utils import SpatialOCR_Module, SpatialGather_Module
from config import cfg


class OCR_block(nn.Module):

    MID_CHANNELS = 512
    KEY_CHANNELS = 256

    def __init__(self, num_classes, high_level_ch):
        super(OCR_block, self).__init__()

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, self.MID_CHANNELS, kernel_size=3, stride=1, padding=1),
            BNReLU(self.MID_CHANNELS)
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.MID_CHANNELS, key_channels=self.KEY_CHANNELS,
                                                 out_channels=self.MID_CHANNELS, scale=1, dropout=0.05)

        self.cls_head = nn.Conv2d(self.MID_CHANNELS, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch, kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        initialize_weights(self.conv3x3_ocr, self.ocr_distri_head, self.ocr_gather_head, self.cls_head, self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    """
    OCR Net
    """

    OCR_ALPHA = 0.4

    def __init__(self, num_classes, trunk="hrnetv2", criterion=None):
        super(OCRNet, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(num_classes, high_level_ch)

    def forward(self, inputs):
        assert "images" in inputs
        x = inputs["images"]

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs["gts"]
            loss = self.OCR_ALPHA * self.criterion(aux_out, gts) + \
                self.criterion(cls_out, gts)
            return loss
        else:
            output_dict = {"pred": cls_out}
            return output_dict


class OCRNetASPP(nn.Module):
    """
    OCR Net
    """

    OCR_ALPHA = 0.4

    def __init__(self, num_classes, trunk="hrnetv2", criterion=None):
        super(OCRNetASPP, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8)

        self.ocr = OCR_block(num_classes, aspp_out_ch)

    def forward(self, inputs):
        assert "images" in inputs
        x = inputs["images"]

        _, _, high_level_features = self.backbone(x)
        aspp = self.aspp(high_level_features)
        cls_out, aux_out, _ = self.ocr(aspp)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            loss = self.OCR_ALPHA * self.criterion(aux_out, gts) + self.criterion(cls_out, gts)
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict


class MscaleOCR(nn.Module):
    """
    OCR Net
    """

    OCR_ALPHA = 0.4
    MSCALE_LO_SCALE = 0.5
    MID_CHANNELS = 512
    SUPERVISED_MSCALE_WT = 0.05
    N_SCALES = [0.5, 1.0, 2.0]

    def __init__(self, num_classes, trunk="hrnetv2", criterion=None):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(num_classes, high_level_ch)
        self.scale_attn = mask_attn_head(in_ch=self.MID_CHANNELS, out_ch=1)

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = F.interpolate(aux_out, size=x_size, mode="bilinear", align_corners=False)
        cls_out = F.interpolate(cls_out, size=x_size, mode="bilinear", align_corners=False)
        attn = F.interpolate(attn, size=x_size, mode="bilinear", align_corners=False)

        return {"cls_out": cls_out,
                "aux_out": aux_out,
                "logit_attn": attn}

    def nsacle_foward(self, inputs, scales):
        x_1x = inputs["images"]

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        scales = sorted(scales, reverse=True)

        pred, aux, output_dict = None, None, {}

        for s in scales:
            x = F.interpolate(x_1x, scale_factor=s, mode="bilinear", align_corners=False)
            outs = self._fwd(x)
            cls_out, aux_out, attn_out = outs["cls_out"], outs["aux_out"], outs["logit_attn"]

            output_dict[fmt_scale("pred", s)] = cls_out
            if s != 2:
                output_dict[fmt_scale("attn", s)] = attn_out

            if pred is None:
                pred, aux = cls_out, aux_out
            elif s >= 1.0:
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            loss = self.OCR_ALPHA * self.criterion(aux, gts) + self.criterion(pred, gts)
            return loss
        else:
            output_dict["pred"] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        assert "images" in inputs
        x_1x = inputs["images"]

        x_lo = F.interpolate(x_1x, scale_factor=self.MSCALE_LO_SCALE, mode="bilinear", align_corners=False)
        lo_outs = self._fwd(x_lo)
        pred_05x, aux_lo, logit_attn = lo_outs["cls_out"], lo_outs["aux_out"], lo_outs["logit_attn"]
        p_lo, attn_05x = pred_05x, logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x, aux_1x = hi_outs["cls_out"], hi_outs["aux_out"]
        p_1x = pred_10x

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)
        logit_attn = scale_as(logit_attn, p_1x)

        # combine with hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs["gts"]
            aux_loss = self.OCR_ALPHA * self.criterion(joint_aux, gts)
            main_loss = self.criterion(joint_pred, gts, do_rmi=True)
            loss = aux_loss + main_loss

            if self.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += self.SUPERVISED_MSCALE_WT * loss_hi
                loss += self.SUPERVISED_MSCALE_WT * loss_lo

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
            return self.nsacle_foward(inputs, self.N_SCALES)

        return self.two_scale_forward(inputs)

    def load_pretrained_model(self, pth):
        weights = torch.load(pth)
        new_state_dict = {}
        for k, v in self.state_dict().items():
            if weights[k].shape == v.shape:
                new_state_dict[k] = weights[k]
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)


def HRNet(num_classes, criterion):
    return OCRNet(num_classes, trunk="hrnetv2", criterion=criterion)


def HRNet_Mscale(num_classes, criterion, pretrained=True):

    model = MscaleOCR(num_classes, trunk="hrnetv2", criterion=criterion)
    if pretrained:
        if "ocrnet_hrnet_mscale" in cfg.MODEL.BACKBONE_CHECKPOINTS.keys():
            model.load_pretrained_model(cfg.MODEL.BACKBONE_CHECKPOINTS["ocrnet_hrnet_mscale"])

    return model


if __name__ == '__main__':

    model = HRNet_Mscale(num_classes=8, criterion=None)
    print(model.state_dict().keys())




