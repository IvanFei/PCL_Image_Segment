import os
import re
import sys
import torch

sys.path.insert(0, "/nfs/users/huangfeifei/PCL_Image_Segment")
from utils.attr_dict import AttrDict
# from runx.logx import logx


__C = AttrDict()
cfg = __C

# A
__C.ASSETS_PATH = "/nfs/users/huangfeifei/dataset"
__C.WEIGHT_PATH = "/nfs/users/huangfeifei/pretrained_weights"
#

# OPTIONS
__C.OPTIONS = AttrDict()

# TRAIN
__C.TRAIN = AttrDict()

# DATASET
__C.DATASET = AttrDict()
# __C.DATASET.PCL_DIR = os.path.join(__C.ASSETS_PATH, "pcl_image_segment")
__C.DATASET.PCL_DIR = os.path.join(__C.ASSETS_PATH, "remote_sensing")

__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]
# __C.DATASET.MEAN = [0.355, 0.384, 0.359]
# __C.DATASET.STD = [0.411, 0.434, 0.416]

__C.DATASET.NUM_CLASSES = 17
__C.DATASET.IGNORE_LABEL = 17
# __C.DATASET.ID_TO_TRAINID = {100: 0, 200: 1, 300: 2, 400: 3, 500: 4, 600: 5, 700: 6, 800: 7}
__C.DATASET.ID_TO_TRAINID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
                             12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16}
# __C.DATASET.TRAINID_TO_ID = {0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600, 6: 700, 7: 800}
__C.DATASET.TRAINID_TO_ID = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13,
                             13: 14, 14: 15, 15: 16, 16: 17}

__C.DATASET.CLASS_UNIFORM_PCT = 0.5
__C.DATASET.FILTER_DATA = False
__C.DATASET.ONE_CLASS_FILTER_PCT = 0.2

# MODEL
__C.MODEL = AttrDict()

# Checkpoint
__C.MODEL.BACKBONE_CHECKPOINTS = {
    "resnet50": os.path.join(__C.WEIGHT_PATH, "resnet50.pth"),
    "resnet101": os.path.join(__C.WEIGHT_PATH, "resnet101.pth"),
    "resnet152": os.path.join(__C.WEIGHT_PATH, "resnet152.pth"),
    "resnext101_32x8d": os.path.join(__C.WEIGHT_PATH, "resnext101_32x8d-8ba56ff5.pth"),
    "xcption": os.path.join(__C.WEIGHT_PATH, "xception-43020ad28.pth"),
    "mobilenetv2": os.path.join(__C.WEIGHT_PATH, "mobilenet_v2.pth.tar"),
    "hrnetv2": os.path.join(__C.WEIGHT_PATH, "hrnetv2_w48-imagenet.pth"),
    "xception71": os.path.join(__C.WEIGHT_PATH, "seg_weights/aligned_xception71.pth"),
    "ocrnet_hrnet_mscale": os.path.join(__C.WEIGHT_PATH, "seg_weights/ocrnet.HRNet_Mscale.pth")
}

# LOSS
__C.LOSS = AttrDict()
__C.LOSS.LOSS_TYPE = "ce"









