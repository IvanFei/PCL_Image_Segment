import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeper import DeeperX71
from .config import cfg


def init_model():
    is_cuda = torch.cuda.is_available()
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    model = DeeperX71(cfg.DATASET.NUM_CLASSES)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["state_dict"])
    if is_cuda:
        model.cuda()
    model.eval()

    return model


