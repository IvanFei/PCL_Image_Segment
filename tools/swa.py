import os
import sys
import pprint
import glob
import argparse
import json

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.nn.functional as F

from utils.utils import cfg_parser
import utils.swa as swa
import utils.checkpoints as util_checkpoint
from transforms.get_transforms import get_transforms
from datasets.pcl_dataset import DataSet
from network.network_factory import get_net
from loss.loss_factory import get_loss
from train import validation


ROOT_DIR = "/nfs/users/huangfeifei/PCL_Image_Segment"
LOG_DIR = "final_logs"


def get_checkpoints(config, num_checkpoints=5):
    checkpoint_dir = os.path.join(ROOT_DIR, LOG_DIR, os.path.basename(config.model_dir))
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    checkpoints.sort()
    checkpoints = checkpoints[-num_checkpoints:]
    return checkpoints


def run(config, num_checkpoints, cuda=False):

    train_joint_transform_list, train_img_transform, train_label_transform = get_transforms(config, mode="train")
    val_joint_transform_list, val_img_transform, val_label_transform = None, None, None

    train_dataset = DataSet(mode="train", joint_transform_list=train_joint_transform_list,
                            img_transform=train_img_transform, label_transform=train_label_transform)
    val_dataset = DataSet(mode="val", joint_transform_list=val_joint_transform_list,
                          img_transform=val_img_transform, label_transform=val_label_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                   num_workers=config.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)

    criterion, val_criterion = get_loss(config, cuda=cuda)

    model = get_net(config, criterion, cuda=cuda)

    checkpoints = get_checkpoints(config, num_checkpoints)
    print("[*] Checkpoints as follow:")
    pprint.pprint(checkpoints)

    util_checkpoint.load_checkpoint(model, None, checkpoints[0])
    for i, checkpoint in enumerate(checkpoints[1:]):
        model2 = get_net(config, criterion, cuda=cuda)

        util_checkpoint.load_checkpoint(model2, None, checkpoint)
        swa.moving_average(model, model2, 1. / (i + 2))

    with torch.no_grad():
        swa.update_bn(train_loader, model, cuda=cuda)

    output_name = "model-swa.pth"
    print(f"[*] SAVED: to {output_name}")
    checkpoint_dir = os.path.join(ROOT_DIR, LOG_DIR, os.path.basename(config.model_dir))
    util_checkpoint.save_checkpoint(checkpoint_dir, output_name, model)

    # test the model
    scores = validation(config, val_loader, model, val_criterion, "swa", cuda=cuda, is_record=False)
    print(scores)
    with open(os.path.join(checkpoint_dir, "swa-scores.json"), "w") as f:
        json.dump(scores["FWIOU"], f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str,
                        default="/nfs/users/huangfeifei/PCL_Image_Segment/final_logs/DeeperX71_ASPP_CE_Adam_Cleaner_Poly/params.json")
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--cuda", type=bool, default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.cfg_path is None:
        raise Exception("[*] Please input the config file.")

    config = cfg_parser(args.cfg_path, None)
    pprint.PrettyPrinter(indent=2).pprint(config)
    run(config, args.num_checkpoints, args.cuda)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()


