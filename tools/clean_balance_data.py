import os
import glob
import random
import json
import torch
import cv2
import argparse
from tqdm import tqdm
from torch.utils import data
from easydict import EasyDict as edict

from config import cfg
from datasets.pcl_dataset import DataSet
from loss.loss_factory import get_loss, MultiIouLoss, CrossEntropyLoss2d
from network.network_factory import get_net
from evaluation.evaluator import SegmentMeter, AverageMeter
from utils.utils import get_logger, cfg_parser, save_args, prepare_dirs, write_mask, load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)  # ugly coding, don't change to False, it dose't work.
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    return args


# Note: set batch size to 1.
def infer_train_info(model, dataloader, criterions, cuda=False):
    model.eval()
    pbar = tqdm(total=len(dataloader), desc="Infer Of Train")
    records = {}

    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            image, target, img_name = data_batch
            records[img_name[0]] = {}
            if cuda:
                image, target = image.cuda(), target.cuda()

            inputs = {"images": image}
            pred = model(inputs)["pred"]

            for k, v in criterions.items():
                records[img_name[0]][k] = v(pred, target).item()

            pbar.update(1)

    return records


def main(args):
    train_dataset = DataSet(mode="train", uniform_sampling=False, filter_data=False, joint_transform_list=None,
                            img_transform=None, label_transform=None)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                   num_workers=8, drop_last=False)

    ce_criterion = CrossEntropyLoss2d()
    iou_criterion = MultiIouLoss(cfg.DATASET.NUM_CLASSES)
    criterions = {"ce_loss": ce_criterion, "iou_loss": iou_criterion}
    model = get_net(args, ce_criterion, cuda=args.cuda)

    if not args.load_path:
        raise KeyError("[*] Please input the load path.")

    if not args.load_path.endswith("pth"):
        raise ValueError("[*] load path should be end with pth")

    checkpoints = torch.load(args.load_path)

    model.load_state_dict(checkpoints["state_dict"])

    records = infer_train_info(model, train_loader, criterions, cuda=args.cuda)

    output_file = os.path.join(args.output_dir, "train_loss_overview.json")
    with open(output_file, "w") as f:
        json.dump(records, f)


if __name__ == '__main__':
    args = get_args()
    main(args)








