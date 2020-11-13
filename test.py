import os
import sys
import json
import torch
import cv2
import argparse
from tqdm import tqdm
from torch.utils import data
from easydict import EasyDict as edict

from config import cfg
from datasets.pcl_dataset import DataSet
from loss.loss_factory import get_loss
from network.network_factory import get_net
from evaluation.evaluator import SegmentMeter, AverageMeter
from utils.utils import get_logger, cfg_parser, save_args, prepare_dirs, write_mask


logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, required=True)
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):

    test_joint_transform_list, test_img_transform, test_label_transform = None, None, None

    test_dataset = DataSet(mode=args.mode, joint_transform_list=test_joint_transform_list,
                           img_transform=test_img_transform, label_transform=test_label_transform)

    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info("[*] Initial the test loader.")

    model = get_net(args, criterion=None, cuda=args.cuda)

    if args.load_path:
        if not (os.path.isfile(args.load_path) and args.load_path.endswith(".pth")):
            raise ValueError("[*] The `load_path` should be exists and end with pth.")

        state_dict = torch.load(args.load_path)
        model.load_state_dict(state_dict["state_dict"])

        logger.info(f"[*] LOADED successfully checkpoints from: {args.load_path}")
    else:
        raise ValueError("[*] The `load_path` should not be None.")

    test(args, test_loader, model, cuda=args.cuda)


def test(args, dataloader, model, cuda=False):
    model.eval()

    pbar = tqdm(total=len(dataloader), desc="test_model")
    segmeter = SegmentMeter(num_class=cfg.DATASET.NUM_CLASSES)
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            if args.mode == "test":
                images, img_names = data_batch
            else:
                images, masks, img_names = data_batch

            if cuda:
                images = images.cuda()
            inputs = {"images": images}

            preds = model(inputs)

            preds = preds["pred"]
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)

            preds = preds.cpu().numpy()

            if args.mode != "test":
                masks = masks.numpy()
                segmeter.add_batch(masks, preds)

            write_mask(args, preds, img_names, save_dir=args.save_dir)

            pbar.update(1)

    if args.mode != "test":
        scores = {}
        scores["PA"] = segmeter.Pixel_Accuracy()
        scores["MPA"] = segmeter.Mean_Pixel_Accuracy()
        scores["MIOU"] = segmeter.Mean_Intersection_over_Union()
        scores["FWIOU"] = segmeter.Frequency_Weighted_Intersection_over_Union()
        scores["IOU"] = segmeter.Intersection_over_Union()
        logger.info((f"| model_name {args.model_name} | PA: {scores['PA']} "
                     f"| mPA: {scores['MPA']} | mIoU: {scores['MIOU']} | FWIoU: {scores['FWIOU']}"))

        logger.info(f"| model_name {args.model_name} | IoU: {scores['IOU']}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()
    load_path = args.load_path
    mode = args.mode
    save_dir = args.save_dir
    with open(args.param_file, "r") as f:
        args = json.load(f)

    args = edict(args)
    args.load_path = load_path
    args.mode = mode
    args.save_dir = save_dir

    main(args)


