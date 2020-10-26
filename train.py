import os
import sys
import time
import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils import data

from config import cfg
from evaluation.evaluator import AverageMeter, SegmentMeter
from utils.utils import prepare_dirs, cfg_parser, get_logger, save_args, TensorBoard, save_checkpoint, get_learning_rate
from datasets.pcl_dataset import DataSet
from network.network_factory import get_model, get_net
from loss.loss_factory import get_loss
from optimizer.optimizer_factory import get_optimizer
from transforms.get_transforms import get_transforms


logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2020)

    cfg = parser.add_argument_group("Config")
    cfg.add_argument("--cfg_path", type=str, default=None)
    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", type=str, default="PCL")
    data.add_argument("--num_workers", type=int, default=8)
    data.add_argument("--uniform_sampling", action="store_true")
    data.add_argument("--crop_size", type=int, default=256)
    data.add_argument("--scale_min", type=float, default=1.0)
    data.add_argument("--scale_max", type=float, default=1.5)
    data.add_argument("--data_filter", action="store_true")

    net = parser.add_argument_group("Net")
    net.add_argument("--load_path", type=str, default="")
    net.add_argument("--arch", type=str, required=True)
    net.add_argument("--model_name", type=str, required=True)

    file = parser.add_argument_group("File")
    file.add_argument("--log_dir", type=str, default="./final_logs")

    optimizer = parser.add_argument_group("Optimizer")
    optimizer.add_argument("--lr", type=float, default=0.01)
    optimizer.add_argument("--optim", type=str, default="sgd")
    optimizer.add_argument("--weight_decay", type=float, default=1e-4)
    optimizer.add_argument("--momentum", type=float, default=0.9)

    schedule = parser.add_argument_group("Scheduler")
    schedule.add_argument("--lr_schedule", type=str, default="reduce_lr_on_plateau")
    schedule.add_argument("--step_size", type=int, default=30)
    schedule.add_argument("--gamma", type=float, default=0.1)
    schedule.add_argument("--poly_exp", type=float, default=2)

    loss = parser.add_argument_group("Loss")
    loss.add_argument("--loss_type", type=str, default="ce")

    train = parser.add_argument_group("Train")
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--log_step", type=int, default=50)
    train.add_argument("--num_epochs", type=int, default=100)
    train.add_argument("--num_steps", type=int, default=3e5)  # for lr schedule
    train.add_argument("--val_freq", type=int, default=1000)
    train.add_argument("--retrain", action="store_true")

    gpu = parser.add_argument_group("GPU")
    gpu.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    return args


def main(args):
    # train_joint_transform_list, train_img_transform, train_label_transform = None, None, None
    # val_joint_transform_list, val_img_transform, val_label_transform = None, None, None
    train_joint_transform_list, train_img_transform, train_label_transform = get_transforms(args, mode="train")
    val_joint_transform_list, val_img_transform, val_label_transform = None, None, None

    train_dataset = DataSet(mode="train", uniform_sampling=args.uniform_sampling,
                            filter_data=args.data_filter, joint_transform_list=train_joint_transform_list,
                            img_transform=train_img_transform, label_transform=train_label_transform)
    val_dataset = DataSet(mode="val", uniform_sampling=False, joint_transform_list=val_joint_transform_list,
                          img_transform=val_img_transform, label_transform=val_label_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    logger.info("[*] Initial the train loader and val loader.")

    criterion, val_criterion = get_loss(args, cuda=args.cuda)
    # logger.info("[*] Loaded the criterion.")

    model = get_net(args, criterion, cuda=args.cuda)
    logger.info("[*] Loaded the model.")

    optimizer, lr_scheduler = get_optimizer(args, model)

    # minloss, maxscore
    maxscore = 0

    if args.load_path:
        if not (os.path.isfile(args.load_path) and args.load_path.endswith("pth")):
            raise ValueError("[*] The `load_path` should be exists and end with pth.")

        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint["state_dict"])
        if not args.retrain:
            start_epoch = checkpoint["epoch"]
            start_step = checkpoint["step"]

            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            start_epoch, start_step = 0, 0

        logger.info(f"[*] LOADED successfully checkpoints from: {args.load_path}")

    else:
        start_epoch, start_step = 0, 0

    tb = TensorBoard(args.model_dir)
    step = start_step

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"| model_name {args.model_name} | epoch {epoch} | lr {args.lr}")

        train_loss, step, maxscore = train(args, train_loader, val_loader, model, val_criterion, optimizer,
                                           lr_scheduler, epoch, step, tb, maxscore, cuda=args.cuda)
        if args.uniform_sampling or args.data_filter:
            train_loader.dataset.build_epoch()


def train(args, train_loader, val_loader, model, val_criterion, optimizer, lr_schedule, epoch, step, tb, max_score, cuda=False):
    """
    Runs the training loop per epoch.
    dataloader: Data loader for train
    args: args
    net: network
    optimizer: optimizer
    cur_epoch: current epoch
    cuda: use gpu or not.
    """
    model.train()

    train_loss = AverageMeter()
    pbar = tqdm(total=len(train_loader), desc="train_model")

    for idx, data_batch in enumerate(train_loader):
        images, targets, img_names = data_batch
        if cuda:
            images, targets = images.cuda(), targets.cuda()

        inputs = {"images": images, "gts": targets}
        loss = model(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), n=1)

        if (step + 1) % args.log_step == 0:
            tb.scalar_summary("model/loss", loss.data, step)
            tb.scalar_summary("model/lr", args.lr, step)

        pbar.set_description(desc=f"train_model| loss: {loss.item():5.3f}")

        if (step + 1) % args.val_freq == 0:
            val_scores = validation(args, val_loader, model, val_criterion, step, cuda=args.cuda)

            logger.info(f"| model_name {args.model_name} | step: {step} | PA: {val_scores['PA']} "
                        f"| mPA: {val_scores['MPA']} | mIoU: {val_scores['MIOU']} | FWIoU: {val_scores['FWIOU']}")

            tb.scalar_summary("val/PA", val_scores["PA"], step)
            tb.scalar_summary("val/mPA", val_scores["MPA"], step)
            tb.scalar_summary("val/mIoU", val_scores["MIOU"], step)
            tb.scalar_summary("val/FWIoU", val_scores["FWIOU"], step)

            is_best = val_scores["FWIOU"] >= max_score
            max_score = max(max_score, val_scores["FWIOU"])

            logger.info(f"[*] Step: {step}, max_score: {max_score}.")

            lr_schedule.step(val_scores["FWIOU"])
            args.lr = get_learning_rate(optimizer)[0]

            state_dict = {
                "epoch": epoch,
                "step": step,
                "state_dict": model.state_dict(),
                "max_score": max_score,
                "optimizer": optimizer.state_dict()
            }

            save_checkpoint(state_dict, step, is_best, args)

        step += 1
        pbar.update(1)

    return train_loss.avg, step, max_score


def validation(args, dataloader, model, criterion, step, cuda=False, is_record=True):
    """
    Runs the validation loops
    """
    model.eval()
    pbar = tqdm(total=len(dataloader), desc="validate_model")
    segmeter = SegmentMeter(num_class=cfg.DATASET.NUM_CLASSES)
    scores = {}
    loss = AverageMeter()
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            images, targets, img_names = data_batch
            if cuda:
                images, targets = images.cuda(), targets.cuda()

            inputs = {"images": images, "gts": targets}
            preds = model(inputs)["pred"]
            cur_loss = criterion(preds, targets)
            loss.update(cur_loss, n=1)
            preds = preds.detach().cpu().numpy()
            argpreds = np.argmax(preds, axis=1)
            targets = targets.cpu().numpy()
            segmeter.add_batch(targets, argpreds)

            pbar.update(1)

    scores["loss"] = loss.avg
    scores["PA"] = segmeter.Pixel_Accuracy()
    scores["MPA"] = segmeter.Mean_Pixel_Accuracy()
    scores["MIOU"] = segmeter.Mean_Intersection_over_Union()
    scores["FWIOU"] = segmeter.Frequency_Weighted_Intersection_over_Union()

    # save the scores
    if is_record:
        checkpoint_tracker_path = os.path.join(args.model_dir, "checkpoint_tracker.json")
        if os.path.exists(checkpoint_tracker_path):
            with open(checkpoint_tracker_path, "r") as f:
                checkpoint_tracker = json.load(f)
        else:
            checkpoint_tracker = {}

        checkpoint_tracker["step_{}".format(step)] = scores["FWIOU"]

        with open(checkpoint_tracker_path, "w") as f:
            json.dump(checkpoint_tracker, f)

    model.train()

    return scores


if __name__ == '__main__':
    args = get_args()
    if args.gpu_id < 0:
        setattr(args, "cuda", False)
    else:
        setattr(args, "cuda", True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    if args.cfg_path is not None:
        logger.info(f"[*] Note: cfg path is not None, read args from config - {args.cfg_path}")
        args = cfg_parser(args.cfg_path, args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    prepare_dirs(args)
    save_args(args)

    main(args)







