import os
import cv2
import torch
import logging
import json
import shutil
import numpy as np

from config import cfg
from easydict import EasyDict as edict
import tensorboardX as tb
from tensorboardX.summary import Summary


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, "_init_done__", None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def prepare_dirs(args):
    if args.model_name:
        if os.path.exists(os.path.join(args.log_dir, args.model_name)):
            raise FileExistsError(f"Model {args.model_name} already exists!!! give a different name.")

    if not hasattr(args, "model_dir"):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    for path in [args.log_dir, args.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def cfg_parser(cfg_path, args):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[*] Error: the `cfg_path` is not exists. - {cfg_path}")
    if not cfg_path.endswith("json"):
        raise Exception(f"[*] Error: the `cfg_path` should be end with json.")

    with open(cfg_path, "r") as f:
        params = json.load(f)

    params = edict(params)
    if args:
        args = vars(args)
        for k, v in args.items():
            if k not in params.keys():
                params[k] = v
    return params


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info(f"[*] MODEL dir: {args.model_dir}")
    logger.info(f"[*] PARAM path: {param_path}")

    with open(param_path, "w") as fp:
        args_dict = args.__dict__
        json.dump(args_dict, fp, indent=4, sort_keys=True)


class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)


def save_checkpoint(state, step, is_best, args):
    filepath = os.path.join(args.model_dir, "model-step-new.pth")
    torch.save(state, filepath)
    logger.info(f"[*] SAVED successfully to {filepath}")
    if is_best:
        cpy_file = os.path.join(args.model_dir, "model-best.pth")
        shutil.copyfile(filepath, cpy_file)
        logger.info(f"[*] SAVED successfully best model at {cpy_file}")


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group["lr"]]

    return lr


def write_mask(args, masks, img_names, save_dir="prediction"):
    res_dir = os.path.join(args.model_dir, save_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    batch_size, H, W = masks.shape
    for i in range(batch_size):
        img_name = img_names[i]
        mask = masks[i]
        seg = np.zeros([H, W]) * cfg.DATASET.IGNORE_LABEL
        for c in range(cfg.DATASET.TRAINID_TO_ID.keys()):
            seg[mask == c] = cfg.DATASET.TRAINID_TO_ID[c]

        save_mask =seg.astype(np.uint16)

        cv2.imwrite(os.path.join(res_dir, img_name + ".png"), save_mask)


