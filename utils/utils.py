import os
import cv2
import glob
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
    else:
        if args.load_path:
            if args.load_path.endswith(".pth"):
                args.model_dir = args.load_path.rsplit("/", 1)[0]
            else:
                args.model_dir = args.load_path

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


def save_model(state, step, args, save_criteria_score=None, max_save_num=5, save_criterion="FWIOU"):
    filepath = os.path.join(args.model_dir, f"model-step-{step}.pth")
    torch.save(state, filepath)
    logger.info(f"[*] SAVED: {filepath}")
    steps = get_save_models_info(args)

    if save_criteria_score is not None:
        checkpoint_path = os.path.join(args.model_dir, "checkpoint_tracker.dat")
        if os.path.exists(checkpoint_path):
            checkpoint_tracker = torch.load(checkpoint_path)
        else:
            checkpoint_tracker = {}
        key = f"step_{step}"
        value = save_criteria_score
        checkpoint_tracker[key] = value
        if len(checkpoint_tracker) > max_save_num:
            low_value = 10000.0
            remove_key = None
            for key, value in checkpoint_tracker.items():
                if low_value > value[save_criterion]:
                    remove_key = key
                    low_value = value[save_criterion]

            del checkpoint_tracker[remove_key]

            remove_step = remove_key.split("_")[1]
            paths = glob.glob(os.path.join(args.model_dir, f"model-step-{remove_step}.pth"))
            for path in paths:
                remove_file(path)

        torch.save(checkpoint_tracker, checkpoint_path)
    else:
        for st in steps[:-max_save_num]:
            paths = glob.glob(os.path.join(args.model_dir, f"model-step-{st}.pth"))
            for path in paths:
                remove_file(path)


def load_model(args, save_criterion="FWIOU"):

    if args.load_path.endswith(".pth"):
        checkpoints = torch.load(args.load_path)
        logger.info(f"[*] LOADED: {args.load_path}")
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoint_tracker.dat")
        if os.path.exists(checkpoint_path):
            checkpoint_tracker = torch.load(checkpoint_path)
            best_key = None
            best_score = -10000.0
            for key, value in checkpoint_tracker.items():
                if best_score < value[save_criterion]:
                    best_score = value[save_criterion]
                    best_key = key
            step = int(best_key.split("_")[1])
        else:
            steps = get_save_models_info(args)

            if len(steps) == 0:
                logger.warning(f"[!] No checkpoint found in {args.model_dir}")
                return

            step = max(steps)

        load_path = f"{args.load_path}/model-step-{step}.pth"
        logger.info(f"[*] LOADED: {load_path}")
        checkpoints = torch.load(load_path)

    return checkpoints


def get_save_models_info(args):
    paths = glob.glob(os.path.join(args.model_dir, "*.pth"))
    paths.sort()
    steps = []

    for path in paths:
        basename = os.path.basename(path.rsplit(".", 1)[0])
        step = int(basename.split("-")[2])
        steps.append(step)

    steps.sort()

    return steps


def remove_file(path):
    if os.path.exists(path):
        logger.info(f"[*] REMOVED: {path}")
        os.remove(path)


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
        for c in cfg.DATASET.TRAINID_TO_ID.keys():
            seg[mask == c] = cfg.DATASET.TRAINID_TO_ID[c]

        save_mask = seg.astype(np.uint16)

        cv2.imwrite(os.path.join(res_dir, img_name + ".png"), save_mask)


def write_pred(args, probs, preds, img_names, save_dir="prediction"):
    res_dir = os.path.join(args.model_dir, save_dir)
    mask_dir = os.path.join(res_dir, "mask")
    pred_dir = os.path.join(res_dir, "probs")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    batch_size, H, W = preds.shape
    for i in range(batch_size):
        img_name = img_names[i]
        prob, mask = probs[i], preds[i]
        seg = np.zeros([H, W]) * cfg.DATASET.IGNORE_LABEL
        for c in cfg.DATASET.TRAINID_TO_ID.keys():
            seg[mask == c] = cfg.DATASET.TRAINID_TO_ID[c]

        np.save(os.path.join(pred_dir, img_name + ".npy"), prob)
        save_mask = seg.astype(np.uint16)
        cv2.imwrite(os.path.join(mask_dir, img_name + ".png"), save_mask)


def get_class_weight():
    logger.info("[*] Loaded the class weights.")
    pixel_dist = [701430751,  405558880, 1210162677, 1097775645,  561645880, 1021073281,  337966950, 1217985936]
    pixel_dist = np.array(pixel_dist)
    weights = pixel_dist.sum() / (cfg.DATASET.NUM_CLASSES * pixel_dist)

    return weights


if __name__ == '__main__':
    weights = get_class_weight()
    print(weights)
