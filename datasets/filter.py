import os
import cv2
import sys
import json
import torch
import numpy as np

from collections import defaultdict
from scipy.ndimage.measurements import center_of_mass
from tqdm import tqdm
from config import cfg
from utils.utils import logger
from datasets import uniform


pbar = None


def build_filter_sets(imgs, num_classes, mode, id2trainid=None):
    if not (mode == "train"):
        return imgs

    json_fn = os.path.join(cfg.DATASET.PCL_DIR, "classwised_set.json")
    if os.path.isfile(json_fn):
        logger.info("[*] Loading Class-wised file: {}".format(json_fn))
        with open(json_fn, "r") as f:
            records = json.load(f)
        records = {int(k): v for k, v in records.items()}
        logger.info("[*] Found {} classes.".format(len(records)))
    else:
        logger.info("[*] Didn\'t find {}, so building it.".format(json_fn))
        records = uniform.generate_classwised_all(imgs, num_classes, id2trainid)
        with open(json_fn, "w") as f:
            json.dump(records, f, indent=4)

    loss_fn = os.path.join(cfg.DATASET.PCL_DIR, "loss_info.json")
    if os.path.isfile(loss_fn):
        logger.info("[*] Loading Loss info file: {}".format(loss_fn))
        with open(loss_fn, "r") as f:
            loss_info = json.load(f)

        new_records = defaultdict(list)
        for k, v in records.items():
            for item in v:
                img_n = os.path.basename(item[0]).split(".")[0]
                loss = loss_info[img_n]["ce_loss"]
                new_records[k].append((*item, loss))

    else:
        logger.info("[*] Didn\'t find {}, so didn\'t use it.")

        new_records = defaultdict(list)
        for k, v in records.items():
            for item in v:
                new_records[k].append((*item, 0))

    records = new_records

    return records


def loss_filter(records, loss_upper_bound):
    """Filter the loss which is larger than loss upper bound.
    Args:
        records: dict, with list [image_fn, mask_fn, centroid, v, loss]
        loss_upper_bound: float
    Returns:
        records
    """
    new_records = defaultdict(list)
    for k, v in records.items():
        for item in v:
            loss = item[-1]
            if loss < loss_upper_bound:
                new_records[k].append(item[:-1])  # to excluding loss item

    return new_records


def filter_duplicate(img_list):
    img_d = dict()
    imgs = []
    for item in img_list:
        if item[0] not in img_d.keys():
            img_d[item[0]] = 1
            imgs.append(item)

    return imgs


def random_sampling(alist, num):
    sampling = []
    len_list = len(alist)
    if num > len_list:
        num = len_list
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i]]
        sampling.append(item)

    return sampling


def build_epoch(imgs, records, num_classes, mode):
    if not (mode == "train"):
        return imgs

    logger.info("[*] Filter the data with high loss.")
    records = loss_filter(records, cfg.DATASET.LOSS_UPPER_BOUND)

    logger.info("[*] Sampling the image with max num of image.")

    imgs_sampling = []
    class_counter = 0
    for class_id in range(num_classes):
        num_sampling = cfg.DATASET.NUM_IMG_PER_CLASS
        records_len = len(records[class_id])
        if records_len == 0:
            pass
        else:
            class_records = random_sampling(records[class_id], num_sampling)
            imgs_sampling.extend(class_records)
            class_counter += 1

    logger.info("[*] Sampling including {} classes.".format(class_counter))
    imgs_sampling = filter_duplicate(imgs_sampling)

    return imgs_sampling






