import os
import cv2
import sys
import json
import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from config import cfg
from utils.utils import logger


pbar = None


def filter_image(item):
    image_fn, mask_fn = item
    image_n = os.path.basename(image_fn).split(".")[0]
    records = defaultdict(list)

    image = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brt = image_gray.mean()
    median_brt = np.median(image_gray)
    num_classes_per_image = np.unique(mask).shape[0]
    records[str(num_classes_per_image == 1)].append((image_fn, mask_fn, num_classes_per_image, mean_brt, median_brt))

    pbar.update(1)

    return records


def pooled_filter_generator(items, num_classes, id2trainid):
    from multiprocessing.dummy import Pool
    pool = Pool(80)
    global pbar
    pbar = tqdm(total=len(items), desc="pooled filter sets", file=sys.stdout)

    records = defaultdict(list)
    new_records = pool.map(filter_image, items)
    pool.close()
    pool.join()
    for image_item in new_records:
        for k, v in image_item.items():
            records[k].extend(v)
    return records


def generate_filter_all(items, num_classes, id2trainid):

    pooled_records = pooled_filter_generator(items, num_classes, id2trainid)

    return pooled_records


def build_filter_sets(imgs, num_classes, mode, id2trainid=None):
    if not (mode == "train"):
        return imgs

    json_fn = os.path.join(cfg.DATASET.PCL_DIR, "filter_set.json")
    if os.path.isfile(json_fn):
        logger.info("[*] Loading Filter sets file: {}".format(json_fn))
        with open(json_fn, "r") as f:
            records = json.load(f)
    else:
        logger.info("[*] Didn\'t find {}, so building it.".format(json_fn))
        records = generate_filter_all(imgs, num_classes, id2trainid)
        with open(json_fn, "w") as f:
            json.dump(records, f, indent=4)

    return records


def ramdom_sampling(alist, num):
    """
    Randomly sample num items from the list
    alist: list of centroids to sample from
    num: can be larger than the list and if so, then wrap around
    return: class uniform samples from the list
    """
    sampling = []
    len_list = len(alist)
    assert len_list, "len list is zero"
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling


def build_epoch(imgs, records, num_classes, mode):
    if not (mode == "train"):
        return imgs

    one_class_filter_pct = cfg.DATASET.ONE_CLASS_FILTER_PCT
    logger.info("[*] One Class Filter Percentage: {}".format(str(one_class_filter_pct)))

    num_imgs = int(len(imgs))
    logger.info("[*] Number of images: {}".format(str(num_imgs)))

    imgs_one_classes = records["True"]

    num_one_class = int(len(imgs_one_classes) * one_class_filter_pct)
    imgs_multi_classes = records["False"]

    one_class_uniform = ramdom_sampling(records["True"], num_one_class)

    imgs_multi_classes.extend(one_class_uniform)

    return imgs_multi_classes



