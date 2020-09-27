import os
import cv2
import sys
import json
import torch
import numpy as np

from collections import defaultdict
from scipy.ndimage.measurements import center_of_mass
from PIL import Image
from tqdm import tqdm
from config import cfg
from utils.utils import logger


pbar = None


def classwised_image(item, num_classes, id2trainid):
    """ For one image. """
    image_fn, mask_fn = item
    mask = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)
    records = defaultdict(list)

    for k, v in id2trainid.items():
        if k in mask:
            binary_mask = (mask == k).astype(int)
            centroid_y, centroid_x = center_of_mass(binary_mask)
            centroid = (int(centroid_x), int(centroid_y))
            records[v].append((image_fn, mask_fn, centroid, v))

    pbar.update(1)

    return records


def pooled_classwised_generator(items, num_classes, id2trainid):
    """
    Generated class wised sets use multi threading.
    Args:
         items: (image_fn, mask_fn)
         num_classes: num of classes
         id2trainid: dict
    Returns:
        dict
    """
    from multiprocessing.dummy import Pool
    from functools import partial
    pool = Pool(80)
    global pbar
    pbar = tqdm(total=len(items), desc="pooled class-wised sets", file=sys.stdout)
    classwised_item = partial(classwised_image, num_classes=num_classes, id2trainid=id2trainid)

    records = defaultdict(list)
    new_records = pool.map(classwised_item, items)
    pool.close()
    pool.join()

    for image_items in new_records:
        for class_id in image_items:
            records[class_id].extend(image_items[class_id])
    return records


def generate_classwised_all(items, num_classes, id2trainid):
    """
    Intermediate function to call pooled_classwised_generator.
    """
    pooled_records = pooled_classwised_generator(items, num_classes, id2trainid)

    return pooled_records


def build_classwised_sets(imgs, num_classes, mode, id2trainid=None):
    """
    Build the Class-wised image sets
    """
    if not (mode == "train" and cfg.DATASET.CLASS_UNIFORM_PCT):
        return []

    json_fn = os.path.join(cfg.DATASET.PCL_DIR, "classwised_set.json")
    if os.path.isfile(json_fn):
        logger.info("[*] Loading Class-wised file: {}".format(json_fn))
        with open(json_fn, "r") as f:
            records = json.load(f)
        records = {int(k): v for k, v in records.items()}
        logger.info("[*] Found {} classes".format(len(records)))
    else:
        logger.info("[*] Didn\'t find {}, so building it.".format(json_fn))
        records = generate_classwised_all(imgs, num_classes, id2trainid)
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
    """
    Generate an epoch of image using uniform sampling
    Will not apply uniform sampling if not train or class uniform is off.

    Args:
        imgs: list of images: (img_fn, mask_fn)
        records: dict of classes which is list including img_fn, mask_fn, class_id
        num_classes: int
        mode: str
    Returns:
        imgs: list of images
    """
    class_uniform_pct = cfg.DATASET.CLASS_UNIFORM_PCT
    if not (mode == "train" and class_uniform_pct):
        return imgs

    logger.info("[*] Class Uniform Percentage: {}".format(str(class_uniform_pct)))
    num_epoch = int(len(imgs))

    logger.info("[*] Class Uniform items per Epoch: {}".format(str(num_epoch)))
    num_per_class = int((num_epoch * class_uniform_pct) / num_classes)
    class_uniform_count = num_per_class * num_classes
    num_rand = num_epoch - class_uniform_count

    imgs_uniform = ramdom_sampling(imgs, num_rand)

    for class_id in range(num_classes):
        num_per_class_biased = num_per_class
        records_len = len(records[class_id])
        if records_len == 0:
            pass
        else:
            class_records = ramdom_sampling(records[class_id], num_per_class_biased)
            imgs_uniform.extend(class_records)

    return imgs_uniform

