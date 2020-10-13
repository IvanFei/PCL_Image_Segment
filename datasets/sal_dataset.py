import os
import cv2
import json
import glob
import torch
import numbers
import random

import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as ttf
from torchvision.transforms import functional as F

from config import cfg


class DataSet(data.Dataset):

    ignore_label = cfg.DATASET.IGNORE_LABEL
    num_classes = cfg.DATASET.NUM_CLASSES
    id_to_trainid = {100: 0, 200: 1, 300: 2, 400: 3, 500: 4, 600: 5, 700: 6, 800: 7}
    trainid_to_id = {0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600, 6: 700, 7: 800}

    def __init__(self, mode, class_id=7, joint_transform_list=None, img_transform=None, label_transform=None, *kwargs):
        self.mode = mode
        self.class_id = class_id
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_root = cfg.DATASET.PCL_DIR
        if self.mode == "train":
            cls_wised_fn = os.path.join(self.image_root, "classwised_set.json")
            with open(cls_wised_fn, "r") as f:
                self.cls_wised_dict = json.load(f)
            self.image_list = self.cls_wised_dict[str(class_id)]
        else:
            img_ext = "tif"
            mask_ext = "png"
            img_root = os.path.join(self.image_root, mode, "image")
            mask_root = os.path.join(self.image_root, mode, "label") if mode != "test" else None
            self.image_list = self.find_images(img_root, mask_root, img_ext, mask_ext)

        self.image_num = len(self.image_list)

        if self.img_transform is None:
            self.img_transform = ttf.Compose([
                ttf.ToTensor(),
                ttf.Normalize(mean=cfg.DATASET.MEAN,
                              std=cfg.DATASET.STD)
            ])

    def __getitem__(self, item):
        if len(self.image_list[item]) == 4:
            image_fn, mask_fn, centroid, cls = self.image_list[item]
        else:
            image_fn, mask_fn = self.image_list[item]

        image, mask, edge, img_name = self.read_images(image_fn, mask_fn)
        image, mask, edge = self.do_transforms(image, mask, edge)

        if mask is not None:
            return image, mask, edge, img_name
        else:
            return image, img_name

    def read_images(self, img_fn, mask_fn=None):
        assert len(self.id_to_trainid) and len(self.trainid_to_id) \
               and len(self.trainid_to_id) == len(self.id_to_trainid), "Unassign/error assign the id to trained."

        if mask_fn is None:
            assert self.mode == "test", "Mask is None, the mode should be test."

        img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode != "test":
            mask_raw = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)

            mask = np.zeros_like(mask_raw)
            mask[mask_raw == self.trainid_to_id[self.class_id]] = 1  # binary classify
            mask = 1 - mask
            edge = self._edge_creator(mask)
            # flip transform
            if random.uniform(0, 1) > 0.5:
                img = img[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
                edge = edge[:, ::-1].copy()

            mask = Image.fromarray(mask.astype(np.uint8))
            edge = Image.fromarray(edge.astype(np.uint8))
        else:
            mask = None
            edge = None

        img = Image.fromarray(img.astype(np.uint8))

        img_name, _ = os.path.splitext(os.path.basename(img_fn))

        return img, mask, edge, img_name

    @staticmethod
    def find_images(img_root, mask_root=None, img_ext="tif", mask_ext="png"):
        img_path = "{}/*.{}".format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_n, _ = os.path.splitext(img_fn)
            if mask_root is not None:
                full_mask_fn = "{}.{}".format(img_n, mask_ext)
                full_mask_fn = os.path.join(mask_root, full_mask_fn)
            else:
                full_mask_fn = None
            items.append((full_img_fn, full_mask_fn))
        return items

    @staticmethod
    def _edge_creator(mask):
        sure_area = (mask == 1).astype(np.uint8) * 255
        edge = cv2.Laplacian(sure_area, ddepth=-1, ksize=5) / 255
        return edge

    def do_transforms(self, img, mask, edge, centroid=None):

        if self.joint_transform_list is not None:
            raise NotImplementedError

        if self.img_transform is not None:
            img = self.img_transform(img)

        if mask is not None:
            if self.label_transform is not None:
                raise NotImplementedError
            else:
                mask = torch.from_numpy(np.asarray(mask, dtype=np.int32)).float()
                edge = torch.from_numpy(np.asarray(edge, dtype=np.int32)).float()
                mask = mask.unsqueeze(dim=0)
                edge = edge.unsqueeze(dim=0)
        else:
            pass

        return img, mask, edge

    def __len__(self):
        return self.image_num

    def save_folder(self):
        return "Sal"


if __name__ == '__main__':
    dataset = DataSet(mode="train")
    print(len(dataset))
    image, mask, edge, img_name = dataset[0]
    print(f"[*] image name: {img_name}")
    print(f"[*] image shape: {image.shape}")
    print(f"[*] mask shape: {mask.shape}")
    print(f"[*] edge shape: {edge.shape}")

    data_loader = data.DataLoader(dataset, shuffle=True, num_workers=4, drop_last=True, batch_size=1)
    for data_batch in data_loader:
        image, mask, edge, img_name = data_batch
        print(f"[*] image name: {img_name}")
        print(f"[*] image shape: {image.shape}")
        print(f"[*] mask shape: {mask.shape}")
        print(f"[*] edge shape: {edge.shape}")
        break




