
import os
import cv2
import glob
import torch
import numpy as np

from PIL import Image
from torch.utils import data

from config import cfg
# from runx.logx import logx
import torchvision.transforms as ttf
from datasets import uniform


class BaseDataset(data.Dataset):

    ignore_label = 255
    num_classes = 255
    id_to_trainid = {}
    trainid_to_id = {}

    def __init__(self, mode, uniform_sampling, joint_transform_list, img_transform, label_transform):
        super(BaseDataset, self).__init__()
        self.mode = mode
        self.uniform_sampling = uniform_sampling
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.all_data = self.build_data()
        self.records = None

        if self.img_transform is None:
            self.img_transform = ttf.Compose([
                ttf.ToTensor(),
                ttf.Normalize(mean=cfg.DATASET.MEAN,
                              std=cfg.DATASET.STD)
            ])

    def build_data(self):
        raise NotImplementedError

    def build_epoch(self):
        self.data = uniform.build_epoch(self.all_data, self.records, self.num_classes, self.mode)

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
            for k, v in self.id_to_trainid.items():
                mask[mask_raw == k] = v
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = None

        img = Image.fromarray(img.astype(np.uint8))

        img_name, _ = os.path.splitext(os.path.basename(img_fn))

        return img, mask, img_name

    def do_transforms(self, img, mask, centroid=None):

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                img, mask = xform(img, mask, centroid)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if mask is not None:
            if self.label_transform is not None:
                mask = self.label_transform(mask)
            else:
                mask = torch.from_numpy(np.asarray(mask, dtype=np.int32)).long()
        else:
            pass

        return img, mask

    def __getitem__(self, item):
        if len(self.data[item]) == 2:
            img_path, mask_path = self.data[item]
            centroid, class_id = None, None
        else:
            img_path, mask_path, centroid, class_id = self.data[item]

        img, mask, img_name = self.read_images(img_path, mask_path)

        img, mask = self.do_transforms(img, mask)

        if mask is not None:
            return img, mask, img_name
        else:
            return img, img_name

    def __len__(self):
        return len(self.data)











