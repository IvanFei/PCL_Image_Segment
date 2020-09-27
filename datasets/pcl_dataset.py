import os
import torch
from torch.utils import data

from config import cfg
# from runx.logx import logx
from datasets.base_dataset import BaseDataset
from datasets import uniform


class DataSet(BaseDataset):

    ignore_label = cfg.DATASET.IGNORE_LABEL
    num_classes = cfg.DATASET.NUM_CLASSES
    id_to_trainid = {100: 0, 200: 1, 300: 2, 400: 3, 500: 4, 600: 5, 700: 6, 800: 7}
    trainid_to_id = {0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600, 6: 700, 7: 800}

    def __init__(self, mode, uniform_sampling=False, joint_transform_list=None, img_transform=None, label_transform=None):

        data_root = cfg.DATASET.PCL_DIR

        assert mode in ("train_val", "train", "val", "test"), "Mode should be `train_val` | `train` | `val` | `test`."

        self.img_ext = "tif"
        self.mask_ext = "png"
        self.img_root = os.path.join(data_root, mode, "image")
        self.mask_root = os.path.join(data_root, mode, "label") if mode != "test" else None

        super(DataSet, self).__init__(mode=mode, uniform_sampling=uniform_sampling,
                                      joint_transform_list=joint_transform_list,
                                      img_transform=img_transform, label_transform=label_transform)

        if self.uniform_sampling:
            self.records = uniform.build_classwised_sets(self.all_data, self.num_classes,
                                                         self.mode, cfg.DATASET.ID_TO_TRAINID)

            self.build_epoch()
        else:
            self.data = self.all_data

    def build_data(self):
        imgs = self.find_images(self.img_root, self.mask_root, self.img_ext, self.mask_ext)

        return imgs


if __name__ == '__main__':
    dataset = DataSet(mode="train", uniform_sampling=True)
    print(len(dataset))
    image, mask, img_name = dataset[0]
    print(f"[*] image name: {img_name}")
    print(f"[*] image shape: {image.shape}")
    print(f"[*] mask shape: {mask.shape}")

    data_loader = data.DataLoader(dataset, shuffle=True, num_workers=4, drop_last=True, batch_size=4)
    for data_batch in data_loader:
        image, mask, img_name = data_batch
        print(f"[*] image name: {img_name}")
        print(f"[*] image shape: {image.shape}")
        print(f"[*] mask shape: {mask.shape}")
        break













