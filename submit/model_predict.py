import os
import cv2
import math
import time
import torch
import copy
import glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as ttf
import numpy as np
from PIL import Image

from .config import cfg
# from model_define import init_model


def write_mask(mask, img_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    H, W = mask.shape
    img_name = img_name
    mask = mask
    seg = np.zeros([H, W]) * cfg.DATASET.IGNORE_LABEL
    for c in cfg.DATASET.TRAINID_TO_ID.keys():
        seg[mask == c] = cfg.DATASET.TRAINID_TO_ID[c]

    save_mask = seg.astype(np.uint16)

    cv2.imwrite(os.path.join(output_dir, img_name + ".png"), save_mask)


def predict(model, input_path, output_dir):
    # start = time.time()
    # judge using cuda or not.
    is_cuda = torch.cuda.is_available()
    # transformer
    img_transform = ttf.Compose([
        ttf.ToTensor(),
        ttf.Normalize(mean=cfg.DATASET.MEAN,
                      std=cfg.DATASET.STD)
    ])

    img_name, _ = os.path.splitext(os.path.basename(input_path))
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w, h, c = img.shape

    img = Image.fromarray(img.astype(np.uint8))
    img = img_transform(img)

    # TODO: cut the img to 256 x 256
    imgs = []
    batch_size = 16
    num_patchs = math.ceil(w / 256)
    sum_strides = num_patchs * 256 - w
    stride_per_patch = int(sum_strides / (num_patchs - 1)) if num_patchs > 1 else 0
    inter_per_patch = 256 - stride_per_patch
    idxs = []
    for i in range(num_patchs):
        idx = i * inter_per_patch
        idx = min(idx, w - 256)
        idxs.append(idx)

    axis_list = []
    for r_idx in idxs:
        for c_idx in idxs:
            axis_list.append((r_idx, c_idx))
            imgs.append(img[:, r_idx:r_idx+256, c_idx:c_idx+256])

    num_imgs = len(imgs)
    num_epochs = math.ceil(num_imgs / batch_size)

    result = np.zeros([cfg.DATASET.NUM_CLASSES, h, w])
    counter = np.zeros([cfg.DATASET.NUM_CLASSES, h, w])
    for epoch in range(num_epochs):
        end_idx = min((epoch + 1) * batch_size, num_imgs)
        img_batch_list = imgs[epoch * batch_size: end_idx]
        img_axis_list = axis_list[epoch * batch_size: end_idx]
        img_batch = torch.stack(img_batch_list, dim=0)

        if is_cuda:
            img_batch = img_batch.cuda()

        inputs = {"images": img_batch}

        with torch.no_grad():
            preds = model(inputs)

        preds = preds["pred"]
        probs = torch.softmax(preds, dim=1)
        probs = probs.cpu().numpy()
        for idx, ax in enumerate(img_axis_list):
            y, x = ax
            result[:, y:y+256, x:x+256] += probs[idx]
            counter[:, y:y+256, x:x+256] += 1

    result = result / counter
    result = np.argmax(result, axis=0)

    write_mask(result, img_name, output_dir)
    # print("[*] Cost time: {}".format(time.time() - start))


# if __name__ == '__main__':
#     img_dir = "/nfs/users/huangfeifei/dataset/remote_sensing/test_multi_scale/image"
#     img_paths = glob.glob(img_dir + '/' + '*')  # 后台存储的测试集图片路径
#     print("[*] image len: {}".format(len(img_paths)))
#     model = init_model()
#     for img_f in img_paths:
#         predict(model, img_f, "./output")



