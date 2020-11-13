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


def get_size(w, h, divisor=32):
    w = (w // divisor) * divisor
    h = (h // divisor) * divisor

    return w, h


def cut_image(img, w, h):
    imgs = []
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

    return imgs, axis_list


def cut_image_new(img, w, h, overlap=32):
    imgs = []
    size_per_img = 256 - overlap
    num_patchs = math.ceil(w / size_per_img)
    inter_per_patch = size_per_img
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

    return imgs, axis_list


def cut_placehold(img, w, h, patch_size=384, overlap=160):
    imgs = [img]
    axis_list = [(0, 0)]

    return imgs, axis_list


def cut_image_v2(img, w, h, patch_size=384, overlap=160):
    imgs = []
    size_per_img = patch_size - overlap
    num_patchs = math.ceil(w / size_per_img)
    inter_per_patch = size_per_img
    idxs = []
    for i in range(num_patchs):
        idx = i * inter_per_patch
        idx = min(idx, w - patch_size)
        idxs.append(idx)

    axis_list = []
    for r_idx in idxs:
        for c_idx in idxs:
            axis_list.append((r_idx, c_idx))
            # ugly coding
            t_img = img[:, r_idx:r_idx+patch_size, c_idx:c_idx+patch_size].unsqueeze(dim=0)
            t_img = F.interpolate(t_img, size=(256, 256), mode="bilinear").squeeze(dim=0)
            imgs.append(t_img)

    return imgs, axis_list


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
    # TODO: save show
    # cv2.imwrite(os.path.join(output_dir, img_name + "_vis.png"), img)
    w, h, c = img.shape

    # TODO: cut the img to 256 x 256
    batch_size = 16

    # TODO plan A: no resize and cut with 256 x 256 including overlap
    # img = Image.fromarray(img.astype(np.uint8))
    # img = img_transform(img)
    # cw, ch = w, h
    # imgs, axis_list = cut_image(img, w, h)

    # TODO plan B: resize with 32 x 32 and cut with 256 x 256  including overlap
    # resize
    patch_size, overlap = 256, 64
    cw, ch = get_size(w, h, divisor=32)
    img_resize = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(img_resize.astype(np.uint8))
    img = img_transform(img)
    if cw <= 288:
        imgs, axis_list = cut_image(img, cw, ch)
    else:
        imgs, axis_list = cut_image_new(img, cw, ch, overlap=overlap)

    # TODO plan C: resize with 32 x 32 and cut with patch_size including overlap
    # cw, ch = get_size(w, h, divisor=32)
    # img_resize = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
    # img = Image.fromarray(img_resize.astype(np.uint8))
    # img = img_transform(img)
    # if cw <= 384:
    #     patch_size, overlap = 256, None
    #     imgs, axis_list = cut_image(img, cw, ch)
    # else:
    #     patch_size, overlap = 384, 160
    #     imgs, axis_list = cut_image_v2(img, cw, ch, patch_size=patch_size, overlap=overlap)

    num_imgs = len(imgs)
    num_epochs = math.ceil(num_imgs / batch_size)

    result = np.zeros([cfg.DATASET.NUM_CLASSES, ch, cw])
    counter = np.zeros([cfg.DATASET.NUM_CLASSES, ch, cw])
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
        if patch_size != 256:
            probs = F.interpolate(probs, size=(patch_size, patch_size), mode="bilinear")
        probs = probs.cpu().numpy()

        for idx, ax in enumerate(img_axis_list):
            y, x = ax
            result[:, y:y+patch_size, x:x+patch_size] += probs[idx]
            counter[:, y:y+patch_size, x:x+patch_size] += 1

    result = result / counter
    result = np.argmax(result, axis=0)
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)

    write_mask(result, img_name, output_dir)
    # print("[*] Cost time: {}".format(time.time() - start))


# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     img_dir = "/nfs/users/huangfeifei/dataset/remote_sensing/test_multi_scale/image"
#     img_paths = glob.glob(img_dir + '/' + '*')  # 后台存储的测试集图片路径
#     print("[*] image len: {}".format(len(img_paths)))
#     model = init_model()
#     for img_f in img_paths:
#         predict(model, img_f, "./output")



