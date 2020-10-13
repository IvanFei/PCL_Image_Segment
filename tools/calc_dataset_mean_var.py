import os
import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm


def calc_mean_var(img_dir, suffix="tif"):
    img_list = glob.glob(os.path.join(img_dir, '*.' + suffix))
    total, r, g, b, r_2, g_2, b_2 = 0, 0, 0, 0, 0, 0, 0
    for img_f in tqdm(img_list):
        img = Image.open(img_f)
        img = np.asarray(img)
        img = img.astype("float32") / 255
        total += img.shape[0] * img.shape[1]

        r += img[:, :, 0].sum()
        g += img[:, :, 1].sum()
        b += img[:, :, 2].sum()

        r_2 += (img[:, :, 0] ** 2).sum()
        g_2 += (img[:, :, 1] ** 2).sum()
        b_2 += (img[:, :, 2] ** 2).sum()

    r_mean = r / total
    g_mean = g / total
    b_mean = b / total

    r_var = r_2 / total - r_mean / total
    g_var = g_2 / total - g_mean / total
    b_var = b_2 / total - b_mean / total

    print(f"[*] Mean is: [{r_mean}, {g_mean}, {b_mean}]")
    print(f"[*] Var is: [{r_var}, {g_var}, {b_var}]")


def args_parse():
    parser = argparse.ArgumentParser("Args")
    parser.add_argument("--image_dir", type=str,
                        default="/nfs/users/huangfeifei/dataset/pcl_image_segment/train/image")
    parser.add_argument("--suffix", type=str, default="tif")

    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    calc_mean_var(args.image_dir, args.suffix)


if __name__ == '__main__':
    main()

