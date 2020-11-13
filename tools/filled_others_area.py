import os
import cv2
import shutil
import glob
import json
import copy
import random
import argparse
import heapq as hq
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


class Buffer(object):
    def __init__(self, x, y, dis, cls):
        self.x = x
        self.y = y
        self.dis = dis
        self.cls = cls

    def __lt__(self, other):
        if self.dis > other.dis:
            return True
        else:
            return False

    def __repr__(self):
        return "[x: {}, y: {}, dis: {}, cls: {}]".format(self.x, self.y, self.dis, self.cls)


def filled_others(mask_f, args, cls_id=17, threshold=900):
    mask_n = os.path.basename(mask_f)
    dst_f = os.path.join(args.output_dir, mask_n)
    mask = cv2.imread(mask_f, cv2.IMREAD_UNCHANGED)
    others = (mask == cls_id)
    num_others = others.sum()
    if num_others > threshold:
        shutil.copyfile(mask_f, dst_f)

    ret = copy.deepcopy(mask)

    kernel = np.ones([3, 3], np.uint8)
    dilation = cv2.dilate(others.astype(np.uint8), kernel, iterations=1)
    search_area = dilation - others
    unknown_y, unknown_x = np.where(others == 1)
    search_y, search_x = np.where(search_area == 1)
    len_unknown, len_search = unknown_x.shape[0], search_x.shape[0]

    for idx in range(len_unknown):
        u_x, u_y = unknown_x[idx], unknown_y[idx]
        heap = []
        for s_idx in range(len_search):
            s_x, s_y = search_x[0], search_y[0]
            dis = abs(s_x - u_x) + abs(s_y - u_y)
            cls = mask[s_y, s_x]
            buf = Buffer(s_x, s_y, dis, cls)
            if len(heap) < 3:
                hq.heappush(heap, buf)
            else:
                largest_dis = heap[0].dis
                if dis < largest_dis:
                    hq.heappop(heap)
                    hq.heappush(heap, buf)
        clss = [buf.cls for buf in heap]
        cls_counter = Counter(clss)
        cls = cls_counter.most_common(1)[0][0]
        ret[u_y, u_x] = cls

    cv2.imwrite(dst_f, ret)


def main(args):
    pool = mp.Pool(args.num_workers)
    mask_fs = glob.glob(os.path.join(args.mask_dir, "*.png"))
    for mask_f in tqdm(mask_fs, desc="filled the others"):
        pool.apply_async(filled_others, args=(mask_f, args, args.cls_id, args.threshold))

    pool.close()
    pool.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str, default="/nfs/users/huangfeifei/dataset/remote_sensing/labels")
    parser.add_argument("--output_dir", type=str, default="/nfs/users/huangfeifei/dataset/remote_sensing/labels_clean")
    parser.add_argument("--cls_id", type=int, default=17)
    parser.add_argument("--threshold", type=int, default=900)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)



