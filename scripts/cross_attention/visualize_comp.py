import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root', type=str, default='./atten_bear_visualization/', help='down or middle or up'
)
parser.add_argument(
    '--num_sample',
    type=int,
    default=3,
    help='number of samples generated in the ldm model',
)
parser.add_argument('--num_level', type=int, default=4, help='(8, 16, 32, 64)')
parser.add_argument('--num_att_comp', type=int, default=3, help='down, up, down&up =3')
parser.add_argument(
    '--num_att_maps',
    type=int,
    default=10,
    help='number of attnetion maps generated in the model.',
)
parser.add_argument(
    '--token_idx',
    type=int,
    default=1,
    help='number of samples generated in the ldm model',
)
args = parser.parse_args()

root = args.root
num_sample = args.num_sample
num_level = args.num_level    # (8, 16, 32, 64)
num_att_comp = args.num_att_comp   # level= 00,22,02
num_att_maps = args.num_att_maps

imgs = sorted(glob(os.path.join(root, '*.png')))  # os.listdir(root)

attmap_list = []
bundle_list = []

""" 1. bundle image 읽어오는 과정 """
cur_idx = num_sample * num_att_maps   # i*num_att_maps + k +1
for m in range(cur_idx, cur_idx + num_att_maps):
    if m == cur_idx:
        h, w, c = cv2.imread(imgs[m]).shape
        white_img = np.ones((h, h * num_sample, 3)) * 255

    bundle_list.append(
        cv2.resize(cv2.imread(imgs[m]), (h * num_sample, h))
    )   # (516, 1548, 3)

""" 2. attention map 읽어오는 과정 """
for i in range(num_sample):   # num_sample
    for k in range(num_att_maps):   # num attention maps in each sample
        attmap_list.append(cv2.resize(cv2.imread(imgs[i * num_att_maps + k]), (h, h)))

""" 3. make bundle composition """
last_bundle = None
for mm in range(num_level):
    if mm == 0:
        init_bundle = np.concatenate((bundle_list[mm], white_img, white_img), axis=1)
    else:
        cur_idx = mm * num_att_comp - (num_sample - 1)
        cur_bundle = np.concatenate(
            bundle_list[cur_idx : cur_idx + num_att_comp], axis=0
        )   # down, downup, up
        if last_bundle is None:
            last_bundle = cur_bundle
        else:
            last_bundle = np.concatenate((last_bundle, cur_bundle), axis=1)
last_bundle = np.concatenate((init_bundle, last_bundle), axis=0)
cv2.imwrite(
    os.path.join(root, './last_bundle_imgs_idx:{0}.png'.format(args.token_idx)),
    last_bundle,
)

""" 4. make attenmap composition
0: 0 10 20
1: 1 11 21/ 2 12 22 /3 13 23
2: 4 14 24 /5 15 25 /6 16 26
....
"""
last_bundle = None
for r in range(num_level):   # 4
    cur_att_map_list = []
    if r == 0:
        for i in range(num_sample):   # 3
            cur_attmap = attmap_list[i * num_att_maps]
            cur_att_map_list.append(cur_attmap)
        cur_att_map_list = np.concatenate(cur_att_map_list, axis=1)

        white_img = np.ones((h, h * num_sample * 2, 3)) * 255
        init_bundle = np.concatenate((cur_att_map_list, white_img), axis=1)
    else:
        cur_idx = r * (num_level - 1) - (num_sample - 1)   # -2
        for i in range(num_sample):   # 3
            cur_subatt_map_list = []
            for j in range(num_sample):
                cur_attmap = attmap_list[i + cur_idx + j * num_att_maps]
                cur_subatt_map_list.append(cur_attmap)
            cur_att_map_list.append(np.concatenate(cur_subatt_map_list, axis=1))

        cur_bundle = np.concatenate(
            cur_att_map_list, axis=0
        )  # .reshape(h*num_sample, h*num_sample, 3)

        if last_bundle is None:
            last_bundle = cur_bundle
        else:
            last_bundle = np.concatenate((last_bundle, cur_bundle), axis=1)
last_bundle = np.concatenate((init_bundle, last_bundle), axis=0)
cv2.imwrite(
    os.path.join(root, './last_bundle_attnmaps_idx:{0}.png'.format(args.token_idx)),
    last_bundle,
)
