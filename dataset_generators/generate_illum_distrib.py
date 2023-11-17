## This script is to generate
# GT illumination distribution .PNG files for mirror ball crops
# using crops pngs ans masks &
#GT illumination distribution .PNG files for grey edges taken from gt.csv of Cube++ dataset

import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm

import cv2
import json

import torch
from math import sqrt

EPSILON = 1e-3

# List of histogram parameters

HIST_BINS = [116, 100]
HIST_RANGE = [[-1.74, 1.74], [-1, 2]]
HIST_RANGE_FLAT = [-1.74, 1.74, -1, 2]
HIST_TARGET_SIZE = [128, 128]
HIST_VERT_PADD = 6
HIST_HORR_PADD = 14

def path_to_list(path_to_directory):
    return os.listdir(path_to_directory)

def open_image(path):
    img = np.array(cv2.imread(path,  cv2.IMREAD_UNCHANGED))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)

def load_mask(path):
    return np.load(path)

def black_level_substraction(pixels):
    good_pxls_list = []
    for p in pixels:
        if not (np.all(p) == 0):
            good_pxls_list.append(p)
    pixels = np.array(good_pxls_list)
    pixels = pixels - 2048
    pixels[pixels < 0] = 0
    return pixels


def get_illum_est_from_csv(csv_data, img_name):
    csv_data = csv_data[csv_data.image == img_name]
    r_l = csv_data['left_r'].tolist()[0]
    g_l = csv_data['left_g'].tolist()[0]
    b_l =  csv_data['left_b'].tolist()[0]
    left = np.array([r_l, g_l, b_l])
    r_r = csv_data['right_r'].tolist()[0]
    g_r = csv_data['right_g'].tolist()[0]
    b_r =  csv_data['right_b'].tolist()[0]
    right = np.array([r_r, g_r, b_r])
    illum_ests = np.array([left, right])
    return illum_ests

# def rgb2hist(pixels_list, bins_num):
#     chrom = rgb2chrom(pixels_list)
#     hist, _, _ = np.histogram2d(chrom[0], chrom[1],
#                                 bins=HIST_BINS,
#                                 range=HIST_RANGE)

#     hist /= hist.sum()
#     padded_hist = np.zeros(HIST_TARGET_SIZE, dtype=float64)
#     padded_hist[HIST_VERT_PADD:-HIST_VERT_PADD,
#                 HIST_HORR_PADD:-HIST_HORR_PADD] = hist

#     return padded_hist.T


def crop_to_pixels(crop):
    pixels_num = crop.shape[0] * crop.shape[1]
    pixels_list = np.reshape(crop, [pixels_num, 3])
    pixels_list = black_level_substraction(pixels_list)

    return pixels_list

def rgb2chrom(rgb):
    rgb = rgb.astype(np.float64)

    chrom_list = []

    for r,g,b in rgb:
        if (r + g + b) > EPSILON:
            alpha = (2 * b  - (r + g)) / (r + g + b)
            beta = np.sqrt(3) * (r - g) / (r + g + b)

            chrom_list.append((beta, alpha))

    chrom = np.array(chrom_list)

    return chrom

def generate_chroma_histogram(pixel_list,
                              bins=HIST_BINS,
                              hrange=HIST_RANGE,
                              target_size=HIST_TARGET_SIZE,
                              vert_pad=HIST_VERT_PADD,
                              horr_padd=HIST_HORR_PADD):

    chrom = rgb2chrom(pixel_list)
    if not chrom.size:
        return np.zeros(target_size, dtype=np.float64)
    hist, _, _ = np.histogram2d(chrom[:,0], # beta
                                chrom[:,1], # alpha
                                bins=bins,
                                range=hrange,
                                normed=0)

    hist = hist.astype(np.float64)
    padded_hist = np.zeros(target_size, dtype=np.float64)
    padded_hist[vert_pad:-vert_pad,
                horr_padd:-horr_padd] = hist

    # Histogram does not follow Cartesian convention,
    # therefore transpose H for visualization purposes.
    # see https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

    return padded_hist.T


def torch_rgb2chrom(pixel_arr): #do not use: possibly broken
    mask = torch.sum(pixel_arr, dim=1) > EPSILON

    masked_arr = pixel_arr[mask, :]
    r, g, b = masked_arr[:, 0], masked_arr[:, 1], masked_arr[:, 2]
    alpha = (2 * b  - (r + g)) / (r + g + b)
    beta = sqrt(3) * (r - g) / (r + g + b)

    chromas = torch.stack((alpha, beta), dim=1)
    return chromas

def torch_pixels2chroma_hist(pixel_arr): #do not use: possibly broken
    chrom = torch_rgb2chrom(pixel_arr)
    hist, _ = torch.histogramdd(chrom, HIST_BINS,
                                range=HIST_RANGE_FLAT,
                                density=False)

    hist = hist.float()
    padded_hist = torch.zeros(HIST_TARGET_SIZE, dtype=torch.float64)
    padded_hist[HIST_VERT_PADD:-HIST_VERT_PADD,
                HIST_HORR_PADD:-HIST_HORR_PADD] = hist

    # Histogram does not follow Cartesian convention,
    # therefore transpose H for visualization purposes.
    # see https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

    return padded_hist.T


def crop_to_distrib(path_to_ball_crops,
                    name,
                    path_to_cube3p_masks,
                    weights=False):
    if weights:
        hist_fun = generate_chroma_histogram_no_weights
    else:
        hist_fun = generate_chroma_histogram
    ball_masked_crop = crop_mask(path_to_ball_crops,
                                 name,
                                 path_to_cube3p_masks)

    crop_pixels = crop_to_pixels(ball_masked_crop)

    hist = hist_fun(crop_pixels,
                    HIST_BINS,
                    HIST_RANGE,
                    HIST_TARGET_SIZE,
                    HIST_VERT_PADD,
                    HIST_HORR_PADD)
    return hist


def generate_chroma_histogram_no_weights(pixel_list,
                                         bins,
                                         hrange,
                                         target_size,
                                         vert_pad,
                                         horr_padd):

    chrom = rgb2chrom(pixel_list)
    hist, _, _ = np.histogram2d(chrom[:,0], # beta
                                chrom[:,1], # alpha
                                bins=bins,
                                range=hrange,
                                normed=0)

    hist = hist.astype(np.float64)
    padded_hist = np.zeros(target_size, dtype=np.float64)
    padded_hist[vert_pad:-vert_pad,
                horr_padd:-horr_padd] = hist

    padded_hist[padded_hist != 0] = 1

    # Histogram does not follow Cartesian convention,
    # therefore transpose H for visualization purposes.
    # see https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

    return padded_hist.T


def save_hist_npy(path_to_save_npy, filename, hist):
    if not os.path.exists(path_to_save_npy):
        os.makedirs(path_to_save_npy)

    hist_npy_file = os.path.join(path_to_save_npy, filename)
    np.save(hist_npy_file, hist)

def save_hist_png(path_to_save_png, filename, hist):
    if not os.path.exists(path_to_save_png):
        os.makedirs(path_to_save_png)

    hist_png_file = os.path.join(path_to_save_png, filename + '.png')
    cv2.imwrite(hist_png_file, hist)

def apply_mask(crop, mask):
    masked_crop = np.zeros(crop.shape)
    for i in range(3):
        masked_crop[:,:,i] = crop[:,:,i] * mask

    return masked_crop

def crop_mask(path_to_ball_png, filename, path_to_masks):
    crop = open_image(os.path.join(path_to_ball_png, filename + '.png'))
    mask = load_mask(os.path.join(path_to_masks, filename + '.npy'))
    masked_crop = apply_mask(crop, mask)

    return masked_crop

def generate_ball_illum_distribs(list_of_masks,
                                 path_to_ball_png,
                                 path_to_masks,
                                 path_to_save_ball_distrib_png,
                                 path_to_save_ball_distrib_npy,
                                 bins_num):

    for file in tqdm(list_of_masks):
        filename = file[:-4]
        ball_masked_crop = crop_mask(path_to_ball_png, filename, path_to_masks)
        crop_pixels = crop_to_pixels(ball_masked_crop)
        hist = generate_chroma_histogram(crop_pixels,
                                         HIST_BINS,
                                         HIST_RANGE,
                                         HIST_TARGET_SIZE,
                                         HIST_VERT_PADD,
                                         HIST_HORR_PADD)

        save_hist_png(path_to_save_ball_distrib_png,
                      filename,
                      hist)
        save_hist_npy(path_to_save_ball_distrib_npy,
                      filename,
                      hist)

def generate_spcube_illum_distribs(list_of_masks,
                                   path_to_spcube_est_csv,
                                   path_to_ball_distrib_png,
                                   path_to_save_spcube_distrib_png,
                                   path_to_save_spcube_distrib_npy,
                                   bins_num):

    csv_data = pd.read_csv(path_to_spcube_est_csv)

    for file in tqdm(list_of_masks):
        filename = file[:-4]
        spcube_illum_est = get_illum_est_from_csv(csv_data, filename)

        hist = generate_chroma_histogram(spcube_illum_est,
                                         HIST_BINS,
                                         HIST_RANGE,
                                         HIST_TARGET_SIZE,
                                         HIST_VERT_PADD,
                                         HIST_HORR_PADD)

        save_hist_png(path_to_save_spcube_distrib_png,
                      filename,
                      hist)
        save_hist_npy(path_to_save_spcube_distrib_npy,
                      filename,
                      hist)

def generate_spcube_illum_distribs_left(list_of_masks,
                                        path_to_spcube_est_csv,
                                        path_to_ball_distrib_png,
                                        path_to_save_spcube_distrib_png,
                                        path_to_save_spcube_distrib_npy,
                                        bins_num):

    csv_data = pd.read_csv(path_to_spcube_est_csv)

    for file in tqdm(list_of_masks):
        filename = file[:-4]
        left_spcube_illum_est = np.array([get_illum_est_from_csv(csv_data, filename)[0]])

        hist = generate_chroma_histogram(left_spcube_illum_est,
                                         HIST_BINS,
                                         HIST_RANGE,
                                         HIST_TARGET_SIZE,
                                         HIST_VERT_PADD,
                                         HIST_HORR_PADD)

        save_hist_png(path_to_save_spcube_distrib_png,
                      filename,
                      hist)
        save_hist_npy(path_to_save_spcube_distrib_npy,
                      filename,
                      hist)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_dataset',
                        type=Path),
    parser.add_argument('--path_to_spcube_est_csv',
                        type=Path),
    parser.add_argument('--bins_num', type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dp = args.path_to_dataset

    path_to_save_ball_distrib_png = dp / 'gt' / 'distribs_png'
    path_to_save_ball_distrib_npy = dp / 'gt' / 'distribs_npy'
    path_to_save_spcube_distrib_png = dp / 'spcube' / 'distribs_png'
    path_to_save_spcube_distrib_npy = dp / 'spcube' / 'distribs_npy'
    path_to_save_spcube_distrib_left_png = dp / 'spcube_left' / 'distribs_png'
    path_to_save_spcube_distrib_left_npy = dp / 'spcube_left' / 'distribs_npy'
    path_to_masks = dp / 'extra' / 'ball' / 'masks'

    list_of_masks = os.listdir(path_to_masks)

    # generate_ball_illum_distribs(list_of_masks,
    #                              args.path_to_ball_png,
    #                              args.path_to_masks,
    #                              args.path_to_save_ball_distrib_png,
    #                              args.path_to_save_ball_distrib_npy,
    #                              args.bins_num)

    generate_spcube_illum_distribs(list_of_masks,
                                   args.path_to_spcube_est_csv,
                                   path_to_save_ball_distrib_png,
                                   path_to_save_spcube_distrib_png,
                                   path_to_save_spcube_distrib_npy,
                                   args.bins_num)
    generate_spcube_illum_distribs_left(list_of_masks,
                                        args.path_to_spcube_est_csv,
                                        path_to_save_ball_distrib_png,
                                        path_to_save_spcube_distrib_left_png,
                                        path_to_save_spcube_distrib_left_npy,
                                        args.bins_num)
