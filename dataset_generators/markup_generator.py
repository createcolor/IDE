import os
import math
import sys

import numpy as np
import pandas as pd
import json
import argparse
from matplotlib import path
import cv2
from bm3d import bm3d_rgb

from tqdm import tqdm
from skimage import draw
from CubePlusPlus.code.utils.cube_io import png16_load, png16_save, jpg_save

# Size of SpyderCube crop for the whole dataset
cube_crop_size = np.array([1000, 700])

#bm3d parameter
PSD = 200

def create_mask_contour(markup_file, shape):
    points = np.array(read_points_from_markup_for_mask(markup_file), dtype=int)

    mask = np.zeros([shape[0], shape[1]], dtype=np.bool8)
    y, x = draw.polygon(points[:,1], points[:,0], shape=shape)

    mask[y, x] = 1

    return mask

def generate_mask(path_to_crop,
                  path_to_markup,
                  filename,
                  path_to_save_mask,
                  path_to_cube2p_props):

    props = pd.read_csv(path_to_cube2p_props) #will be deleted later

    markup_fname = os.path.join(path_to_markup, filename + '.json')
    with open(markup_fname) as f:
        js = json.load(f)
        img_shape = cube_crop_size
        max_lvl = js["max_lvl"]

    p1, p2 = calculate_crop_coordinates_for_mask(path_to_markup, filename, img_shape)
    crop_shape = np.array([p2[0]- p1[0], p2[1]- p1[1]])

    mask_contour = create_mask_contour(markup_fname, crop_shape)

    crop = png16_load(os.path.join(path_to_crop, filename + '.png'))
    crop_max_channelwise = np.amax(crop, axis=2)
    crop_min_channelwise = np.amin(crop, axis=2)

    mask_not_saturated = np.zeros(crop_max_channelwise.shape, dtype=np.bool8)
    mask_not_saturated[crop_max_channelwise < min(max_lvl)] = 1

    mask_not_black = np.zeros(crop_min_channelwise.shape, dtype=np.bool8)
    mask_not_black[crop_min_channelwise > 2100] = 1

    mask_saturated = np.zeros(crop_max_channelwise.shape, dtype=np.bool8)
    mask_saturated[crop_max_channelwise >= min(max_lvl)] = 1
    saturated_area = np.sum(mask_saturated)

    if not os.path.exists(path_to_save_mask):
        os.makedirs(path_to_save_mask)

    mask = mask_contour & mask_not_saturated & mask_not_black

    if saturated_area < np.sum(mask) / 2:
        mask_file = os.path.join(path_to_save_mask, filename)
        mask[int(mask.shape[0]*1/2):, :] = 0

        indicator = props[props['image'] == filename]['place'].values[0]
        if indicator != 'unknown':

            np.save(mask_file, mask)

def compute_chanelvise_satlvl(img):
    rmax, gmax, bmax = np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2])
    # This minus according to CubePlusPlus recommendation:
    rgb_thd = np.array([rmax, gmax, bmax]) - 50

    return rgb_thd

def save_new_markup(path_to_cube2p,
                    path_to_save_cube3p,
                    points,
                    maxval):

    with open(path_to_cube2p) as json_cube2p:
        cube2p_markup = json.load(json_cube2p)

        with open(path_to_save_cube3p, 'w') as json_cube3p:
            cube3p_markup = cube2p_markup

            cube3p_markup['mirror_ball_points'] = points.tolist()
            cube3p_markup['max_lvl'] = maxval.tolist()

            json.dump(cube3p_markup, json_cube3p, indent=2)

def generate_markup(img,
                    filename,
                    path_to_markup,
                    path_to_cube2_markup,
                    path_to_save_markup):

    markup_fname  = os.path.join(path_to_markup, filename + '.jpg.json')
    points        = read_points_from_markup(markup_fname)
    p1, _         = calculate_crop_borders(points)
    crop_boundary = np.array(p1)

    # shift of coordinates to ball crop from cube crop
    points -= crop_boundary[::-1]
    channels_maxlvl = compute_chanelvise_satlvl(img)

    if not os.path.exists(path_to_save_markup):
        os.makedirs(path_to_save_markup)

    save_new_markup(os.path.join(path_to_cube2_markup, filename + '.json'),
                    os.path.join(path_to_save_markup,  filename + '.json'),
                    points,
                    channels_maxlvl)

def read_points_from_markup(markup_file):
    with open(markup_file) as json_file:
        json_data = json.load(json_file)

    return json_data["objects"][0]["data"]

def read_points_from_markup_for_mask(markup_file):
    with open(markup_file) as json_file:
        json_data = json.load(json_file)

    return json_data["mirror_ball_points"]

def calculate_crop_borders(points):
    points = np.array(points)

    x_sort = sorted(points.T[0])
    y_sort = sorted(points.T[1])

    p1 = [math.floor(y_sort[0]), math.floor(x_sort[0])]
    p2 = [math.ceil(y_sort[-1]), math.ceil(x_sort[-1])]

    return p1, p2

def calculate_crop_coordinates(path_to_markup, filename, img_shape):
    markup_fname = os.path.join(path_to_markup, filename + '.jpg.json')
    points       = read_points_from_markup(markup_fname)

    crop_p1, crop_p2 = calculate_crop_borders(points)

    img_shape = np.array(img_shape)[:-1]
    p1 = img_shape - cube_crop_size + crop_p1
    p2 = img_shape - cube_crop_size + crop_p2

    return [p1, p2]

def calculate_crop_coordinates_for_mask(path_to_markup, filename, img_shape):
    markup_fname = os.path.join(path_to_markup, filename + '.json')
    points       = read_points_from_markup_for_mask(markup_fname)

    crop_p1, crop_p2 = calculate_crop_borders(points)

    img_shape = np.array(img_shape)[:-1]
    p1 = img_shape - cube_crop_size + crop_p1
    p2 = img_shape - cube_crop_size + crop_p2

    return [p1, p2]

def generate_crop_png(img, filename, path_to_markup, \
                      path_to_save_crop, path_to_save_orig_crop):
    [p1, p2] = calculate_crop_coordinates(path_to_markup,
                                          filename,
                                          np.shape(img))

    path_to_save = os.path.join(path_to_save_crop, filename + '.png')
    path_to_save_orig = os.path.join(path_to_save_orig_crop, filename + '.png')
    ball_crop = img[p1[0]:p2[0], p1[1]:p2[1], :]
    bm3d_ball_crop = bm3d_rgb(ball_crop, PSD).astype(np.uint16)

    if not os.path.exists(path_to_save_crop):
        os.makedirs(path_to_save_crop)
    if not os.path.exists(path_to_save_orig_crop):
        os.makedirs(path_to_save_orig_crop)


    png16_save(path_to_save, bm3d_ball_crop)
    png16_save(path_to_save_orig, ball_crop)

def generate_crop_png_blurred(img, img_jpg, filename, path_to_markup,
                              path_to_save_crop, path_to_cube2p_props,
                              path_to_unknown):
    [p1, p2] = calculate_crop_coordinates(path_to_markup,
                                          filename,
                                          np.shape(img))

    props = pd.read_csv(path_to_cube2p_props)
    indicator = props[props['image'] == filename]['place'].values[0]
    if indicator == 'unknown':
        path_to_save_unknown = os.path.join(path_to_unknown, filename + '.jpg')
        jpg_save(path_to_save_unknown, img_jpg)


    path_to_save = os.path.join(path_to_save_crop, filename + '.png')
    ball_crop = img[p1[0]:p2[0], p1[1]:p2[1], :]
    ball_crop = blur(ball_crop, indicator)

    if not os.path.exists(path_to_save_crop):
        os.makedirs(path_to_save_crop)

    png16_save(path_to_save, ball_crop)

def generate_crop_jpg(img, filename, path_to_markup, path_to_save_crop):
    [p1, p2] = calculate_crop_coordinates(path_to_markup,
                                          filename,
                                          np.shape(img))

    path_to_save = os.path.join(path_to_save_crop, filename + '.jpg')
    ball_crop = img[p1[0]:p2[0], p1[1]:p2[1], :]

    if not os.path.exists(path_to_save_crop):
        os.makedirs(path_to_save_crop)

    jpg_save(path_to_save, ball_crop)

def blur(img, indicator):
    if indicator == 'outdoor':
        size = 3
        sigma = 3
    elif indicator == 'indoor':
        size = 7
        sigma = 7
    else:
        size = 3
        sigma = 3
    ksize = (size, size)
    img = cv2.GaussianBlur(img, ksize, sigma)
    return img

def generate_data(path_to_source_png,
                  path_to_source_markup,
                  path_to_save_crop,
                  path_to_save_mask,
                  path_to_cube2p_markup,
                  path_to_save_markup):

    list_of_source_files = os.listdir(path_to_source_png)

    for filename in tqdm(list_of_source_files):
        img = png16_load(os.path.join(path_to_source_png, filename))

        # generate and save crop
        generate_crop(img,
                      filename.split('.')[0],
                      path_to_source_markup,
                      path_to_save_crop)

        # saving_new_markup
        generate_markup(img,
                        filename.split('.')[0],
                        path_to_source_markup,
                        path_to_cube2p_markup,
                        path_to_save_markup)

        # saving mask
        generate_mask(path_to_save_crop,
                      path_to_save_markup,
                      filename.split('.')[0],
                      path_to_save_mask)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_source',
                        default=r'../../data/Cube++/PNG/') #png
    parser.add_argument('--path_to_markup',
                        default=r'../markup/final/')
    parser.add_argument('--path_to_cube2p_markup',
                        default=r'../../data/Cube++/auxiliary/extra/gt_json/')
    parser.add_argument('--path_to_save_crop',
                        default=r'../../data/cube3p_new/ball_crops_png')
    parser.add_argument('--path_to_save_mask',
                        default=r'../../data/cube3p_new/masks')
    parser.add_argument('--path_to_save_markup',
                        default=r'../../data/cube3p_new/markup')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    generate_data(args.path_to_source,
                  args.path_to_markup,
                  args.path_to_save_crop,
                  args.path_to_save_mask,
                  args.path_to_cube2p_markup,
                  args.path_to_save_markup)
