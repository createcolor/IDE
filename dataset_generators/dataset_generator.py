import os
import sys
import numpy as np
import random

from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import imageio
import cv2

from dataset_generators.markup_generator import generate_crop_png, generate_crop_jpg, \
                             generate_crop_png_blurred, \
                             generate_markup, generate_mask
from dataset_generators.generate_illum_distrib import crop_mask, crop_to_pixels, \
                                   generate_chroma_histogram, save_hist_png, \
                                   generate_chroma_histogram_no_weights, \
                                   save_hist_npy, crop_to_distrib

from CubePlusPlus.code.utils.cube_io import png16_load, png16_save, jpg_load, jpg_save

# List of dataset folders
ball_crops_png = 'extra/ball/crops_png/'
ball_crops_png_orig = 'extra/ball/crops_png_orig/'
ball_crops_jpg = 'extra/ball/crops_jpg/'
cube3p_markup = 'markup/'
masks = 'extra/ball/masks/'
distribs_png = 'gt/distribs_png/'
distribs_npy = 'gt/distribs_npy/'
png = 'PNG/'

# List of image parameters

TARGET_IMG_SIZE = (192, 128)
CROP_SIZE = [704, 1056]

# List of histogram parameters

HIST_BINS = [116, 100]
HIST_RANGE = [[-1.74, 1.74], [-1, 2]]
HIST_TARGET_SIZE = [128, 128]
HIST_VERT_PADD = 6
HIST_HORR_PADD = 14

#kernel size for opening
OPEN_KERNEL = np.ones((3,2),np.uint8)

#kernel size for closing
CLOSE_KERNEL = np.ones((2,2),np.uint8)

#parameters for np.roll to return distribution to its original position
SHIFT = (-1, -2)

def generate_imgs(path_to_cube2p_png, path_to_save_cube3p_imgs, name, size):
    if not os.path.exists(path_to_save_cube3p_imgs):
        os.makedirs(path_to_save_cube3p_imgs)

    img = png16_load(os.path.join(path_to_cube2p_png, name))

    # crop cube from image
    img_wo_cube = img[:,:img.shape[1] - CROP_SIZE[0]]
    png16_save(os.path.join(path_to_save_cube3p_imgs, name),
               cv2.resize(img_wo_cube, size))

def dataset_generator(path_to_cube2p,
                      path_to_markup,
                      path_to_save_cube3p,
                      train_rate,
                      val_rate,
                      weights=False
                      ):
    path_to_cube2p_png = os.path.join(path_to_cube2p,'PNG/')
    path_to_cube2p_jpg = os.path.join(path_to_cube2p,'JPG/')
    path_to_cube2p_markup = os.path.join(path_to_cube2p, 'auxiliary/extra/gt_json/')
    path_to_cube2p_props = os.path.join(path_to_cube2p, 'properties.csv')
    path_to_unknown = os.path.join(path_to_cube2p, 'utils/unknown/')


    # splitting: 60% - train, 20% - validation, 20% - test
    list_of_images = os.listdir(path_to_cube2p_png)
    trainval_list_of_images, test_list_of_images = train_test_split(list_of_images,
                                                                 train_size=train_rate,
                                                                 random_state=57)

    train_list_of_images, validation_list_of_images = train_test_split(trainval_list_of_images,
                                                                 train_size=val_rate,
                                                                 random_state=57)

    datasets = {
                'test': test_list_of_images,
                'train':train_list_of_images,
                'val':  validation_list_of_images,
                }

    for dataset in datasets:
        path_to_save_ball_crops = os.path.join(path_to_save_cube3p, dataset, ball_crops_png)
        path_to_save_orig_ball_crops = os.path.join(path_to_save_cube3p, dataset, ball_crops_png_orig)
        path_to_save_ball_crops_jpg = os.path.join(path_to_save_cube3p, dataset, ball_crops_jpg)
        path_to_save_cube3p_markup = os.path.join(path_to_save_cube3p, dataset, cube3p_markup)
        path_to_save_cube3p_masks = os.path.join(path_to_save_cube3p, dataset, masks)
        path_to_save_cube3p_imgs = os.path.join(path_to_save_cube3p, dataset, png)

        print('Generating', dataset, 'mirror ball crops:')
        for filename in tqdm(datasets[dataset]):
            img = png16_load(os.path.join(path_to_cube2p_png, filename))
            img_jpg = jpg_load(os.path.join(path_to_cube2p_jpg, filename[:-4] + '.jpg'))
            name = filename.split('.')[0]

            generate_crop_png(img,
                        name,
                        path_to_markup,
                        path_to_save_ball_crops,
                        path_to_save_orig_ball_crops)
            # generate_crop_png_blurred(img,
            #             img_jpg,
            #             name,
            #             path_to_markup,
            #             path_to_save_ball_crops,
            #             path_to_cube2p_props,
            #             path_to_unknown)

        print('Generating', dataset, 'mirror ball crops jpg:')
        for filename in tqdm(datasets[dataset]):
            img = jpg_load(os.path.join(path_to_cube2p_jpg, filename[:-4] + '.jpg'))
            name = filename.split('.')[0]

            generate_crop_jpg(img,
                          name,
                          path_to_markup,
                          path_to_save_ball_crops_jpg)

        print('Generating', dataset, 'markup files:')
        for filename in tqdm(datasets[dataset]):
            img = png16_load(os.path.join(path_to_cube2p_png, filename))
            name = filename.split('.')[0]

            generate_markup(img,
                            name,
                            path_to_markup,
                            path_to_cube2p_markup,
                            path_to_save_cube3p_markup)

        print('Generating', dataset, 'masks for good pixels of mirror ball:')
        for filename in tqdm(datasets[dataset]):
            img = png16_load(os.path.join(path_to_cube2p_png, filename))
            name = filename.split('.')[0]

            generate_mask(path_to_save_ball_crops,
                        path_to_save_cube3p_markup,
                        name,
                        path_to_save_cube3p_masks,
                        path_to_cube2p_props)

        list_of_masks = os.listdir(path_to_save_cube3p_masks)

        # Attention: from this point I'm iterating over for loop
        print('Generating', dataset, 'illumination distribution:')
        for filename in tqdm(list_of_masks):
            name = filename.split('.')[0]

            hist = crop_to_distrib(path_to_save_ball_crops,
                                   name,
                                   path_to_save_cube3p_masks,
                                   weights)
            hist_orig = crop_to_distrib(path_to_save_orig_ball_crops,
                                        name,
                                        path_to_save_cube3p_masks)

            hist[hist_orig == 0] = 0

            hist = cv2.morphologyEx(hist, cv2.MORPH_OPEN, OPEN_KERNEL)
            hist = cv2.morphologyEx(hist, cv2.MORPH_CLOSE, CLOSE_KERNEL)
            hist = np.roll(hist, SHIFT, axis = (0,1))

            if np.max(hist) != 0: #this is needed because some distributions might not have points. should be changed later.

                save_hist_png(os.path.join(path_to_save_cube3p, dataset, distribs_png),
                        name,
                        hist)

                save_hist_npy(os.path.join(path_to_save_cube3p, dataset, distribs_npy),
                            filename,
                            hist)

        # list_of_masks = os.listdir(path_to_save_cube3p_masks)
        list_of_gt = os.listdir(os.path.join(path_to_save_cube3p, dataset, distribs_npy))

        print('Generating', dataset, 'images:')
        # for filename in tqdm(list_of_masks):
        for filename in tqdm(list_of_gt):
            name = filename.split('.')[0]
            generate_imgs(path_to_cube2p_png,
                        path_to_save_cube3p_imgs,
                        name + '.png',
                        TARGET_IMG_SIZE)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_cube2p',
                        default=r'../../data/Cube++/') #png
    parser.add_argument('--path_to_markup',
                        default=r'../markup/final/')
    parser.add_argument('--path_to_save_cube3p',
                        default=r'../../data/cube3p_v9')
    parser.add_argument('-tr', '--train_rate',
                        default= 0.8, type=int)
    parser.add_argument('-val', '--val_rate',
                        default= 0.75, type=int)
    parser.add_argument('-w', '--weights',
                        action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_generator(args.path_to_cube2p,
                      args.path_to_markup,
                      args.path_to_save_cube3p,
                      args.train_rate,
                      args.val_rate,
                      args.weights)
