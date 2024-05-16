import itertools as it
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from domain.patches import create_patches, pad_image_and_label
from utils.filename_utils import get_image_set

TEST_NUMBER = 5
CAL_NUMBER = 31
RES_NUMBER = 34
PERCENTAGE_VAL = .2


def write_patches(data_path, patch_size, pad_size, overlap):
    """
    Write patches of images and labels to disk.

    Args:
        data_path (str): The path to the data directory.
        patch_size (int): The size of each patch.
        pad_size (int): The size of padding around each patch.
        overlap (int): The amount of overlap between patches.

    Returns:
        None
    """
    random.seed(42)
    mean_R, mean_G, mean_B = [], [], []
    for img_name in tqdm(os.listdir(os.path.join(data_path, "01_raw_images", "images"))):
        if img_name.split(".")[0] not in ["kitsap4", "kitsap5"]:  # Images that have wrong labels
            img = cv2.imread(os.path.join(data_path, "01_raw_images", "images", img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = cv2.imread(os.path.join(data_path, "01_raw_images", "gt", img_name), cv2.IMREAD_GRAYSCALE)
            padded_img, padded_label = pad_image_and_label(img, label, pad_size)
            ml_set = get_image_set(img_name, TEST_NUMBER, CAL_NUMBER, RES_NUMBER, PERCENTAGE_VAL)

            if ml_set == "train":
                mean_R.append(padded_img.mean(axis=0).mean(axis=0)[0])
                mean_G.append(padded_img.mean(axis=0).mean(axis=0)[1])
                mean_B.append(padded_img.mean(axis=0).mean(axis=0)[2])

            image_patches = create_patches(padded_img, patch_size, overlap)
            label_patches = create_patches(padded_label, patch_size, overlap)

            for ix, iy in it.product(range(image_patches.shape[0]), range(image_patches.shape[1])):
                ptc_name = img_name.split(".")[0] + f"_ptc_{str(ix).zfill(2)}_{str(iy).zfill(2)}"
                image_patch = image_patches[ix, iy]
                label_patch = label_patches[ix, iy]
                if not os.path.exists(os.path.join(data_path, "02_prepared_data_small", ml_set, "images")):
                    os.makedirs(os.path.join(data_path, "02_prepared_data_small", ml_set, "images"))
                    os.makedirs(os.path.join(data_path, "02_prepared_data_small", ml_set, "labels"))
                cv2.imwrite(
                    os.path.join(
                        data_path, "02_prepared_data_small", ml_set,
                        "images", ptc_name + ".tif"
                    ), image_patch
                )
                cv2.imwrite(
                    os.path.join(
                        data_path, "02_prepared_data_small", ml_set,
                        "labels", ptc_name + ".tif"
                    ), label_patch
                )

    rgb_means = np.array([np.mean(mean_R), np.mean(mean_G), np.mean(mean_B)])
    np.save(os.path.join(data_path, "01_raw_images", "rgb_means.npy"), rgb_means)
