import numpy as np
from patchify import patchify


def pad_image_and_label(img, label, pad_size):

    padded_img = np.pad(
        img,
        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="symmetric"
    )
    padded_label = np.pad(
        label,
        pad_width=((pad_size, pad_size), (pad_size, pad_size)),
        mode="symmetric"
    )

    return padded_img, padded_label


def create_patches(img, patch_size, overlap):
    step = patch_size - overlap
    patch_size = (patch_size, patch_size, 3) if img.ndim == 3 else (patch_size, patch_size)
    image_patches = patchify(img, patch_size, step=step).squeeze()
    return image_patches
