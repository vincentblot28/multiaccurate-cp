import glob
import os
import pathlib

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class AerialImageDataset(Dataset):

    def __init__(self, images_dir, labels_dir=None, split="train", mean=[0., 0., 0.]):

        self.list_imgs_path = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        if labels_dir is not None:
            self.list_masks_path = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))
            self._check_alignement(self.list_imgs_path, self.list_masks_path)
        self.split = split

        self.std = [1., 1., 1.]
        self.mean = [0., 0., 0.]
        if mean is not None:
            self.mean = [i/255. for i in mean]
        if self.split == "train":
            self.transform = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

        else:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def __len__(self): return len(self.list_imgs_path)

    def _check_alignement(self, list_images_path, list_masks_path):
        assert len(list_images_path) == len(list_masks_path)
        for p1, p2 in tqdm(zip(list_images_path, list_masks_path)):
            assert pathlib.Path(p1).stem == pathlib.Path(p2).stem

    def get_image_path(self, idx):
        return self.list_imgs_path[idx]

    def get_label_path(self, idx):
        return self.list_masks_path[idx]

    def _load_img(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return self.transform(image=img)["image"]

    def _load_img_and_mask(self, path_img, path_mask):
        img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)
        mask = np.where(mask == 255, 1, 0)  # convert building pixels from 255 to 1

        transformed = self.transform(image=img, mask=mask)

        return transformed["image"], transformed["mask"]

    def __getitem__(self, idx):
        if self.split == "test":
            path_img = self.list_imgs_path[idx]
            filename = pathlib.Path(path_img).stem
            return self._load_img(path_img), filename

        path_img, path_mask = self.list_imgs_path[idx], self.list_masks_path[idx]
        return self._load_img_and_mask(path_img, path_mask)
