import glob
import os
import pathlib

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.ml_utils import get_threshold


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
        if self.split in ["test", "cal", "res"]:
            path_img = self.list_imgs_path[idx]
            filename = pathlib.Path(path_img).stem
            return self._load_img(path_img), filename

        path_img, path_mask = self.list_imgs_path[idx], self.list_masks_path[idx]
        return self._load_img_and_mask(path_img, path_mask)


class ResidualDataset(Dataset):

    def __init__(
            self, images_dir, labels_dir=None, pred_probas_dir=None,
            target_recall=.9, mode="train", mean=[0., 0., 0.], return_img_path=False,
            model_input="images"
    ):
        self.model_input = model_input
        self.list_pred_probas_path = sorted(glob.glob(os.path.join(pred_probas_dir, "*.npy")))
        self.list_images_path = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        if labels_dir is not None:
            self.list_masks_path = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))
            self._check_alignement(self.list_images_path, self.list_masks_path)
        self.mode = mode
        self.return_img_path = return_img_path
        self.target_recall = target_recall
        self.std = [1., 1., 1.]
        self.mean = [0., 0., 0.]
        if mean is not None:
            self.mean = [i/255. for i in mean]
        if self.mode == "train":
            self.transform = A.Compose([
                # A.Flip(p=0.5),
                # A.RandomRotate90(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

        else:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.list_images_path)

    def _check_alignement(self, list_images_path, list_masks_path):
        assert len(list_images_path) == len(list_masks_path)
        for p1, p2 in tqdm(zip(list_images_path, list_masks_path)):
            assert pathlib.Path(p1).stem == pathlib.Path(p2).stem

    def get_image_path(self, idx):
        return self.list_images_path[idx]

    def get_label_path(self, idx):
        return self.list_masks_path[idx]

    def _load_input(self, path_input_images, path_input_probas, path_input_embeddings):
        if self.model_input == "images":
            model_input = cv2.cvtColor(cv2.imread(path_input_images), cv2.COLOR_BGR2RGB)
            return self.transform(image=model_input)["image"]
        elif self.model_input == "probas":
            probas = np.load(path_input_probas)[:, :, np.newaxis]
            probas = np.transpose(probas, (2, 0, 1))
            return probas
        elif self.model_input == "embeddings":
            return np.load(path_input_embeddings)
        elif self.model_input == "image_and_probas":
            input1 = cv2.cvtColor(cv2.imread(path_input_images), cv2.COLOR_BGR2RGB)
            input1_trfm = self.transform(image=input1)["image"]
            input2 = np.load(path_input_probas)[:, :, np.newaxis]
            input2 = np.transpose(input2, (2, 0, 1))
            model_input = np.concatenate([input1_trfm, input2], axis=0)
            return torch.tensor(model_input)

    def _load_input_and_th(self, path_images, path_mask, path_pred_probas):
        path_embeddings = path_pred_probas.replace("pred_probas", "embeddings")
        model_input = self._load_input(path_images, path_pred_probas, path_embeddings)
        label = cv2.imread(path_mask, cv2.COLOR_BGR2GRAY) / 255
        pred_probas = np.load(path_pred_probas)
        if np.sum(label) == 0:
            threshold = 1
        else:
            threshold = get_threshold(pred_probas, label, self.target_recall)

        return model_input, threshold

    def __getitem__(self, idx):
        if self.mode in ["test", "cal"]:
            path_input = self.list_images_path[idx]
            filename = pathlib.Path(path_input).stem
            return self._load_input(path_input), filename

        path_input, path_mask, path_pred_probas = (
            self.list_images_path[idx],
            self.list_masks_path[idx],
            self.list_pred_probas_path[idx]
        )
        if self.return_img_path:
            return self._load_input_and_th(path_input, path_mask, path_pred_probas), path_input
        else:
            return self._load_input_and_th(path_input, path_mask, path_pred_probas)
