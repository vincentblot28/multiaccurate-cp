import glob
import os
import pathlib

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
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
            target_recall=.9, mode="train",
            mean=[0., 0., 0.], return_img_path=False,
            model_input="images", polyp=False
    ):
        self.model_input = model_input
        self.list_pred_probas_path = sorted(glob.glob(os.path.join(pred_probas_dir, "*.npy")))
        self.list_images_path = sorted(glob.glob(os.path.join(images_dir, "*")))
        if labels_dir is not None:
            self.list_masks_path = sorted(glob.glob(os.path.join(labels_dir, "*")))
            self._check_alignement(self.list_images_path, self.list_masks_path)
        self.mode = mode
        self.return_img_path = return_img_path
        self.target_recall = target_recall
        self.std = [1., 1., 1.]
        self.mean = [0., 0., 0.]
        self.polyp = polyp
        self.polyp_size = 352
        if mean is not None:
            self.mean = [i/255. for i in mean]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        if not self.polyp:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                A.Resize(self.polyp_size, self.polyp_size),
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

    def _load_input(self, path_input_images, path_input_probas):
        if self.model_input == "images":
            model_input = cv2.cvtColor(cv2.imread(path_input_images), cv2.COLOR_BGR2RGB)
            return self.transform(image=model_input)["image"]
        elif self.model_input == "probas":
            probas = np.load(path_input_probas)[:, :, np.newaxis]
            probas = np.transpose(probas, (2, 0, 1))
            return probas
        elif self.model_input == "image_and_probas":
            input1 = cv2.cvtColor(cv2.imread(path_input_images), cv2.COLOR_BGR2RGB)
            input1_trfm = self.transform(image=input1)["image"]
            input2 = np.load(path_input_probas)
            input2 = cv2.resize(input2, (input1_trfm.shape[2], input1_trfm.shape[1]))[:, :, np.newaxis]
            input2 = np.transpose(input2, (2, 0, 1))
            model_input = np.concatenate([input1_trfm, input2], axis=0)
            return torch.tensor(model_input)

    def _load_input_and_th(self, path_images, path_mask, path_pred_probas):

        model_input = self._load_input(path_images, path_pred_probas)
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


class PolypDataset(Dataset):
    def __init__(self, image_root, testsize=352):
        self.testsize = testsize
        self.images = [
            os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.transform = A.Compose([
            A.Resize(self.testsize, self.testsize),
            A.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        self.size = len(self.images)

    def __len__(self):
        return self.size

    def load_data(self, index):
        image = self.rgb_loader(self.images[index])
        img_shape = image.shape
        image = self.transform(image=image)["image"]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, name, img_shape

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return np.array(img.convert('RGB'))

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, index):
        return self.load_data(index)
