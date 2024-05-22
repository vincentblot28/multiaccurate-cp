import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from domain.model.data_generator import ResidualDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)


class Resnet(pl.LightningModule):

    def __init__(self, resnet, embedding_size, target_recall,
                 model_input, lr=1e-3,
                 train_images_dir="./", train_labels_dir="./", train_probas_dir="./",
                 mean=(0., 0., 0.), train_batch_size=10, val_batch_size=10,
                 num_workers=8, return_embeddings=False, polyp=False):
        super().__init__()
        self.model_input = model_input
        self.target_recall = target_recall
        self.model = torch.hub.load('pytorch/vision:v0.9.0', resnet, pretrained=True)
        self.polyp = polyp
        old_fc = self.model.fc
        old_fc_size = old_fc.weight.shape[1]
        if old_fc_size > embedding_size:
            modules = []
            while old_fc_size > embedding_size:
                modules.append(torch.nn.Linear(old_fc_size, old_fc_size // 2))
                old_fc_size = old_fc_size//2
            modules.append(torch.nn.Linear(old_fc_size, 1))
            self.model.fc = torch.nn.Sequential(*modules)
        else:
            self.model.fc = torch.nn.Linear(old_fc_size, 1)
        old_conv1 = self.model.conv1
        if self.model_input == "image_and_probas":
            self.model.conv1 = torch.nn.Conv2d(
                4, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding, bias=False
            )
        elif self.model_input == "probas":
            self.model.conv1 = torch.nn.Conv2d(
                1, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding, bias=False
            )
        self.return_embeddings = return_embeddings
        self.lr = lr

        # Data params
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.train_probas_dir = train_probas_dir
        self.mean = mean
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.criterion = MSELoss()
        self.mse = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.criterion(x.squeeze(), y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.criterion(x.squeeze(), y.float())
        self.mse(x.squeeze(), y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", self.mse, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optim]

    def train_dataloader(self):
        full_dataset = ResidualDataset(
            self.train_images_dir,
            self.train_labels_dir,
            self.train_probas_dir,
            self.target_recall,
            mode="train", mean=self.mean,
            model_input=self.model_input,
            polyp=self.polyp
        )
        train_set_size = int(0.8 * len(full_dataset))
        valid_set_size = len(full_dataset) - train_set_size
        seed = seed = torch.Generator().manual_seed(42)
        train_set, _ = data.random_split(full_dataset, [train_set_size, valid_set_size], generator=seed)

        return DataLoader(
            train_set, batch_size=self.train_batch_size,
            shuffle=True, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=g,
        )

    def val_dataloader(self):
        full_dataset = ResidualDataset(
            self.train_images_dir,
            self.train_labels_dir,
            self.train_probas_dir,
            self.target_recall,
            mode="train", mean=self.mean,
            model_input=self.model_input,
            polyp=self.polyp
        )
        train_set_size = int(0.8 * len(full_dataset))
        valid_set_size = len(full_dataset) - train_set_size
        seed = seed = torch.Generator().manual_seed(42)
        _, valid_set = data.random_split(full_dataset, [train_set_size, valid_set_size], generator=seed)

        return DataLoader(
            valid_set, batch_size=self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=g,
        )
