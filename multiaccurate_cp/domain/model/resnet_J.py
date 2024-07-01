import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from domain.model.data_generator import JOptimDataset
from utils.multiaccurate_mlp import J


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, mask, mask_pred, pred_th, alpha):
        # Loss with ridge regularization

        return torch.tensor(J(
            mask.detach().cpu().numpy(),
            mask_pred.detach().cpu().numpy(),
            pred_th.detach().cpu().numpy(),
            alpha, len(mask)
        ))


class Resnet(pl.LightningModule):

    def __init__(self, resnet, embedding_size, input_size, target_recall, lr=1e-3, weight_decay=1e-4,
                 scheduler_step_size=10, scheduler_gamma=0.1,
                 train_images_dir="./", train_labels_dir="./", train_probas_dir="./",
                 mean=(0., 0., 0.), train_batch_size=10, val_batch_size=10,
                 num_workers=8):
        super().__init__()
        self.target_recall = target_recall
        self.model = torch.hub.load('pytorch/vision:v0.9.0', resnet, pretrained=False)
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
        self.model.conv1 = torch.nn.Conv2d(
            4, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride, padding=old_conv1.padding, bias=False
        )
        self.input_size = input_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = scheduler_step_size
        self.gamma = scheduler_gamma
        # Data params
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.train_probas_dir = train_probas_dir
        self.mean = mean
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.automatic_optimization = False
        self.criterion = CustomLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_th = self.forward(x)
        loss = self.criterion(
            y, x[:, -1], pred_th, 1 - self.target_recall
        )
        self.log("train_loss", loss)
        print(loss)
        pred_th.backward(loss.repeat(len(x)).to("cuda")[:, None])
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_th = self.forward(x)
        loss = self.criterion(
            y, x[:, -1], pred_th, 1 - self.target_recall
        )

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optim]

    def train_dataloader(self):
        full_dataset = JOptimDataset(
            self.train_images_dir,
            self.train_labels_dir,
            self.train_probas_dir,
            self.target_recall,
            self.input_size,
            mode="train", mean=self.mean
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
        full_dataset = JOptimDataset(
            self.train_images_dir,
            self.train_labels_dir,
            self.train_probas_dir,
            self.target_recall,
            self.input_size,
            mode="train", mean=self.mean
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
