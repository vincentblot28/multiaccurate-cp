import random
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, F1Score
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss

from domain.model.data_generator import AerialImageDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)


class UnetModule(nn.Module):

    def __init__(self, initial_filters_nb=32, dropout=0.5):
        super(UnetModule, self).__init__()
        ks = 3  # Kernel size
        out_channels = 1
        features = initial_filters_nb

        self.return_embeddings = False

        self.encoder1 = UnetModule.conv_block(
            3, features, kernel_size=ks,
            dropout=dropout, layer_name="encoder1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UnetModule.conv_block(
            features, features*2, kernel_size=ks,
            dropout=dropout, layer_name="encoder2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UnetModule.conv_block(
            features*2, features*4, kernel_size=ks,
            dropout=dropout, layer_name="encoder3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UnetModule.conv_block(
            features*4, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder5"
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder6"
        )
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder7 = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder7"
        )
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder8 = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder8"
        )
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder9 = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="encoder9"
        )

        self.pool9 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UnetModule.conv_block(
            features*8, features*8, kernel_size=ks,
            dropout=dropout, layer_name="bottleneck"
        )
        self.upconv9 = nn.ConvTranspose2d(
            features*16, features*8, kernel_size=2, stride=2
        )
        self.decoder9 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder9"
        )
        self.upconv8 = nn.ConvTranspose2d(
            features*8, features*8, kernel_size=2, stride=2
        )
        self.decoder8 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder8"
        )
        self.upconv7 = nn.ConvTranspose2d(
            features*8, features*8, kernel_size=2, stride=2
        )
        self.decoder7 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder7"
        )
        self.upconv6 = nn.ConvTranspose2d(
            features*8, features*8, kernel_size=2, stride=2
        )
        self.decoder6 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder6"
        )
        self.upconv5 = nn.ConvTranspose2d(
            features*8, features*8, kernel_size=2, stride=2
        )
        self.decoder5 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder5"
        )
        self.upconv4 = nn.ConvTranspose2d(
            features*8, features*8, kernel_size=2, stride=2
        )
        self.decoder4 = UnetModule.conv_block(
            features*16, features*8, kernel_size=ks,
            dropout=dropout, layer_name="decoder4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            features*8, features*4, kernel_size=2, stride=2
        )
        self.decoder3 = UnetModule.conv_block(
            features*8, features*4, kernel_size=ks,
            dropout=dropout, layer_name="decoder3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            features*4, features*2, kernel_size=2, stride=2
        )
        self.decoder2 = UnetModule.conv_block(
            features*4, features*2, kernel_size=ks,
            dropout=dropout, layer_name="decoder2"
        )
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UnetModule.conv_block(
            features*2, features, kernel_size=ks,
            dropout=None, layer_name="decoder1"
        )

        self.output = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool1(encoder1))
        encoder3 = self.encoder3(self.pool2(encoder2))
        encoder4 = self.encoder4(self.pool3(encoder3))  # 71x71
        encoder5 = self.encoder5(self.pool4(encoder4))  # 35x35
        encoder6 = self.encoder6(self.pool5(encoder5))  # 17x17
        encoder7 = self.encoder7(self.pool6(encoder6))  # 8x8
        encoder8 = self.encoder8(self.pool7(encoder7))  # 4x4
        # encoder9 = self.encoder9(self.pool8(encoder8))  # 2x2

        bottleneck = self.bottleneck(self.pool4(encoder8))

        # decoder9 = self.decode(encoder9, bottleneck, self.decoder9, self.upconv9)
        decoder8 = self.decode(encoder8, bottleneck, self.decoder8, self.upconv8)
        decoder7 = self.decode(encoder7, decoder8, self.decoder7, self.upconv7)
        decoder6 = self.decode(encoder6, decoder7, self.decoder6, self.upconv6)
        decoder5 = self.decode(encoder5, decoder6, self.decoder5, self.upconv5)
        decoder4 = self.decode(encoder4, decoder5, self.decoder4, self.upconv4)
        decoder3 = self.decode(encoder3, decoder4, self.decoder3, self.upconv3)
        decoder2 = self.decode(encoder2, decoder3, self.decoder2, self.upconv2)
        decoder1 = self.decode(encoder1, decoder2, self.decoder1, self.upconv1)

        output = self.output(decoder1)
        if self.return_embeddings:
            return output, bottleneck
        else:
            return output

    def decode(self, encoder, decoder, decoder_func, upconv_func):
        decoder_1 = upconv_func(decoder)

        # SKIP CONNECTIONS
        if encoder.shape[2:] != decoder_1.shape[2:]:
            pad_row = encoder.shape[2] - decoder_1.shape[2]
            pad_col = encoder.shape[3] - decoder_1.shape[3]
            decoder_1 = nn.functional.pad(
                decoder_1, (0, pad_col, 0, pad_row), mode="replicate"
            )
        decoder_1 = torch.cat([encoder, decoder_1], dim=1)
        decoder_1 = decoder_func(decoder_1)

        return decoder_1

    @staticmethod
    def conv_block(in_channels, features, kernel_size, dropout, layer_name):
        layers = OrderedDict([
            (
                layer_name+"_conv1",
                nn.Conv2d(in_channels=in_channels, out_channels=features,
                          kernel_size=kernel_size, padding=1, bias=False)
            ),
            (layer_name+"_bn1", nn.BatchNorm2d(num_features=features)),
            (layer_name+"_act1", nn.ReLU(inplace=True)),
            (
                layer_name+"_conv2",
                nn.Conv2d(in_channels=features, out_channels=features,
                          kernel_size=kernel_size, padding=1, bias=False)
            ),
            (layer_name+"_bn2", nn.BatchNorm2d(num_features=features)),
            (layer_name+"_act2", nn.ReLU(inplace=True))
        ])
        if dropout is not None:
            layers[layer_name+"_dropout"] = nn.Dropout(p=dropout)

        return nn.Sequential(layers)


class Unet(pl.LightningModule):

    def __init__(self, initial_filters_nb=32, dropout=0.5, lr=1e-3,
                 train_images_dir="./", train_labels_dir="./",
                 val_images_dir="./", val_labels_dir="./",
                 mean=(0., 0., 0.),
                 train_batch_size=10, val_batch_size=10, num_workers=8):
        super().__init__()

        self.model = UnetModule(initial_filters_nb, dropout)
        self.lr = lr

        # Data params
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.val_images_dir = val_images_dir
        self.val_labels_dir = val_labels_dir
        self.mean = mean
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.criterion = SoftBCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")
        self.iou = JaccardIndex(num_classes=2, ignore_index=0, task="binary")
        self.fscore = F1Score(task="binary")

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
        x = torch.sigmoid(x.squeeze())
        self.accuracy(x, y)
        self.iou(x, y)
        self.fscore(x, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.log("val_iou", self.iou, prog_bar=True)
        self.log("val_fscore", self.fscore, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optim]

    def train_dataloader(self):
        dataset = AerialImageDataset(
            self.train_images_dir,
            self.train_labels_dir,
            split="train", mean=self.mean
        )
        return DataLoader(
            dataset, batch_size=self.train_batch_size,
            shuffle=True, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=g,
        )

    def val_dataloader(self):
        dataset = AerialImageDataset(
            self.val_images_dir,
            self.val_labels_dir,
            split="val", mean=self.mean
        )
        return DataLoader(
            dataset, batch_size=self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=g,
        )
