import datetime as dt
import os

import numpy as np
import pytorch_lightning as pl

from domain.model.resnet import Resnet
from utils.config_utils import store_config


def train(config):
    """
    Trains a U-Net model using the provided configuration.

    Args:
        config (Config): The configuration object containing the model parameters and paths.

    Returns:
        None
    """

    model_params = config.model.dict()
    ml_data_dir = config.ml_data_dir
    probas_dir = config.probas_dir
    output_dir = config.output_dir
    mean_RGB_values_path = config.mean_RGB_values_path
    polyp = config.polyp
    if mean_RGB_values_path is not None:
        mean_RGB_values = np.load(mean_RGB_values_path)
    else:
        mean_RGB_values = None

    train_images_dir = os.path.join(ml_data_dir, "res/images")
    train_labels_dir = os.path.join(ml_data_dir, "res/labels")
    train_probas_dir = os.path.join(probas_dir, "res/pred_probas")

    model_date = dt.datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = os.path.join(output_dir, model_date)
    ckpt_dir = os.path.join(model_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)
    store_config(config, os.path.join(model_dir, "config.yaml"))

    model = Resnet(
        resnet=model_params["resnet"],
        embedding_size=model_params["embedding_size"],
        target_recall=model_params["target_recall"],
        model_input=model_params["model_input"],
        lr=model_params["lr"],
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        train_probas_dir=train_probas_dir,
        mean=mean_RGB_values,
        train_batch_size=model_params["batch_size"],
        val_batch_size=model_params["batch_size"],
        num_workers=model_params["num_workers"],
        polyp=polyp
    )

    pl.utilities.model_summary.summarize(model)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mse",
        dirpath=ckpt_dir,
        filename="ckpt-{epoch:03d}-{val_mse:.5f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_mse",
        patience=model_params["patience"],
        mode="min",
    )

    # Training engine
    trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=model_params["epochs"],
        precision=16,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model)
