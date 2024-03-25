#!/usr/bin/env python3

from clidantic import Parser
from tqdm.contrib.logging import logging_redirect_tqdm

from application import create_patches, train_unet, infer_unet, train_resnet, infer_resnet, infer_pranet
from config import InferPolypConfig, InferResConfig, InferUnetConfig, PatchesConfig, TrainResConfig, TrainUnetConfig

cli = Parser()


@cli.command()
def prepare(patches_config: PatchesConfig):
    return create_patches.write_patches(
        patches_config.data_path,
        patches_config.patch_size,
        patches_config.pad_size,
        patches_config.overlap,
    )


@cli.command()
def train_segmentation(train_config: TrainUnetConfig):
    return train_unet.train(config=train_config)


@cli.command()
def infer(infer_config: InferUnetConfig):
    return infer_unet.infer(
        infer_config.model_dir,
        infer_config.model_name,
        infer_config.data_dir,
        infer_config.ml_set,
        infer_config.output_dir,
        infer_config.mean_RGB_values_path
    )


@cli.command()
def train_residual(train_config: TrainResConfig):
    return train_resnet.train(config=train_config)


@cli.command()
def infer_residual(infer_config: InferResConfig):
    return infer_resnet.infer_resnet(
        infer_config.model_dir,
        infer_config.model_name,
        infer_config.data_dir,
        infer_config.pred_proba_dir,
        infer_config.ml_set,
    )


@cli.command()
def infer_polyp(infer_polyp: InferPolypConfig):
    return infer_pranet.infer(
        infer_polyp.model_dir,
        infer_polyp.data_dir,
        infer_polyp.ml_set,
        infer_polyp.output_dir,
    )


if __name__ == '__main__':
    with logging_redirect_tqdm():
        cli()
