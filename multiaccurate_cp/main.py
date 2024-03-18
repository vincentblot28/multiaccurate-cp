#!/usr/bin/env python3

from clidantic import Parser
from tqdm.contrib.logging import logging_redirect_tqdm

from application import create_patches, train_unet
from config import PatchesConfig, TrainConfig

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
def train(train_config: TrainConfig):
    return train_unet.train(config=train_config)


if __name__ == '__main__':
    with logging_redirect_tqdm():
        cli()
