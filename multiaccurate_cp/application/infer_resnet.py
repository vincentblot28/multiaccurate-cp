import glob
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from domain.model.data_generator import ResidualDataset
from domain.model.resnet import Resnet


def infer_resnet(models_dir, model_name, data_dir, pred_proba_dir, ml_set):
    file_path = os.path.join(models_dir, model_name, "config.yaml")

    # Read the YAML file
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    ds = ResidualDataset(
        images_dir=os.path.join(data_dir, ml_set, "images"),
        labels_dir=os.path.join(data_dir, ml_set, "labels"),
        pred_probas_dir=os.path.join(pred_proba_dir, ml_set, "pred_probas"),
        target_recall=config["model"]["target_recall"],
        return_img_path=True,
        model_input=config["model"]["model_input"],
    )
    models_dir = glob.glob(os.path.join(models_dir, model_name, "checkpoints", "*.ckpt"))
    model = Resnet.load_from_checkpoint(
        models_dir[0], resnet=config["model"]["resnet"], model_input=config["model"]["model_input"],
        embedding_size=config["model"]["embedding_size"], target_recall=config["model"]["target_recall"]
    )
    model.model.fc = nn.Sequential(*[model.model.fc[i] for i in range(len(model.model.fc) - 1)])

    if not os.path.exists(os.path.join(pred_proba_dir, ml_set, "res_embeddings", model_name)):
        os.makedirs(os.path.join(pred_proba_dir, ml_set, "res_embeddings", model_name))
    for (model_input, _), img_path in tqdm(ds):
        embedding = model(torch.tensor(model_input[np.newaxis, ...]).to("cuda")).cpu().detach().numpy()[0, :]
        np.save(
            os.path.join(
                pred_proba_dir, ml_set,
                "res_embeddings", model_name,
                os.path.basename(img_path).replace(".tif", ".npy")
            ),
            embedding
        )
