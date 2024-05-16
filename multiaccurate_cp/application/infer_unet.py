import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain.model.data_generator import AerialImageDataset
from domain.model.unet import Unet


def infer(model_dir, model_name, data_dir, ml_set, output_dir, mean_RGB_values_path):

    rgb_means = np.load(mean_RGB_values_path)
    weight_path = glob.glob(os.path.join(model_dir, model_name, "checkpoints/*.ckpt"))
    assert len(weight_path) == 1, "Zero or more than 1 checkpoint file found !"
    weight_path = weight_path[0]
    model = Unet.load_from_checkpoint(weight_path)
    model.model.return_embeddings = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = AerialImageDataset(images_dir=os.path.join(data_dir, ml_set, "images"), split=ml_set, mean=rgb_means)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch[0].to(device)
            image_names = batch[1]
            preds, embeddings = model(images)
            preds = torch.sigmoid(preds)
            for i in range(len(images)):
                emb = embeddings[i, :, 0, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()
                if not os.path.exists(os.path.join(output_dir, ml_set, "embeddings")):
                    os.makedirs(os.path.join(output_dir, ml_set, "embeddings"))
                    os.makedirs(os.path.join(output_dir, ml_set, "pred_probas"))

                np.save(os.path.join(output_dir, ml_set, "embeddings", f"{image_names[i]}.npy"), emb)
                np.save(os.path.join(output_dir, ml_set, "pred_probas", f"{image_names[i]}.npy"), pred)
