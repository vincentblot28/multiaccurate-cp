import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain.model.data_generator import PolypDataset
from domain.model.pranet import PraNet


def infer(model_dir, data_dir, ml_set, output_dir):

    weight_path = glob.glob(os.path.join(model_dir, "*.pth"))
    assert len(weight_path) == 1, "Zero or more than 1 checkpoint file found !"
    weight_path = weight_path[0]
    model = PraNet()
    model.load_state_dict(torch.load(weight_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = PolypDataset(image_root=os.path.join(data_dir, ml_set, "images"))
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch[0].to(device)
            image_names = batch[1]
            image_shapes = batch[2]
            _, _, _, preds, embeddings = model(images)
            for i in range(len(images)):
                pred = preds[i, 0].cpu().numpy()
                embedding = embeddings[i, 0].cpu().numpy()
                pred = cv2.resize(pred, (int(image_shapes[1][i]), int(image_shapes[0][i])))
                pred = torch.sigmoid(torch.tensor(pred) / 10).numpy()
                np.save(
                    os.path.join(
                        output_dir, ml_set, "pred_probas",
                        f"{image_names[i].split('.')[0]}.npy"
                    ), pred
                )
                np.save(
                    os.path.join(
                        output_dir, ml_set, "embeddings",
                        f"{image_names[i].split('.')[0]}.npy"
                    ), embedding
                )
