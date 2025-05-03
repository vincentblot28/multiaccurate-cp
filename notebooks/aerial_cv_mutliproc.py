# %%
import sys
sys.path.append("/home/vblot/multiaccurate_cp/")
sys.path.append("/home/vblot/multiaccurate-cp/")
sys.path.append("/home/vblot/multiaccurate-cp/multiaccurate_cp/")

# %%
from copy import deepcopy
import os
import yaml

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from mapie.multi_label_classification import MapieMultiLabelClassifier
from scipy import integrate
from scipy.optimize import minimize
from tqdm import tqdm


# %%
BASE_DIR = "/mnt/ssd/multiaccurate/aerial"
MODELS_PATH = os.path.join(BASE_DIR, "03_model_weights/resnet")
MODEL_NAME = "20240322_1039"

# %%
# Specify the path to the YAML file
file_path = os.path.join(MODELS_PATH, MODEL_NAME, "config.yaml")

# Read the YAML file
with open(file_path, "r") as file:
    config = yaml.safe_load(file)
config

# %%
IMG_SIZE = 64
RESNET_EMBEDDING_SIZE = config["model"]["embedding_size"]


# %%
DIR_CAL_PRED_PROBAS = os.path.join(BASE_DIR, "04_predictions/cal/pred_probas")
DIR_CAL_LABELS = os.path.join(BASE_DIR, "02_prepared_data/cal/labels")
DIR_TEST_PRED_PROBAS = DIR_CAL_PRED_PROBAS.replace("cal", "test")
DIR_TEST_LABELS = DIR_CAL_LABELS.replace("cal", "test")

DIR_CAL_RES_EMB = os.path.join(BASE_DIR, "04_predictions/cal/res_embeddings", MODEL_NAME)
DIR_TEST_RES_EMB = DIR_CAL_RES_EMB.replace("cal", "test")




# %%
def load_data(pred_probas_dir, res_emb_dir, labels_dir):
    pred_probas = np.zeros((len(os.listdir(pred_probas_dir)), IMG_SIZE, IMG_SIZE))
    res_emb = np.zeros((len(os.listdir(res_emb_dir)), RESNET_EMBEDDING_SIZE))
    labels = np.zeros((len(os.listdir(labels_dir)), IMG_SIZE, IMG_SIZE))

    for i, (pred_proba_file, res_emb_file, label_file) in enumerate(
        tqdm(zip(
            sorted(os.listdir(pred_probas_dir)),
            sorted(os.listdir(res_emb_dir)),
            sorted(os.listdir(labels_dir))
        ))
    ):
        pred_proba = np.load(os.path.join(pred_probas_dir, pred_proba_file))
        pred_probas[i] = cv2.resize(pred_proba, (IMG_SIZE, IMG_SIZE))
        res_emb[i] = np.load(os.path.join(res_emb_dir, res_emb_file))
        label = cv2.imread(os.path.join(labels_dir, label_file), cv2.IMREAD_GRAYSCALE) / 255
        labels[i] = cv2.resize(label, (IMG_SIZE, IMG_SIZE)) > .5    
    return pred_probas, res_emb, labels


# %%
load_cal_pred_probas, load_cal_res_emb, load_cal_labels = load_data(
    DIR_CAL_PRED_PROBAS, DIR_CAL_RES_EMB, DIR_CAL_LABELS
)


# %%
load_test_pred_probas, load_test_res_emb, load_test_labels = load_data(
    DIR_TEST_PRED_PROBAS, DIR_TEST_RES_EMB, DIR_TEST_LABELS
)

# %%
all_pred_probas = np.concatenate([load_cal_pred_probas, load_test_pred_probas])
all_res_emb = np.concatenate([load_cal_res_emb, load_test_res_emb])
all_labels = np.concatenate([load_cal_labels, load_test_labels])
# all_res_emb = np.concatenate([all_res_emb, np.ones((all_res_emb.shape[0], 1))], axis=1)

# %%
all_res_emb.min(), all_res_emb.max(), all_res_emb.mean()

# %%
ALPHA = .1

# %%
class MapieWrapper():
    def __init__(self):
        self.trained_ = True
        self.classes_ = 1

    def fit(self, X, y=None):
        pass

    def predict_proba(self, X):
        return X

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        return pred_proba >= .5

    def __sklearn_is_fitted__(self):
        return True

# %% [markdown]
# # Cross validation

# %%
class LinearModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, int(input_size / 2))
        # self.fc2 = torch.nn.Linear(int(input_size / 2), 1)
        
        # torch.nn.init.xavier_uniform_(self.fc1.weight) 
        # torch.nn.init.zeros_(self.fc1.bias)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.zeros_(self.fc2.bias)
       

    def forward(self, x):
        x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)
        x = torch.sigmoid(x)
        return x[:, 0]

class CustomLoss(torch.nn.Module):
    def __init__(self, alpha, n):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.n = n

    def forward(self, masks, masks_pred, preds_th, th_n_plus_1):
        integrals = self._I_gpu(masks, masks_pred, preds_th)

        # Ensure the returned loss is differentiable
        return torch.sum(integrals) / (self.n + 1) + (1 - self.alpha) * th_n_plus_1 / (self.n + 1)
    
    def _I_gpu(self, masks, masks_pred, preds_th, steps_trapz=100):
        integrals = []  # Use a list to accumulate the results
        for i in range(len(masks)):
            mask = torch.clone(masks[i]).cuda()
            mask_pred = torch.clone(masks_pred[i]).cuda()
            pred_th = preds_th[i]

            mask = torch.repeat_interleave(
                mask[None, :, :], steps_trapz, dim=0
            )
            mask_pred = torch.repeat_interleave(
                mask_pred[None, :, :], steps_trapz, dim=0
            )
            start = torch.tensor(0., device=mask.device, requires_grad=True)
            end = pred_th  # pred_th should already have requires_grad=True
            steps = steps_trapz

            # Differentiable linspace
            us = torch.lerp(start, end, torch.linspace(0, 1, steps, device=mask.device))
            us = us.view(-1, 1, 1) # Add dimensions for broadcasting

            mask_pred_th = torch.sigmoid(1000 * (mask_pred - us))

            loss = 1 - ((mask_pred_th * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)))
            integral = torch.trapz(loss - self.alpha, us.squeeze())
            integrals.append(integral)

            del mask, mask_pred, start, end, steps, us, mask_pred_th, loss, integral
        
        # Stack to create a tensor with gradients
        return torch.stack(integrals)

# %%
def train_model(
    model, masks, masks_pred, embeddings, lr, weight_decay, n_epochs, batch_size, alpha, x_n_plus_1
):
    masks = torch.tensor(masks.astype(np.float32))
    masks_pred = torch.tensor(masks_pred.astype(np.float32))
    embeddings = torch.tensor(embeddings.astype(np.float32)).to("cuda")
    model = model.to("cuda")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CustomLoss(alpha, batch_size)
    losses = []
    best_model = deepcopy(model)
    best_loss = np.inf
    for epoch in range(n_epochs):
        for i in range(0, len(masks), batch_size):
            masks_batch = masks[i:i + batch_size]
            masks_pred_batch = masks_pred[i:i + batch_size]
            embeddings_batch = embeddings[i:i + batch_size]
            optimizer.zero_grad()
            ths_pred = model(embeddings_batch)
            th_n_plus_1 = model(torch.tensor(x_n_plus_1).to("cuda"))
            loss = criterion(masks_batch, masks_pred_batch, ths_pred, th_n_plus_1)
            losses.append(loss.item())
            # print(f"Epoch {epoch} / {n_epochs} -- Loss: {loss.item()} -- min th: {ths_pred.min().item()} -- max th: {ths_pred.max().item()}", end="\r")
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = deepcopy(model)
    return best_model, losses


# %%
import pickle

# %%
def get_ths(test_res_emb, cal_labels, cal_pred_probas, cal_res_emb, count):
    base_model = LinearModel(RESNET_EMBEDDING_SIZE).share_memory()
    ths_pred_test = []
    n_examples = len(test_res_emb)
    for i, test_emb in enumerate(test_res_emb[:n_examples]):
        if i == 0:
            n_epochs = 100
        else:
            if i == 1:
                print(f"Done 100 epoch training for CV={count}")
            if i % 10 == 0:
                print(f"Done {round(i/n_examples, 2)} data for CV={count}", end="\r")
            n_epochs = 2
        test_emb = test_emb[np.newaxis, :]
        base_model.train()
        model, _ = train_model(
            base_model,
            cal_labels,
            cal_pred_probas,
            cal_res_emb,
            lr=1e-4,
            weight_decay=1e-5,
            n_epochs=n_epochs,
            batch_size=len(cal_labels),
            alpha=ALPHA,
            x_n_plus_1=test_emb.astype(np.float32)
        )
        if i ==0:
            base_model = deepcopy(model)
        model.eval()
        th_n_plus_1 = model(torch.tensor(test_emb.astype(np.float32)).to("cuda"))
        ths_pred_test.append(th_n_plus_1.item())

    return ths_pred_test

# %%
def aa_crc_cv(temp_cal_pred_probas, temp_cal_res_emb, temp_cal_labels, temp_test_pred_probas, temp_test_res_emb, temp_test_labels, count):
    # temp_cal_pred_probas = all_pred_probas[cal_inx]
    # temp_cal_res_emb = all_res_emb[cal_inx]
    # temp_cal_labels = all_labels[cal_inx]

    index_not_empy = np.where(temp_cal_labels.sum(axis=(1, 2)) > 0)[0]
    temp_cal_labels = temp_cal_labels[index_not_empy]
    temp_cal_res_emb = temp_cal_res_emb[index_not_empy]
    temp_cal_pred_probas = temp_cal_pred_probas[index_not_empy]

    # temp_test_pred_probas = all_pred_probas[test_inx]
    # temp_test_res_emb = all_res_emb[test_inx]
    # temp_test_labels = all_labels[test_inx]

    # CRC
    mapie_crc = MapieMultiLabelClassifier(MapieWrapper(), method="crc")
    mapie_crc.lambdas = np.arange(0, 1, 0.001)
    # print("entering CRC")
    for i in range(len(temp_cal_pred_probas)):
        X, y = temp_cal_pred_probas[i], temp_cal_labels[i]
        mapie_crc.partial_fit(X.ravel()[np.newaxis, :], y.ravel()[np.newaxis, :])
    # print("outing CRC")
    _, _ = mapie_crc.predict(temp_test_pred_probas.ravel()[np.newaxis, :], alpha=ALPHA)
    th_crc = mapie_crc.lambdas_star
    # print(f"th_crc = {th_crc}")
    # th_crc = 0
    print(f"Done CRC {count}/100")
    all_ths_crc = th_crc
    ths_res = get_ths(
        temp_test_res_emb,
        temp_cal_labels,
        temp_cal_pred_probas,
        temp_cal_res_emb,
        count
    )
    all_ths_res = ths_res
    print(f"Done AA-CRC {count}/100")
    ths_res = np.array(ths_res)[:, None, None]

    y_pred_test_th_crc = (temp_test_pred_probas >= th_crc).astype(int)
    y_pred_test_th_res = (temp_test_pred_probas >= ths_res).astype(int)

    recalls_crc = np.nanmean((y_pred_test_th_crc * temp_test_labels).sum(axis=(1, 2)) / temp_test_labels.sum(axis=(1, 2)))
    recalls_resnet = np.nanmean((y_pred_test_th_res * temp_test_labels).sum(axis=(1, 2)) / temp_test_labels.sum(axis=(1, 2)))

    precisions_crc = np.nanmean((y_pred_test_th_crc * temp_test_labels).sum(axis=(1, 2)) / y_pred_test_th_crc.sum(axis=(1, 2)))
    precisions_resnet = np.nanmean((y_pred_test_th_res * temp_test_labels).sum(axis=(1, 2)) / y_pred_test_th_res.sum(axis=(1, 2)))

    # with open("thresholds_aacrc.pkl", "wb") as f:
    #     pickle.dump(all_ths_res, f)
    # with open("thresholds_crc.pkl", "wb") as f:
    #     pickle.dump(all_ths_crc, f)
    # with open("recalls_precisions.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "recalls_crc": recalls_crc,
    #             "recalls_resnet": recalls_resnet,
    #             "precisions_crc": precisions_crc,
    #             "precisions_resnet": precisions_resnet
    #         },
    #         f
    #     )
    # print(f"min th_res = {ths_res.min()}, max th = {ths_res.max()}")
    # print(f"recalls: CRC = {recalls_crc[-1]}, MACP = {recalls_resnet[-1]}")
    # print(f"precisions: CRC = {precisions_crc[-1]}, MACP = {precisions_resnet[-1]}")
    
    return recalls_crc, recalls_resnet, precisions_crc, precisions_resnet, all_ths_res, all_ths_crc
        
        

# %%
def get_test_inx(cal_inx, n_data):
    test_inx = np.array([i for i in range(n_data) if i not in cal_inx])
    return test_inx

# %%
from multiprocessing import Pool
import random
def unpack_and_run(args):
    return aa_crc_cv(*args)
with Pool(7) as p:
    cal_inx = [random.sample(range(len(all_pred_probas)), int(len(all_pred_probas) * .5)) for _ in range(100)]
    test_inx = [get_test_inx(cal, len(all_pred_probas)) for cal in cal_inx]
    count = [i for i in range(100)]
    temp_cal_pred_probas = [all_pred_probas[cal_inx[i]] for i in range(len(cal_inx))]
    temp_cal_res_emb = [all_res_emb[cal_inx[i]] for i in range(len(cal_inx))]
    temp_cal_labels = [all_labels[cal_inx[i]] for i in range(len(cal_inx))]
    temp_test_pred_probas = [all_pred_probas[test_inx[i]] for i in range(len(test_inx))]
    temp_test_res_emb = [all_res_emb[test_inx[i]] for i in range(len(test_inx))]
    temp_test_labels = [all_labels[test_inx[i]] for i in range(len(test_inx))]

    print("done")
    data = list(zip(
        temp_cal_pred_probas, temp_cal_res_emb,
        temp_cal_labels, temp_test_pred_probas,
        temp_test_res_emb, temp_test_labels, count
    ))

    result = list(tqdm(p.imap(unpack_and_run, data), total=len(data)))

# %%
recalls_crc = np.array([r[0] for r in result])
recalls_resnet = np.array([r[1] for r in result])
precisions_crc = np.array([r[2] for r in result])
precisions_resnet = np.array([r[3] for r in result])
all_ths_res = np.array([r[4] for r in result])
all_ths_crc = np.array([r[5] for r in result])


# %%
with open("thresholds_aacrc_multiproc.pkl", "wb") as f:
        pickle.dump(all_ths_res, f)
with open("thresholds_crc_multiproc.pkl", "wb") as f:
    pickle.dump(all_ths_crc, f)
with open("recalls_precisions_multiproc.pkl", "wb") as f:
    pickle.dump(
        {
            "recalls_crc": recalls_crc,
            "recalls_resnet": recalls_resnet,
            "precisions_crc": precisions_crc,
            "precisions_resnet": precisions_resnet
        },
        f
    )
