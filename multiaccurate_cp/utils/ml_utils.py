import numpy as np


def get_threshold(pred, label):
    pred = pred[:, :, np.newaxis]
    pred = np.repeat(pred, 100, axis=2)
    label = label[:, :, np.newaxis] / 255
    label = np.repeat(label, 100, axis=2)

    pred_th = pred >= np.linspace(0, 1, 100)
    recalls = np.round((pred_th * label).sum(axis=(0, 1)) / label.sum(axis=(0, 1)), 2)
    unique_recalls, index_unique = np.unique(recalls, return_index=True)
    unique_ths = np.linspace(0, 1, 100)[index_unique]
    return unique_recalls, unique_ths
