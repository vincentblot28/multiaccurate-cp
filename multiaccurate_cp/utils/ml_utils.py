import numpy as np
from scipy.optimize import minimize


def get_threshold(pred, label, target_recall):
    def to_optimize(threshold):
        pred_binary = pred > threshold
        recall = np.sum(pred_binary * label) / np.sum(label)
        return np.abs(recall - target_recall)

    res = minimize(to_optimize, 0.5, method="Nelder-Mead")
    threshold = res.x[0]
    return threshold
