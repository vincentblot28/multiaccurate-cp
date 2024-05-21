import numpy as np
from scipy.optimize import minimize


def get_threshold2(pred, label, target_recall):
    def to_optimize(threshold):
        pred_binary = pred > threshold
        recall = np.sum(pred_binary * label) / np.sum(label)
        return np.abs(recall - target_recall)

    res = minimize(to_optimize, 0.5, method="Nelder-Mead")
    threshold = res.x[0]
    return threshold


def get_threshold(pred, label, target_recall):
    
    alphas = np.linspace(0, 1, 1000)
    recalls = []
    for alpha in alphas:
        pred_binary = pred > alpha
        recall = np.sum(pred_binary * label) / np.sum(label)
        recalls.append(recall)

    recalls = np.array(recalls) >= target_recall
    recalls = recalls[::-1]
    # return small alpha that satisfies the condition
    return alphas[np.argmax(recalls)]
