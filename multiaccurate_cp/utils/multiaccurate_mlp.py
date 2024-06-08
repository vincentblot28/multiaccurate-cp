from multiprocessing import Pool

import numpy as np
from scipy import integrate


INF_BORN_INT = 100


def _I_prime(y, y_pred, u, alpha, n):
    if y_pred.ndim == 2:
        y_pred_th = (y_pred[:, :, np.newaxis] > u).astype(int)
        loss = 1 - (y_pred_th * y[:, :, np.newaxis]).sum(axis=(0, 1)) / y.sum()
    else:
        y_pred_th = (y_pred > u[:, np.newaxis, np.newaxis]).astype(int)
        loss = 1 - (y_pred_th * y).sum(axis=(1, 2)) / y.sum(axis=(1, 2))
    return loss - (alpha - 1 / n)


def _I_prime_list(y, y_pred, u, alpha, n):
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim == 2:
            y_pred_th = (y_pred[:, :, np.newaxis] > u).astype(int)
            loss = 1 - (y_pred_th * y[:, :, np.newaxis]).sum(axis=(0, 1)) / y.sum()
        else:
            y_pred_th = (y_pred > u[:, np.newaxis, np.newaxis]).astype(int)
            loss = 1 - (y_pred_th * y).sum(axis=(1, 2)) / y.sum(axis=(1, 2))
    elif isinstance(y_pred, list):
        y_pred_th = [y_pred[i] >= u[i] for i in range(len(y_pred))]
        loss = 1 - np.array([np.sum(y_pred_th[i] * y[i]) / np.sum(y[i]) for i in range(len(y))])
    else:
        raise ValueError("y_pred must be a numpy array or a list of numpy arrays")

    return loss - (alpha - 1 / n)


def _integrate_i(y_i, y_pred_i, us_i, alpha, n):
    if us_i <= 0:
        return - (INF_BORN_INT + us_i) * (alpha - 1 / n)  # - us_i * (alpha - 1 / n)
    return integrate.fixed_quad(
            lambda theta_prime: _I_prime_list(y_i, y_pred_i, theta_prime, alpha, n),
            0, us_i, n=100
        )[0] - (INF_BORN_INT * (alpha - 1 / n))


def _I_vec_multi_proc2(y, y_pred, pred_th, alpha, n):
    us = np.maximum(pred_th, - INF_BORN_INT)
    if us.max() <= 0:
        return - (INF_BORN_INT + us) * (alpha - 1 / n)
    else:
        with Pool(20) as p:
            data = [(y[i], y_pred[i], us[i], alpha, n) for i in range(len(y))]
            integrals = p.starmap(_integrate_i, data)
        return integrals


def _I_vec_multi_proc(y, y_pred, pred_th, alpha, n):
    us = np.maximum(pred_th, -INF_BORN_INT)
    if us.max() <= 0:
        return np.zeros(len(y))
    else:
        integrals = []
        for i in range(len(y)):
            integrals.append(_integrate_i(y[i], y_pred[i], us[i], alpha, n))
        return np.array(integrals)


def J(y, y_pred, pred_th, alpha, n):
    integral = _I_vec_multi_proc(y, y_pred, pred_th, alpha, n)
    return np.sum(integral) / (n + 1) + (1 - alpha) / (n + 1)
