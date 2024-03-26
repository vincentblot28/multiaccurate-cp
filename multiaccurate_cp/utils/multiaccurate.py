from multiprocessing import Pool

import numpy as np
from scipy import integrate


def _I_prime(y, y_pred, u, alpha, n):
    if y_pred.ndim == 2:
        y_pred_th = (y_pred[:, :, np.newaxis] > u).astype(int)
        loss = 1 - (y_pred_th * y[:, :, np.newaxis]).sum(axis=(0, 1)) / y.sum()
    else:
        y_pred_th = (y_pred > u[:, np.newaxis, np.newaxis]).astype(int)
        loss = 1 - (y_pred_th * y).sum(axis=(1, 2)) / y.sum(axis=(1, 2))
    return loss - (alpha - 1 / n)


def _integrate_i(y_i, y_pred_i, us_i, alpha, n):
    return integrate.fixed_quad(
            lambda theta_prime: _I_prime(y_i, y_pred_i, theta_prime, alpha, n),
            0, us_i, n=100
        )[0]


def _I_vec_multi_proc(y, y_pred, emb, theta, alpha, n):
    us = emb @ theta

    with Pool(20) as p:
        data = [(y[i], y_pred[i], us[i], alpha, n) for i in range(len(y))]
        integrals = p.starmap(_integrate_i, data)
    return integrals


def J(theta, y, y_pred, emb, alpha, n):
    integral = _I_vec_multi_proc(y, y_pred, emb, theta, alpha, n)
    return np.mean(integral)


def J_prime(theta, y, y_pred, emb, alpha, n):
    return np.mean(emb * np.array(_I_prime(y, y_pred, emb @ theta, alpha, n)).reshape(-1, 1), axis=0)
