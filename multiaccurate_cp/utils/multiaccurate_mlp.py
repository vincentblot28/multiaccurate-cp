from multiprocessing import Pool

import numpy as np
import torch


INF_BORN_INT = 100


def _I_prime(y, y_pred, u, alpha, n):
    if y_pred.ndim == 2:
        y_pred_th = (y_pred[:, :, None] > u)
        loss = 1 - (y_pred_th * y[:, :, None]).sum(axis=(0, 1)) / y.sum()
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
    to_integrate = torch.zeros(101, requires_grad=False)
    for u in range(101):
        to_integrate[u] += _I_prime(y_i, y_pred_i, u * us_i / 100, alpha, n)[0]
    return torch.trapz(to_integrate)


def _I_vec_multi_proc2(y, y_pred, us, alpha, n):
    if us.max() <= 0:
        return - (INF_BORN_INT + us) * (alpha - 1 / n)
    else:
        with Pool(20) as p:
            data = [(y[i], y_pred[i], us[i], alpha, n) for i in range(len(y))]
            integrals = p.starmap(_integrate_i, data)
        return integrals


def _I_vec_multi_proc(y, y_pred, us, alpha, n):
    us = torch.maximum(us, - torch.tensor(INF_BORN_INT))
    if us.max() <= 0:
        return - (INF_BORN_INT + us) * (alpha - 1 / n)
    integrals = torch.zeros(len(y), requires_grad=False)
    for i in range(len(y)):
        integrals[i] += _integrate_i(y[i], y_pred[i], us[i], alpha, n)
    return integrals


def J(y, y_pred, u, alpha):
    n = len(y)
    integral = _I_vec_multi_proc(y, y_pred, u, alpha, n)
    return integral


def J_prime(theta, y, y_pred, emb, alpha, n, reg=None, lambda_rg=None):
    lambda_prime = emb
    if reg == "ridge":
        return np.mean(
            lambda_prime * np.array(
                _I_prime_list(y, y_pred, np.maximum(0, emb @ theta), alpha, n)
            ).reshape(-1, 1) + lambda_rg * 2 * theta,
            axis=0
        )
    elif reg == "lasso":
        return np.mean(
            lambda_prime * np.array(
                _I_prime_list(y, y_pred, np.maximum(0, emb @ theta), alpha, n)
            ).reshape(-1, 1) + lambda_rg * np.sign(theta),
            axis=0
        )
    elif reg is None:
        return np.mean(
            lambda_prime * np.array(
                _I_prime_list(y, y_pred, np.maximum(0, emb @ theta), alpha, n)
            ).reshape(-1, 1),
            axis=0
        )
    else:
        raise ValueError("reg must be 'ridge', 'lasso' or None")
