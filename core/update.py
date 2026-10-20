import numpy as np

def compute_weights(costs, lam):
    rho     = np.min(costs)
    weights = np.exp(-(costs - rho) / lam)
    return weights / np.sum(weights)

def update(controls, noise, costs, cfg):
    weights        = compute_weights(costs, cfg.lam)
    weighted_noise = np.sum(
        weights[:, None, None] * noise, axis=0,
    )
    return controls + weighted_noise, weights
