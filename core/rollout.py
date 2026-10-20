import numpy as np
from core.dynamics import step_batch

def rollout(state, controls, noise, dt):
    K, H, _ = noise.shape
    trajectories = np.zeros((K, H + 1, 3))
    trajectories[:, 0, :] = state
    perturbed_controls = controls[None, :, :] + noise
    for i in range(H):
        trajectories[:, i + 1, :] = step_batch(
            trajectories[:, i, :],
            perturbed_controls[:, i, :],
            dt,
        )
    return trajectories
