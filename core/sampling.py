import numpy as np

def sample_noise(K, H, noise_std, rng):
    return rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(K, H, 2),
    )
