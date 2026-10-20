import numpy as np

def step(state, control, dt):
    x, y, theta = state
    v, omega    = control
    return np.array([
        x     + v * np.cos(theta) * dt,
        y     + v * np.sin(theta) * dt,
        theta + omega * dt,
    ])

def step_batch(states, controls, dt):
    x     = states[:, 0]
    y     = states[:, 1]
    theta = states[:, 2]
    v     = controls[:, 0]
    omega = controls[:, 1]
    return np.stack([
        x     + v * np.cos(theta) * dt,
        y     + v * np.sin(theta) * dt,
        theta + omega * dt,
    ], axis=1)
