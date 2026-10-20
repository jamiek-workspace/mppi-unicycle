import numpy as np
from core.sampling import sample_noise
from core.rollout  import rollout
from core.cost     import compute_costs
from core.update   import update

class MPPIController:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.rng      = np.random.default_rng(cfg.seed)
        self.controls = np.zeros((cfg.H, 2))
        self.controls[:, 0] = 1.0
        self.weights_history = []
        self.ess_history     = []
        self.cost_history    = []

    def command(self, state):
        cfg   = self.cfg
        noise = sample_noise(cfg.K, cfg.H, cfg.noise_std, self.rng)
        trajectories, effective_noise = rollout(
            state, self.controls, noise, cfg.dt,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            omega_max=cfg.omega_max,
        )
        costs = compute_costs(
            trajectories,
            self.controls,
            effective_noise,        # clip 전 noise → effective_noise
            np.array(cfg.goal),
            [np.array(obs) for obs in cfg.obstacles],
            cfg,
        )
        updated_controls, weights = update(
            self.controls,
            effective_noise,        # clip 전 noise → effective_noise
            costs,
            cfg,
        )
        ess = 1.0 / np.sum(weights ** 2)
        self.weights_history.append(weights)
        self.ess_history.append(ess)
        self.cost_history.append(np.min(costs))
        action        = updated_controls[0].copy()
        action[0]     = np.clip(action[0], cfg.v_min, cfg.v_max)
        action[1]     = np.clip(action[1], -cfg.omega_max, cfg.omega_max)
        self.controls = np.vstack([
            updated_controls[1:],
            updated_controls[-1:],
        ])
        return action
