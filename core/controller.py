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
        trajectories = rollout(state, self.controls, noise, cfg.dt)
        costs = compute_costs(
            trajectories,
            self.controls,
            noise,
            np.array(cfg.goal),
            np.array(cfg.obstacle),
            cfg,
        )
        updated_controls, weights = update(
            self.controls, noise, costs, cfg,
        )
        ess = 1.0 / np.sum(weights ** 2)
        self.weights_history.append(weights)
        self.ess_history.append(ess)
        self.cost_history.append(np.min(costs))
        action        = updated_controls[0].copy()
        self.controls = np.vstack([
            updated_controls[1:],
            updated_controls[-1:],
        ])
        return action
