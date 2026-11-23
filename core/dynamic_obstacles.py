"""Shared crossing scenarios and obstacle costs for Single MPPI."""
from types import SimpleNamespace

import numpy as np

from core.controller import MPPIController
from core.cost import running_cost


class MultiObstacleController(MPPIController):
    """
    여러 장애물에 대해 Single MPPI의 장애물 비용을 합산한다.
    core/cost.py는 단일 장애물 인터페이스를 유지하므로,
    첫 번째 장애물은 기존 compute_costs로, 나머지는 장애물 항만 더한다.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.obstacles  = None   # (N, 3)
        self.velocities = None   # (N, 2)

    def compute_costs(self, trajectories, effective_noise):
        cfg     = self.cfg
        t_steps = np.arange(1, cfg.H + 1) * cfg.dt
        cfg.obstacle     = self.obstacles[0]
        cfg.obs_velocity = self.velocities[0]
        costs = super().compute_costs(trajectories, effective_noise)

        # 나머지 장애물: q_goal=0으로 목표 비용 중복 없이 장애물 비용만 계산
        positions = trajectories[:, 1:, :2]
        goal      = np.array(cfg.goal)
        for obs, vel in zip(self.obstacles[1:], self.velocities[1:]):
            sub = SimpleNamespace(**vars(cfg))
            sub.q_goal = 0.0
            sub.obs_velocity = vel
            r_cost = running_cost(positions, goal, obs, sub, t_steps)
            costs  = costs + np.sum(r_cost * cfg.dt, axis=1)
        return costs


# ── 실험 정의
# velocity(k): 스텝 k에서 장애물의 실제 속도 (N, 2)
EXPERIMENTS = {
    "A": {
        "title":     "A: single obstacle crossing",
        "obstacles": [[2.5, 1.2, 0.5]],
        "velocity":  lambda k: np.array([[0.0, -0.4]]),
    },
    "B": {
        "title":     "B: two obstacles, opposite directions",
        "obstacles": [[2.0, 0.5, 0.4], [3.5, -0.5, 0.4]],
        "velocity":  lambda k: np.array([[0.0, -0.5], [0.0, +0.5]]),
    },
    "C": {
        "title":     "C: velocity reverses after 50 steps",
        "obstacles": [[2.5, 0.6, 0.5]],
        "velocity":  lambda k: np.array([[0.0, -0.4 if k < 50 else +0.4]]),
    },
}
