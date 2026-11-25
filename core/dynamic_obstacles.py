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


CORRIDOR_GAPS = (0.3, 0.2, 0.15, 0.1)
CORRIDOR_SPEED = 0.5

WALL_Y, WALL_R = 1.1, 0.4
OBS_Y0, OBS_R  = 0.6, 0.4
X_CENTER       = 2.5


def stop_y(gap):
    """위쪽 통로 폭이 gap이 되는 obs1 정지 y (obs2는 부호 반대)"""
    y = (WALL_Y - WALL_R) - gap - OBS_R
    assert y >= 0.0, f"gap {gap} m는 이 배치에서 만들 수 없음 (장애물이 중심선을 넘어 교차)"
    return y


def make_corridor(gap):
    """장애물 순서: obs1, obs2, wall_top, wall_bottom (회피 방향은 obs1 기준으로 분류)"""
    s = stop_y(gap)
    return {
        "obstacles": [[X_CENTER, +OBS_Y0, OBS_R], [X_CENTER, -OBS_Y0, OBS_R],
                      [X_CENTER, +WALL_Y, WALL_R], [X_CENTER, -WALL_Y, WALL_R]],
        "velocity":  lambda k: np.array([[0.0, -CORRIDOR_SPEED], [0.0, +CORRIDOR_SPEED], [0.0, 0.0], [0.0, 0.0]]),
        "y_limits":  [(s, OBS_Y0), (-OBS_Y0, -s), (WALL_Y, WALL_Y), (-WALL_Y, -WALL_Y)],
    }


def clipped_prediction(base):
    """
    롤아웃 비용의 장애물 예측에도 정지 모델 적용
    core.cost는 선형 예측 p0 + v * t만 지원하므로, clip된 예측 위치 p(t)를
    스텝별 유효 속도 (p(t) - p0) / t, shape (H, 2)로 바꿔 넘긴다
    → core.cost 안에서 p0 + v_eff * t = p(t)
    """
    class ClippedPrediction(base):
        def __init__(self, cfg, y_limits, **kwargs):
            super().__init__(cfg, **kwargs)
            self.y_limits = np.array(y_limits)   # (N, 2)

        def compute_costs(self, trajectories, effective_noise):
            t_steps  = np.arange(1, self.cfg.H + 1) * self.cfg.dt
            observed = self.velocities
            p0       = self.obstacles[:, None, :2]                           # (N, 1, 2)
            pred     = p0 + observed[:, None, :] * t_steps[None, :, None]    # (N, H, 2)
            lo, hi   = self.y_limits.T
            pred[:, :, 1] = np.clip(pred[:, :, 1], lo[:, None], hi[:, None])
            self.velocities = (pred - p0) / t_steps[None, :, None]           # (N, H, 2)
            try:
                return super().compute_costs(trajectories, effective_noise)
            finally:
                self.velocities = observed

    ClippedPrediction.__name__ = f"Clipped{base.__name__}"
    return ClippedPrediction


def gazebo_experiment(scenario, gap):
    if scenario == "B":
        return {"obstacles": EXPERIMENTS["B"]["obstacles"]}
    if scenario == "CORRIDOR":
        if gap not in CORRIDOR_GAPS:
            raise ValueError(f"corridor gap must be one of {CORRIDOR_GAPS}")
        return make_corridor(gap)
    if scenario == "CORRIDOR_PHYSICAL":
        if not np.isfinite(gap) or gap < 0.40:
            raise ValueError("physical corridor gap must be at least 0.40 m")
        # Leave clearance between both moving cylinders at their stop positions
        # and between each moving cylinder and its outer wall at spawn.
        stop = OBS_R + 0.05
        start = 0.80
        wall = stop + OBS_R + WALL_R + gap
        return {
            "obstacles": [[X_CENTER, start, OBS_R], [X_CENTER, -start, OBS_R],
                          [X_CENTER, wall, WALL_R], [X_CENTER, -wall, WALL_R]],
            "velocity": lambda _: np.array([[0.0, -CORRIDOR_SPEED], [0.0, CORRIDOR_SPEED],
                                            [0.0, 0.0], [0.0, 0.0]]),
            "y_limits": [(stop, start), (-start, -stop), (wall, wall), (-wall, -wall)],
        }
    raise ValueError("scenario must be B, CORRIDOR or CORRIDOR_PHYSICAL")
