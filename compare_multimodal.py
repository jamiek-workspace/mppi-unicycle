from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from core.dynamics  import step
from core.sampling  import sample_noise
from core.rollout   import rollout
from core.cost      import control_cost, terminal_cost
from core.update    import compute_weights


# 장애물 2개 가로 배치
# (main 브랜치의 core.cost는 장애물 1개만 처리하므로
#  비용 계산도 이 파일 안에서 장애물 여러 개용으로 정의)
OBSTACLES = [
    np.array([1.8, 0.0, 0.4]),
    np.array([3.2, 0.0, 0.4]),
]
GOAL = np.array(cfg.goal)
INITIAL_STATE = np.array([1.0, 0.0, 0.0])   # 장애물 바로 앞에서 출발


def compute_costs_multi(trajectories, controls, noise, goal, obstacles):
    """core.cost.compute_costs와 같은 구조, 장애물 여러 개 지원"""
    positions = trajectories[:, 1:, :2]
    r_cost = cfg.q_goal * np.sum((positions - goal) ** 2, axis=2)
    for obs in obstacles:
        clearance = np.linalg.norm(positions - obs[:2], axis=2) - obs[2]
        proximity = np.maximum(
            0.0, (cfg.margin - clearance) / cfg.margin,
        ) ** 2
        r_cost += cfg.q_obstacle * proximity
        r_cost += cfg.q_collision * (clearance <= 0.0)
    c_cost = control_cost(controls, noise, cfg)
    t_cost = terminal_cost(trajectories[:, -1, :], goal, cfg)
    return np.sum((r_cost + c_cost) * cfg.dt, axis=1) + t_cost


def min_clearance(position):
    return min(
        np.linalg.norm(position - obs[:2]) - obs[2]
        for obs in OBSTACLES
    )


def weighted_update(costs, effective_noise, idx):
    """idx 샘플들만으로 가중평균한 노이즈 반환"""
    weights = compute_weights(costs[idx], cfg.lam)
    return np.sum(weights[:, None, None] * effective_noise[idx], axis=0)


def mppi_step(state, controls, noise, multimodal=False):
    """
    MPPI 한 스텝: (적용 입력, 다음 계획, 선택 모드) 반환
    multimodal=True면 샘플을 위/아래 모드로 나눠
    모드별 가중평균 후 최소 비용 모드를 선택
    """
    traj, effective_noise = rollout(
        state, controls, noise, cfg.dt,
        v_min=cfg.v_min, v_max=cfg.v_max,
        omega_max=cfg.omega_max,
    )
    costs = compute_costs_multi(
        traj, controls, effective_noise, GOAL, OBSTACLES,
    )

    mode = None
    if not multimodal:
        weighted_noise = weighted_update(
            costs, effective_noise, np.arange(len(costs)),
        )
    else:
        # 1. 샘플을 위/아래 모드로 분리
        #    첫 번째 장애물 x에 가장 가까운 시간 스텝의 y 부호로 구분
        x_obs = OBSTACLES[0][0]
        t_idx = np.argmin(np.abs(traj[:, :, 0] - x_obs), axis=1)   # (K,)
        y_at  = traj[np.arange(len(costs)), t_idx, 1]
        modes = {
            "upper": np.flatnonzero(y_at > 0),
            "lower": np.flatnonzero(y_at <= 0),
        }

        # 2. 각 모드 안에서 가중평균, 모드 비용 = 모드 내 최소 비용
        mode_cost = {
            name: np.min(costs[idx]) if len(idx) > 0 else np.inf
            for name, idx in modes.items()
        }

        # 3. 더 좋은 모드 선택
        mode = min(mode_cost, key=mode_cost.get)
        weighted_noise = weighted_update(
            costs, effective_noise, modes[mode],
        )

    updated = controls + weighted_noise

    action    = updated[0].copy()
    action[0] = np.clip(action[0], cfg.v_min, cfg.v_max)
    action[1] = np.clip(action[1], -cfg.omega_max, cfg.omega_max)

    next_controls = np.vstack([updated[1:], updated[-1:]])
    return action, next_controls, mode


def run_comparison(seed):
    """
    같은 시드로 두 방법을 실행해서 경로와 종료 상태 반환

    반환:
        {"mean": (states, status, mode_switches),
         "multi": (states, status, mode_switches)}
        status: "success" / "collision" / "timeout"
    """
    rng = np.random.default_rng(seed)

    init_controls = np.zeros((cfg.H, 2))
    init_controls[:, 0] = 1.0

    methods = {
        "mean":  {"multimodal": False},
        "multi": {"multimodal": True},   # 위/아래 모드 선택
    }
    for m in methods.values():
        m["state"]    = INITIAL_STATE.copy()
        m["controls"] = init_controls.copy()
        m["states"]   = [m["state"].copy()]
        m["status"]   = "timeout"
        m["done"]     = False
        m["mode"]     = None
        m["switches"] = 0   # 선택 모드가 바뀐 횟수

    for _ in range(cfg.max_steps):
        # 같은 노이즈 사용 (공정한 비교)
        noise = sample_noise(cfg.K, cfg.H, cfg.noise_std, rng)

        for m in methods.values():
            if m["done"]:
                continue
            action, m["controls"], mode = mppi_step(
                m["state"], m["controls"], noise, m["multimodal"],
            )
            if m["mode"] is not None and mode != m["mode"]:
                m["switches"] += 1
            m["mode"] = mode
            m["state"] = step(m["state"], action, cfg.dt)
            m["states"].append(m["state"].copy())

            if min_clearance(m["state"][:2]) <= 0.0:
                m["status"], m["done"] = "collision", True
            elif (np.linalg.norm(m["state"][:2] - GOAL)
                  <= cfg.goal_tolerance):
                m["status"], m["done"] = "success", True

        # 두 방법 모두 종료하면 끝
        if all(m["done"] for m in methods.values()):
            break

    return {
        name: (np.asarray(m["states"]), m["status"], m["switches"])
        for name, m in methods.items()
    }


def main():
    seeds = list(range(50))

    results = {"mean": [], "multi": []}

    print("시드별 실행 중...")
    for seed in seeds:
        out = run_comparison(seed)
        for name in results:
            results[name].append(out[name])
        print(
            f"  seed={seed:2d}  "
            f"mean: {out['mean'][1]:9s} ({len(out['mean'][0]) - 1:3d} steps)  "
            f"multi: {out['multi'][1]:9s} ({len(out['multi'][0]) - 1:3d} steps, "
            f"mode switches {out['multi'][2]})"
        )

    print()
    for name, label in [("mean", "전체 평균"), ("multi", "Multimodal")]:
        statuses = [s for _, s, _ in results[name]]
        steps = [len(st) - 1 for st, s, _ in results[name] if s == "success"]
        upper = sum(
            1 for st, s, _ in results[name]
            if s == "success" and np.median(
                st[np.abs(st[:, 0] - 2.5) < 1.0, 1]
            ) > 0
        )
        clear = [
            min(min_clearance(p) for p in st[:, :2])
            for st, _, _ in results[name]
        ]
        print(
            f"{label}: 최소 여유 거리 min {np.min(clear):.3f} / "
            f"평균 {np.mean(clear):.3f} m, "
            f"margin({cfg.margin}) 미만 {sum(c < cfg.margin for c in clear)}회"
        )
        print(
            f"{label}: 성공 {statuses.count('success')}, "
            f"충돌 {statuses.count('collision')}, "
            f"시간 초과 {statuses.count('timeout')} | "
            f"성공 중 위쪽 {upper} / 아래쪽 "
            f"{statuses.count('success') - upper} | "
            f"평균 스텝 {np.mean(steps) if steps else float('nan'):.1f}"
        )

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))

    titles = ["Standard MPPI (all samples)",
              "Multimodal MPPI (mode selection)"]
    colors = ["steelblue", "darkorange"]

    for ax, title, name, color in zip(
        axes, titles, ["mean", "multi"], colors
    ):
        # 장애물
        for obs in OBSTACLES:
            ax.add_patch(plt.Circle(
                obs[:2], obs[2],
                color="crimson", alpha=0.35,
            ))
            ax.add_patch(plt.Circle(
                obs[:2], obs[2] + cfg.margin,
                fill=False, edgecolor="crimson",
                linestyle=":",
            ))

        # 목표
        ax.add_patch(plt.Circle(
            GOAL, cfg.goal_tolerance,
            color="purple", alpha=0.12,
        ))
        ax.scatter(
            *GOAL, marker="*", s=150,
            color="purple", zorder=5, label="Goal",
        )

        # 경로 (실패는 회색 점선)
        first_ok, first_fail = True, True
        for states, status, _ in results[name]:
            if status == "success":
                ax.plot(
                    states[:, 0], states[:, 1],
                    color=color, alpha=0.4, linewidth=1.2,
                    label="Success" if first_ok else None,
                )
                first_ok = False
            else:
                ax.plot(
                    states[:, 0], states[:, 1],
                    color="gray", alpha=0.4, linewidth=0.8,
                    linestyle="--",
                    label="Failed" if first_fail else None,
                )
                first_fail = False

        # 시작점
        ax.scatter(
            INITIAL_STATE[0], INITIAL_STATE[1],
            color="green", s=80, zorder=5, label="Start",
        )

        n_ok = sum(1 for _, s, _ in results[name] if s == "success")
        ax.set_title(f"{title}\nsuccess {n_ok}/{len(seeds)}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, 6.0)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Standard MPPI vs Multimodal MPPI\n"
        f"K={cfg.K}, lam={cfg.lam}, start=({INITIAL_STATE[0]}, {INITIAL_STATE[1]}), modes=upper/lower, "
        f"seeds 0~{len(seeds)-1}",
        fontsize=12,
    )
    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "compare_multimodal.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그래프 저장: {output_path}")


if __name__ == "__main__":
    main()
