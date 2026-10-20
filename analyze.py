from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from core.controller import MPPIController
from core.dynamics   import step


def run_once(seed):
    """
    단일 시드로 MPPI 실행 후 (경로, 상태) 반환
    상태: 목표 도달 시 "success", 충돌/최대 반복 시 "fail"
    """
    _cfg       = cfg
    controller = MPPIController(_cfg)
    # 시드만 덮어쓰기
    controller.rng = np.random.default_rng(seed)

    goal     = np.array(_cfg.goal)
    obstacles = [np.array(obs) for obs in _cfg.obstacles]
    state    = np.array(_cfg.initial_state, dtype=float)
    states   = [state.copy()]

    for _ in range(_cfg.max_steps):
        action = controller.command(state)
        state  = step(state, action, _cfg.dt)
        states.append(state.copy())

        goal_distance = np.linalg.norm(state[:2] - goal)
        clearance     = min(
            np.linalg.norm(state[:2] - obs[:2]) - obs[2]
            for obs in obstacles
        )

        if clearance <= 0.0:
            return np.asarray(states), "fail"      # 충돌
        if goal_distance <= _cfg.goal_tolerance:
            return np.asarray(states), "success"   # 목표 도달

    return np.asarray(states), "fail"   # 최대 반복 도달


def classify(states, obstacles):
    """
    경로의 y 중앙값으로 통로 분류
    장애물 2개 기준 통로 4개:
      위쪽    : y > upper_center + radius
      사이 위  : 0 < y < upper_center - radius
      사이 아래: lower_center + radius < y < 0
      아래쪽   : y < lower_center - radius
    """
    upper = obstacles[0]   # 위쪽 장애물
    lower = obstacles[1]   # 아래쪽 장애물

    # 장애물 x 범위 안의 상태만 추출
    x_center = upper[0]
    mask = np.abs(states[:, 0] - x_center) < upper[2] * 2
    if not np.any(mask):
        return "unknown"

    y_median = np.median(states[mask, 1])

    if y_median > upper[1]:
        return "위쪽"
    elif y_median > 0:
        return "사이 위"
    elif y_median > lower[1]:
        return "사이 아래"
    else:
        return "아래쪽"


def main():
    seeds    = list(range(20))   # 시드 0~19
    goal     = np.array(cfg.goal)
    obstacles = [np.array(obs) for obs in cfg.obstacles]

    results = {
        "위쪽": [], "사이 위": [],
        "사이 아래": [], "아래쪽": [], "실패": [], "unknown": [],
    }

    print("시드별 실행 중...")
    for seed in seeds:
        states, status = run_once(seed)
        if status == "fail":
            results["실패"].append((seed, states))
            print(f"  seed={seed:2d} → 실패")
        else:
            mode = classify(states, obstacles)
            results[mode].append((seed, states))
            print(f"  seed={seed:2d} → {mode} 회피")

    print()
    for mode in ["위쪽", "사이 위", "사이 아래", "아래쪽", "실패"]:
        print(f"{mode}: {len(results[mode])}회")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 7))

    # 장애물
    for i, obs in enumerate(obstacles):
        ax.add_patch(plt.Circle(
            obs[:2], obs[2],
            color="crimson", alpha=0.35,
            label="Obstacle" if i == 0 else None,
        ))
        ax.add_patch(plt.Circle(
            obs[:2], obs[2] + cfg.margin,
            fill=False, edgecolor="crimson", linestyle=":",
            label="Safety margin" if i == 0 else None,
        ))

    # 목표
    ax.add_patch(plt.Circle(
        goal, cfg.goal_tolerance,
        color="purple", alpha=0.12,
    ))
    ax.scatter(*goal, marker="*", s=150,
               color="purple", zorder=5, label="Goal")

    # 경로 색상 4가지 (그래프 라벨은 한글 폰트가 없어 영어로 표시)
    colors = {
        "위쪽":    ("steelblue",  "Above upper"),
        "사이 위":  ("green",      "Gap, upper half"),
        "사이 아래": ("darkorange", "Gap, lower half"),
        "아래쪽":   ("purple",     "Below lower"),
    }

    for mode, (color, name) in colors.items():
        for i, (seed, states) in enumerate(results[mode]):
            ax.plot(
                states[:, 0], states[:, 1],
                color=color, alpha=0.5, linewidth=1.2,
                label=f"{name} ({len(results[mode])})" if i == 0 else None,
            )

    # 실패 경로
    for i, (seed, states) in enumerate(results["실패"]):
        ax.plot(
            states[:, 0], states[:, 1],
            color="gray", alpha=0.3, linewidth=0.8,
            linestyle="--",
            label=f"Failed ({len(results['실패'])})" if i == 0 else None,
        )

    ax.scatter(
        cfg.initial_state[0], cfg.initial_state[1],
        color="green", s=80, zorder=5, label="Start",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"MPPI avoidance modes (seed 0~{len(seeds)-1})\n"
        f"Above: {len(results['위쪽'])}  "
        f"Gap+: {len(results['사이 위'])}  "
        f"Gap-: {len(results['사이 아래'])}  "
        f"Below: {len(results['아래쪽'])}  "
        f"Failed: {len(results['실패'])}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    output_path = Path(__file__).resolve().parent / "analyze_modes.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그래프 저장: {output_path}")


if __name__ == "__main__":
    main()
