from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from core.controller import MPPIController
from core.dynamics   import step


def run_once(seed):
    """
    단일 시드로 MPPI 실행 후 경로 반환
    목표 도달 실패 시 None 반환
    """
    _cfg       = cfg
    controller = MPPIController(_cfg)
    # 시드만 덮어쓰기
    controller.rng = np.random.default_rng(seed)

    goal     = np.array(_cfg.goal)
    obstacle = np.array(_cfg.obstacle)
    state    = np.array(_cfg.initial_state, dtype=float)
    states   = [state.copy()]

    for _ in range(_cfg.max_steps):
        action = controller.command(state)
        state  = step(state, action, _cfg.dt)
        states.append(state.copy())

        goal_distance = np.linalg.norm(state[:2] - goal)
        clearance     = np.linalg.norm(
            state[:2] - obstacle[:2]
        ) - obstacle[2]

        if clearance <= 0.0:
            return None   # 충돌
        if goal_distance <= _cfg.goal_tolerance:
            return np.asarray(states)   # 목표 도달

    return None   # 최대 반복 도달


def classify(states, obstacle):
    """
    경로가 장애물을 위쪽/아래쪽 중 어디로 회피했는지 분류
    장애물 x 범위 안에서 y 중앙값으로 판단
    """
    center = obstacle[:2]
    radius = obstacle[2]

    # 장애물 x 범위 안의 상태만 추출
    mask = np.abs(states[:, 0] - center[0]) < radius * 2
    if not np.any(mask):
        return "unknown"

    y_median = np.median(states[mask, 1])
    return "위쪽" if y_median > center[1] else "아래쪽"


def main():
    seeds    = list(range(50))   # 시드 0~49
    goal     = np.array(cfg.goal)
    obstacle = np.array(cfg.obstacle)

    results = {"위쪽": [], "아래쪽": [], "실패": [], "unknown": []}

    print("시드별 실행 중...")
    for seed in seeds:
        states = run_once(seed)
        if states is None:
            results["실패"].append(seed)
            print(f"  seed={seed:2d} → 실패")
        else:
            mode = classify(states, obstacle)
            results[mode].append((seed, states))
            print(f"  seed={seed:2d} → {mode} 회피")

    print(f"\n위쪽:  {len(results['위쪽'])}회")
    print(f"아래쪽: {len(results['아래쪽'])}회")
    print(f"실패:  {len(results['실패'])}회")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 7))

    # 장애물
    ax.add_patch(plt.Circle(
        obstacle[:2], obstacle[2],
        color="crimson", alpha=0.35, label="Obstacle",
    ))
    ax.add_patch(plt.Circle(
        obstacle[:2], obstacle[2] + cfg.margin,
        fill=False, edgecolor="crimson", linestyle=":",
        label="Safety margin",
    ))

    # 목표
    ax.add_patch(plt.Circle(
        goal, cfg.goal_tolerance,
        color="purple", alpha=0.12,
    ))
    ax.scatter(*goal, marker="*", s=150,
               color="purple", zorder=5, label="Goal")

    # 위쪽 회피 경로
    for i, (seed, states) in enumerate(results["위쪽"]):
        ax.plot(
            states[:, 0], states[:, 1],
            color="steelblue", alpha=0.5, linewidth=1.2,
            label="Upper pass" if i == 0 else None,
        )

    # 아래쪽 회피 경로
    for i, (seed, states) in enumerate(results["아래쪽"]):
        ax.plot(
            states[:, 0], states[:, 1],
            color="darkorange", alpha=0.5, linewidth=1.2,
            label="Lower pass" if i == 0 else None,
        )

    ax.scatter(
        cfg.initial_state[0], cfg.initial_state[1],
        color="green", s=80, zorder=5, label="Start",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"MPPI avoidance modes (seed 0~{len(seeds)-1})\n"
        f"Upper: {len(results['위쪽'])}  "
        f"Lower: {len(results['아래쪽'])}  "
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
