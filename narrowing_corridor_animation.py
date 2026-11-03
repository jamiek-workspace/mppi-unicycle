"""
좁아지는 통로 실험 애니메이션: 단일 MPPI vs Multimodal MPPI를 나란히 재생

같은 gap, 같은 시드로 두 방법을 실행하고, 장애물이 닫히는 과정과
로봇 궤적을 실시간 속도(시뮬레이션 2스텝 = 1프레임, 25 fps)로 GIF에 저장한다.

실행: python narrowing_corridor_animation.py [gap] [seed]   (기본 gap=0.1, seed=0)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import config
from multimodal_dynamic_experiment import CROSS, INK, METHOD_COLORS, OBS_GRAY, PATH_COLORS, style_axis
from narrowing_corridor_experiment import (
    LABELS, METHODS, V_Y, WALL_GRAY, make_env, run_once,
)

STEP_PER_FRAME = 2                                      # 2스텝 × dt 0.02 s = 0.04 s
FPS            = round(1 / (STEP_PER_FRAME * config.dt))  # = 25 → 실시간
HOLD_FRAMES    = FPS                                    # 끝난 뒤 1초 정지 화면
HEADING_LEN    = 0.3


def setup_panel(ax, env, method, result):
    for i, (x, y, r) in enumerate(env["obstacles"]):
        if i >= 2:   # 정적 벽
            ax.add_patch(plt.Circle((x, y), r, color=WALL_GRAY, alpha=0.55, lw=0))
        else:        # 동적 장애물의 정지 위치 (점선)
            y_end = env["y_limits"][i][0] if i == 0 else env["y_limits"][i][1]
            ax.add_patch(plt.Circle((x, y_end), r, fill=False, edgecolor=OBS_GRAY,
                                    linestyle="--", lw=1))
    goal = np.array(config.goal)
    ax.add_patch(plt.Circle(goal, config.goal_tolerance, color=INK, alpha=0.08, lw=0))
    ax.scatter(*goal, marker="*", s=120, color=INK, zorder=5)
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    style_axis(ax)

    outcome = result["outcome"]
    ax.set_title(f"{LABELS[method]}  →  {outcome} at step {result['steps']}",
                 fontsize=10, color="#0b0b0b")

    artists = {
        "obstacles": [ax.add_patch(plt.Circle((0, 0), r, color=OBS_GRAY, alpha=0.45, lw=0))
                      for (_, _, r) in env["obstacles"][:2]],
        "trail":     ax.plot([], [], color=METHOD_COLORS[method], lw=1.5)[0],
        "heading":   ax.plot([], [], color="#0b0b0b", lw=2, solid_capstyle="round")[0],
        "robot":     ax.scatter([], [], s=50, color="#0b0b0b", zorder=7),
        "collision": ax.scatter(np.empty(0), np.empty(0), marker="x",
                                s=90, color=CROSS, linewidths=2.5, zorder=8),
        "timeout":   ax.scatter(np.empty(0), np.empty(0), marker="o", s=90,
                                facecolors="none", edgecolors=INK, linewidths=1.5, zorder=8),
        "text":      ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                             fontsize=9, color=INK, family="monospace"),
    }
    return artists


def update_panel(artists, method, result, step):
    states, obstacles = result["states"], result["obstacles"]
    last = len(states) - 1
    k    = min(step, last)
    x, y, theta = states[k]

    for patch, (ox, oy, _) in zip(artists["obstacles"], obstacles[k][:2]):
        patch.center = (ox, oy)
    artists["trail"].set_data(states[:k + 1, 0], states[:k + 1, 1])
    artists["heading"].set_data([x, x + HEADING_LEN * np.cos(theta)],
                                [y, y + HEADING_LEN * np.sin(theta)])
    artists["robot"].set_offsets([[x, y]])

    # 모드는 command 호출(스텝 k → k+1)마다 기록되므로 k번째 계획의 모드를 표시
    status = "running"
    if method != "single":
        mode = result["modes"][min(k, len(result["modes"]) - 1)]
        artists["robot"].set_color(PATH_COLORS[mode])
        status = f"mode: {mode}"
    if k == last:
        status = result["outcome"]
        if result["outcome"] in ("collision", "timeout"):
            artists[result["outcome"]].set_offsets([[x, y]])
    artists["text"].set_text(f"t = {k * config.dt:4.2f} s  step {k:3d}\n{status}")


def main():
    gap  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    env  = make_env(gap)

    results = {m: run_once(m, seed, gap) for m in METHODS}
    for m, r in results.items():
        print(f"{LABELS[m]:16s} {r['outcome']:9s} steps {r['steps']:3d}  "
              f"min clearance {r['min_clear']:+.3f} m")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = {m: setup_panel(ax, env, m, results[m]) for ax, m in zip(axes, METHODS)}
    fig.suptitle(f"Narrowing corridor, gap = {gap} m, v_y = {V_Y} m/s, seed {seed}  "
                 f"(real-time playback)", fontsize=11, color="#0b0b0b")
    fig.tight_layout()

    n_steps  = max(r["steps"] for r in results.values())
    frames   = list(range(0, n_steps + 1, STEP_PER_FRAME))
    if frames[-1] != n_steps:
        frames.append(n_steps)
    frames  += [n_steps] * HOLD_FRAMES

    def update(step):
        for m in METHODS:
            update_panel(panels[m], m, results[m], step)
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    output_path = (Path(__file__).resolve().parent
                   / f"narrowing_corridor_gap{gap}_seed{seed}.gif")
    anim.save(output_path, writer=PillowWriter(fps=FPS), dpi=80)
    print(f"GIF 저장: {output_path}  ({len(frames)} frames, {FPS} fps)")


if __name__ == "__main__":
    main()
