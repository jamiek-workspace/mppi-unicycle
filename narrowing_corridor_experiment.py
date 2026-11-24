"""
좁아지는 통로 실험: 위아래 정적 벽 + 중앙을 막는 동적 장애물

시작 [0, 0, 0] → 목표 [5, 0]
정적 벽  : wall_top [2.5, +1.1, 0.4], wall_bottom [2.5, -1.1, 0.4]
동적 장애물: obs1 [2.5, +0.6, 0.4], velocity [0, -V_Y], y = stop_y(gap)에서 정지
            obs2 [2.5, -0.6, 0.4], velocity [0, +V_Y], y = -stop_y(gap)에서 정지
→ 중앙은 막히고, 벽과 장애물 사이의 위/아래 통로가 각각 gap 폭으로 남음
  위쪽 통로 폭 = (wall_top_y - wall_r) - (obs1_stop_y + obs_r) = gap

난이도 스윕: gap = 0.3, 0.2, 0.15, 0.1 m, 각 gap에서 시드 0~49
비교: 단일 MPPI vs Multimodal MPPI (양쪽 업데이트 + bias decay), obs_dynamic=True
      장애물 4개 비용을 각각 계산해서 합산 (MultiObstacleController 재사용)
      롤아웃 예측도 같은 정지 모델 사용: predicted_y = clip(y + v_y * t, lo, hi)

실행: python narrowing_corridor_experiment.py [시드수]
"""
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config
from core.dynamic_obstacles import (
    CORRIDOR_GAPS, CORRIDOR_SPEED as V_Y, MultiObstacleController,
    clipped_prediction, make_corridor as make_env, stop_y,
)
from multimodal_dynamic_experiment import (
    BIAS_DECAY_DIST, CROSS, FAIL, GRID, INK, K_UPPER, METHOD_COLORS, OBS_GRAY,
    PATH_COLORS, MultimodalController, run_env, style_axis, summarize,
)

# 위/아래 통로 폭 (작을수록 어려움)
# 최대 0.3 m: 더 넓히려면 obs1/obs2가 중심선을 넘어 교차해야 하는데,
# 그러면 반대쪽 장애물이 통로를 막아 실제 폭이 오히려 줄어든다 (gap 0.5 → 실제 0.1)
GAPS     = CORRIDOR_GAPS
PATH_GAPS = (0.2, 0.1)            # 경로 패널을 그릴 gap
METHODS  = ("single", "mm_both")
LABELS   = {"single": "Single MPPI", "mm_both": "Multimodal MPPI"}

ClippedSingle     = clipped_prediction(MultiObstacleController)
ClippedMultimodal = clipped_prediction(MultimodalController)


def make_controller(method, cfg, y_limits):
    if method == "single":
        return ClippedSingle(cfg, y_limits)
    return ClippedMultimodal(cfg, y_limits, update_all=True, bias_decay=True)


def run_once(method, seed, gap, dynamic=True):
    env = make_env(gap)
    return run_env(method, seed, env, dynamic,
                   make=partial(make_controller, y_limits=env["y_limits"]))


# ── 시각화
WALL_GRAY = "#5f5e5a"


def draw_environment(ax, gap):
    env = make_env(gap)
    for i, (x, y, r) in enumerate(env["obstacles"]):
        if i >= 2:   # 정적 벽
            ax.add_patch(plt.Circle((x, y), r, color=WALL_GRAY, alpha=0.55, lw=0))
            continue
        y_end = env["y_limits"][i][0] if i == 0 else env["y_limits"][i][1]
        ax.add_patch(plt.Circle((x, y), r, color=OBS_GRAY, alpha=0.2, lw=0))
        ax.add_patch(plt.Circle((x, y_end), r, fill=False, edgecolor=OBS_GRAY,
                                linestyle="--", lw=1))
        ax.annotate("", xy=(x, y_end), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=OBS_GRAY, lw=1.5))
    goal = np.array(config.goal)
    ax.add_patch(plt.Circle(goal, config.goal_tolerance, color=INK, alpha=0.08, lw=0))
    ax.scatter(*goal, marker="*", s=120, color=INK, zorder=5)
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    style_axis(ax)


def plot_paths(ax, gap, runs, method):
    draw_environment(ax, gap)
    for r in runs:
        s = r["states"]
        if r["outcome"] != "success":
            ax.plot(s[:, 0], s[:, 1], color=FAIL, lw=1, ls="--")
            if r["outcome"] == "collision":
                ax.scatter(s[-1, 0], s[-1, 1], marker="x", s=60, linewidths=2,
                           color=CROSS, zorder=6)
            else:   # timeout: 멈춘 위치 표시
                ax.scatter(s[-1, 0], s[-1, 1], marker="o", s=30, facecolors="none",
                           edgecolors=INK, linewidths=1.2, zorder=6)
            continue
        key = "single" if method == "single" else r["side"]
        ax.plot(s[:, 0], s[:, 1], color=PATH_COLORS[key], lw=1, alpha=0.45)

    n    = len(runs)
    rate = lambda outcome: sum(r["outcome"] == outcome for r in runs) / n
    ax.set_title(f"{LABELS[method]}, gap = {gap} m\n"
                 f"success {rate('success'):.0%} · collision {rate('collision'):.0%}"
                 f" · timeout {rate('timeout'):.0%}",
                 fontsize=9, color="#0b0b0b")

    handles = ([plt.Line2D([], [], color=PATH_COLORS["single"], lw=1.5)] if method == "single"
               else [plt.Line2D([], [], color=PATH_COLORS["upper"], lw=1.5),
                     plt.Line2D([], [], color=PATH_COLORS["lower"], lw=1.5)])
    labels  = ["Success"] if method == "single" else ["Upper pass", "Lower pass"]
    handles += [plt.Line2D([], [], color=FAIL, lw=1, ls="--", marker="o",
                           markerfacecolor="none", markeredgecolor=INK),
                plt.Line2D([], [], color=FAIL, lw=1, ls="--", marker="x",
                           markeredgecolor=CROSS, markeredgewidth=2),
                plt.Line2D([], [], color=WALL_GRAY, marker="o", ls="", markersize=8, alpha=0.55)]
    labels  += ["Timeout (○ = final position)", "Collision", "Static wall"]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="lower left", ncol=2)


def plot_success_curve(ax, summaries):
    # 두 선이 겹쳐도 보이도록 단일 MPPI를 굵게 아래에, Multimodal을 가늘게 위에 그림
    styles = {"single":  dict(lw=5, markersize=13, zorder=2),
              "mm_both": dict(lw=2, markersize=7,  zorder=3)}
    for method in METHODS:
        rates = [summaries[(g, method)]["success_rate"] for g in GAPS]
        ax.plot(GAPS, rates, color=METHOD_COLORS[method], marker="o",
                markeredgecolor="white", markeredgewidth=2,
                label=LABELS[method], **styles[method])
        ax.annotate(f"{rates[-1]:.0%}", (GAPS[-1], rates[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

    # 두 선이 갈라지는 첫 gap 표시 (넓은 gap → 좁은 gap 순서)
    split = next((g for g in GAPS
                  if summaries[(g, "single")]["success_rate"]
                  != summaries[(g, "mm_both")]["success_rate"]), None)
    if split is not None:
        ax.axvline(split, color=INK, lw=1, ls=":")
        ax.annotate(f"lines diverge\nat gap = {split} m", (split, 0.5),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    else:
        ax.text(0.5, 0.5, "no divergence in this sweep", transform=ax.transAxes,
                ha="center", fontsize=8, color=INK)

    ax.set_xticks(GAPS, [f"{g}" for g in GAPS])
    ax.set_xlim(max(GAPS) + 0.05, min(GAPS) - 0.08)   # 오른쪽으로 갈수록 좁은(어려운) 통로
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("corridor gap [m]  (narrower →)", fontsize=9, color=INK)
    ax.set_title("Success rate vs corridor gap", fontsize=10, color="#0b0b0b")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style_axis(ax)
    ax.grid(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds   = list(range(n_seeds))

    jobs = [(method, seed, g) for g in GAPS for method in METHODS for seed in seeds]
    with Pool() as pool:
        results = pool.starmap(run_once, jobs)

    runs = {}
    for (method, _, g), r in zip(jobs, results):
        runs.setdefault((g, method), []).append(r)
    summaries = {key: summarize(rs) for key, rs in runs.items()}

    print(f"{'gap':>4s}  {'stop y':>6s}  {'method':10s} {'success':>8s} {'collision':>9s}"
          f" {'timeout':>8s}  {'min clearance [m]':>17s}  {'steps (success)':>15s}"
          f"  {'upper/lower':>11s}")
    for g in GAPS:
        for method in METHODS:
            s       = summaries[(g, method)]
            timeout = 1.0 - s["success_rate"] - s["collision_rate"]
            sides   = [r["side"] for r in runs[(g, method)] if r["outcome"] == "success"]
            print(f"{g:4.2f}  {stop_y(g):+6.2f}  "
                  f"{'single' if method == 'single' else 'multimodal':10s}"
                  f" {s['success_rate']:8.0%} {s['collision_rate']:9.0%} {timeout:8.0%}"
                  f"  {s['clear_mean']:+8.3f} ± {s['clear_std']:.3f}"
                  f"  {s['steps_mean']:7.1f} ± {s['steps_std']:5.1f}"
                  f"  {sides.count('upper'):5d}/{sides.count('lower'):<5d}")

    fig = plt.figure(figsize=(19, 12))
    gs  = GridSpec(len(PATH_GAPS), 3, figure=fig, width_ratios=[1.3, 1.3, 1],
                   wspace=0.22, hspace=0.3)
    for row, g in enumerate(PATH_GAPS):
        for col, method in enumerate(METHODS):
            plot_paths(fig.add_subplot(gs[row, col]), g, runs[(g, method)], method)
    plot_success_curve(fig.add_subplot(gs[:, 2]), summaries)
    fig.suptitle(
        f"Narrowing corridor (static walls + closing obstacles, v_y = {V_Y} m/s): "
        f"single vs multimodal MPPI  (obs_dynamic=True, K={config.K} = "
        f"{K_UPPER}+{config.K - K_UPPER}, bias=±{config.multimodal_bias} rad/s, "
        f"decay within {2 * BIAS_DECAY_DIST}→{BIAS_DECAY_DIST} m, seeds 0~{n_seeds - 1})",
        fontsize=11, color="#0b0b0b",
    )

    output_path = Path(__file__).resolve().parent / "narrowing_corridor_experiment.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print("그래프 저장:", output_path)


if __name__ == "__main__":
    main()
