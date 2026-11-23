"""
동적 장애물(등속도 모델) 실험

실험 A: 장애물 1개가 경로를 수직 횡단
실험 B: 장애물 2개가 서로 반대 방향으로 이동
실험 C: 장애물 속도가 50스텝 이후 방향 반전

각 실험에서 정적 처리(obs_dynamic=False)와 동적 처리(obs_dynamic=True)를
같은 시드(0~49)로 비교한다. 두 처리 모두 매 스텝 장애물의 현재 위치는
관측하고, 동적 처리만 현재 속도로 롤아웃 구간의 미래 위치를 예측한다.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import config
from core.dynamic_obstacles import EXPERIMENTS, MultiObstacleController
from core.dynamics   import step


def make_cfg(**overrides):
    """config 모듈을 복사한 실행별 설정 (원본 config는 건드리지 않음)"""
    base = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    base.update(overrides)
    return SimpleNamespace(**base)


def run_once(key, dynamic, seed):
    cfg = make_cfg(obs_dynamic=dynamic, seed=seed)
    return simulate(EXPERIMENTS[key], cfg, MultiObstacleController(cfg))


def simulate(exp, cfg, controller):
    """controller로 exp 환경을 한 번 실행 (장애물은 velocity(k)로 이동)"""
    goal      = np.array(cfg.goal)
    obstacles = np.array(exp["obstacles"], dtype=float)
    state     = np.array(cfg.initial_state, dtype=float)

    states      = [state.copy()]
    obs_history = [obstacles.copy()]
    min_clear   = np.inf
    outcome     = "timeout"

    for k in range(cfg.max_steps):
        velocities = exp["velocity"](k)
        # 컨트롤러는 장애물의 현재 위치·속도를 관측
        controller.obstacles  = obstacles.copy()
        controller.velocities = velocities

        action    = controller.command(state)
        state     = step(state, action, cfg.dt)
        obstacles[:, :2] += velocities * cfg.dt
        if "y_limits" in exp:   # 장애물별 y 이동 범위 (범위 끝에서 정지)
            lo, hi = np.array(exp["y_limits"]).T
            obstacles[:, 1] = np.clip(obstacles[:, 1], lo, hi)

        states.append(state.copy())
        obs_history.append(obstacles.copy())

        clearance = np.min(
            np.linalg.norm(state[:2] - obstacles[:, :2], axis=1)
            - obstacles[:, 2]
        )
        min_clear = min(min_clear, clearance)

        if clearance <= 0.0:
            outcome = "collision"
            break
        if np.linalg.norm(state[:2] - goal) <= cfg.goal_tolerance:
            outcome = "success"
            break

    return {
        "states":    np.asarray(states),
        "obstacles": np.asarray(obs_history),
        "outcome":   outcome,
        "min_clear": min_clear,
        "steps":     len(states) - 1,
    }


def summarize(runs):
    outcomes = np.array([r["outcome"] for r in runs])
    success  = outcomes == "success"
    return {
        "success_rate":   success.mean(),
        "collision_rate": (outcomes == "collision").mean(),
        "min_clear_mean": np.mean([r["min_clear"] for r in runs]),
        "min_clear_min":  np.min([r["min_clear"] for r in runs]),
        "mean_steps":     (np.mean([r["steps"] for r in runs if r["outcome"] == "success"])
                           if success.any() else np.nan),
    }


# ── 시각화
COLORS = {False: "#eb6834", True: "#2a78d6"}   # static: orange, dynamic: blue
LABELS = {False: "Static (obs_dynamic=False)", True: "Dynamic (obs_dynamic=True)"}
INK    = "#52514e"
GRID   = "#e4e3df"


def plot_trajectories(ax, exp, runs, dynamic, n_seeds):
    color = COLORS[dynamic]
    goal  = np.array(config.goal)

    # 장애물 이동 경로 (가장 오래 진행된 실행 기준, 장애물 궤적은 시드와 무관)
    obs_path = max(runs, key=lambda r: len(r["obstacles"]))["obstacles"]
    for i in range(obs_path.shape[1]):
        start, end, r = obs_path[0, i, :2], obs_path[-1, i, :2], obs_path[0, i, 2]
        ax.add_patch(plt.Circle(start, r, color="#8a8984", alpha=0.18, lw=0))
        ax.add_patch(plt.Circle(end, r, fill=False, edgecolor="#8a8984",
                                linestyle="--", lw=1))
        ax.plot(obs_path[:, i, 0], obs_path[:, i, 1], color="#8a8984", lw=1)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1))

    for r in runs:
        s = r["states"]
        ax.plot(s[:, 0], s[:, 1], color=color, alpha=0.35, lw=1)
        if r["outcome"] == "collision":
            ax.scatter(s[-1, 0], s[-1, 1], marker="x", s=60, linewidths=2,
                       color="#e34948", zorder=6)

    ax.add_patch(plt.Circle(goal, config.goal_tolerance, color=INK, alpha=0.08, lw=0))
    ax.scatter(*goal, marker="*", s=120, color=INK, zorder=5)

    n_ok  = sum(r["outcome"] == "success" for r in runs)
    n_col = sum(r["outcome"] == "collision" for r in runs)
    mode  = "Dynamic" if dynamic else "Static"
    ax.set_title(f"{exp['title']}\n{mode}: success {n_ok}/{n_seeds}, "
                 f"collision {n_col}/{n_seeds}", fontsize=9, color="#0b0b0b")
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_metrics(ax, summaries, exp_key):
    metrics = [("success_rate", "Success"), ("collision_rate", "Collision")]
    x     = np.arange(len(metrics))
    width = 0.36
    for j, dynamic in enumerate([False, True]):
        vals = [summaries[(exp_key, dynamic)][m] for m, _ in metrics]
        bars = ax.bar(x + (j - 0.5) * (width + 0.02), vals, width,
                      color=COLORS[dynamic], label=LABELS[dynamic])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8, color=INK)

    s_st, s_dy = summaries[(exp_key, False)], summaries[(exp_key, True)]
    ax.text(0.5, -0.30,
            f"min clearance (mean)  static {s_st['min_clear_mean']:+.3f} m"
            f" / dynamic {s_dy['min_clear_mean']:+.3f} m\n"
            f"mean steps (success)  static {s_st['mean_steps']:.1f}"
            f" / dynamic {s_dy['mean_steps']:.1f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK)

    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds   = list(range(n_seeds))

    jobs = [(key, dynamic, seed)
            for key in EXPERIMENTS for dynamic in (False, True) for seed in seeds]
    with Pool() as pool:
        results = pool.starmap(run_once, jobs)

    all_runs, summaries = {}, {}
    for key in EXPERIMENTS:
        for dynamic in (False, True):
            runs = [r for (k, d, _), r in zip(jobs, results) if k == key and d == dynamic]
            all_runs[(key, dynamic)]  = runs
            summaries[(key, dynamic)] = summarize(runs)
            s = summaries[(key, dynamic)]
            print(f"[{key}] {'dynamic' if dynamic else 'static ':7s}  "
                  f"success {s['success_rate']:6.1%}  "
                  f"collision {s['collision_rate']:6.1%}  "
                  f"min clear mean {s['min_clear_mean']:+.3f} m "
                  f"(worst {s['min_clear_min']:+.3f} m)  "
                  f"mean steps {s['mean_steps']:.1f}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13),
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    for row, (key, exp) in enumerate(EXPERIMENTS.items()):
        plot_trajectories(axes[row, 0], exp, all_runs[(key, False)], False, n_seeds)
        plot_trajectories(axes[row, 1], exp, all_runs[(key, True)],  True,  n_seeds)
        plot_metrics(axes[row, 2], summaries, key)

    fig.suptitle(
        f"MPPI with constant-velocity obstacle prediction  "
        f"(K={config.K}, H={config.H}, dt={config.dt}, seeds 0~{n_seeds - 1})",
        fontsize=11, color="#0b0b0b", y=0.995,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[d]) for d in (False, True)]
    fig.legend(handles, [LABELS[False], LABELS[True]], loc="upper center",
               bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)

    output_path = Path(__file__).resolve().parent / "dynamic_obstacle_experiment.png"
    fig.savefig(output_path, dpi=130)
    print("그래프 저장:", output_path)


if __name__ == "__main__":
    main()
