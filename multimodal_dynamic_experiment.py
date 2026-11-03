"""
동적 장애물 환경에서 단일 MPPI vs 사전 샘플 분할 Multimodal MPPI 비교

환경 (실행 인자로 선택: python multimodal_dynamic_experiment.py [시드수] [B|headon])
  B      : dynamic_obstacle_experiment.py의 실험 B (장애물 2개가 반대 방향으로 이동)
  headon : 장애물 1개가 경로 정중앙에서 로봇 쪽으로 정면 접근

단일 MPPI      : K=1000개 전부 N(0, noise_std)
Multimodal MPPI: upper/lower 모드가 각자 nominal 제어열을 유지
                 같은 K=1000개 노이즈를 500/500으로 나눠
                 upper 샘플 = controls_upper + noise + [0, +bias]
                 lower 샘플 = controls_lower + noise + [0, -bias]
                 모드별로 rollout + 비용 계산, 최소 비용이 더 낮은 모드의 첫 입력을 적용
                 (bias는 샘플링에만 사용, 가중평균 업데이트에서는 제외)

  - 이전 (mm_select): 선택된 모드만 업데이트, 나머지는 shift만, bias 고정
  - 수정 (mm_both)  : 두 모드 모두 매 스텝 자기 샘플로 업데이트,
                      목표 근처에서 bias decay
                      bias *= clip((dist_to_goal - 0.5) / 0.5, 0, 1)

모든 방법이 같은 시드의 rng에서 매 스텝 같은 노이즈를 뽑는다.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config
from core.sampling import sample_noise
from core.rollout  import rollout
from core.update   import update
from dynamic_obstacle_experiment import (
    EXPERIMENTS, MultiObstacleController, make_cfg, simulate,
)

DECISION_Y = 0.05   # 장애물 중심 y로부터 이만큼 벗어나야 위/아래로 결정된 것으로 봄 [m]

ENVS = {
    "B": {
        **EXPERIMENTS["B"],
        "label":   "exp. B",
        "methods": ("single", "mm_select", "mm_both"),
        "output":  "multimodal_dynamic_experiment.png",
    },
    "headon": {
        "title":     "head-on obstacle",
        "obstacles": [[3.5, 0.0, 0.5]],
        "velocity":  lambda k: np.array([[-0.3, 0.0]]),
        "label":     "head-on",
        "methods":   ("single", "mm_both"),
        "output":    "multimodal_dynamic_headon.png",
    },
}
K_UPPER = 500   # 나머지 K - K_UPPER개가 lower 모드
BIAS_DECAY_DIST = 0.5   # 목표까지 이 거리 이하면 bias = 0 (2배 거리부터 선형 감소)


class MultimodalController(MultiObstacleController):
    """
    사전 분할 Multimodal MPPI: upper/lower 모드가 각자 nominal 제어열을 유지
    (장애물 여러 개 + 동적 예측은 부모 클래스 비용 사용)

    update_all=False, bias_decay=False → 이전 구조 (선택 모드만 업데이트)
    update_all=True,  bias_decay=True  → 수정 구조
    """

    def __init__(self, cfg, update_all=True, bias_decay=True):
        super().__init__(cfg)
        self.update_all    = update_all
        self.bias_decay    = bias_decay
        self.mode_controls = {
            "upper": self.controls.copy(),
            "lower": self.controls.copy(),
        }
        self.mode_history = []

    def bias_scale(self, state):
        if not self.bias_decay:
            return 1.0
        dist = np.linalg.norm(state[:2] - np.array(self.cfg.goal))
        return np.clip((dist - BIAS_DECAY_DIST) / BIAS_DECAY_DIST, 0.0, 1.0)

    def command(self, state):
        cfg   = self.cfg
        noise = sample_noise(cfg.K, cfg.H, cfg.noise_std, self.rng)
        bias  = cfg.multimodal_bias * self.bias_scale(state)
        split = {
            "upper": (slice(None, K_UPPER), np.array([0.0, +bias])),
            "lower": (slice(K_UPPER, None), np.array([0.0, -bias])),
        }

        # 모드별로 독립적으로 rollout + 비용 계산
        # 샘플 = controls_mode + noise + bias  (bias는 샘플링에만 사용)
        results = {}
        for mode, (idx, offset) in split.items():
            self.controls = self.mode_controls[mode]   # compute_costs의 제어 비용 기준
            trajectories, effective_noise = rollout(
                state, self.controls, noise[idx] + offset, cfg.dt,
                v_min=cfg.v_min,
                v_max=cfg.v_max,
                omega_max=cfg.omega_max,
            )
            costs = self.compute_costs(trajectories, effective_noise)
            results[mode] = (effective_noise - offset, costs)   # 업데이트에서 bias 제외

        # 모드 내 최소 비용이 더 낮은 모드의 계획을 실행
        mode = min(results, key=lambda m: np.min(results[m][1]))
        updated = {}
        for m in (results if self.update_all else [mode]):
            update_noise, costs = results[m]
            updated[m] = update(self.mode_controls[m], update_noise, costs, cfg)
            self.mode_controls[m] = updated[m][0]
        updated_controls, weights = updated[mode]
        costs = results[mode][1]

        self.mode_history.append(mode)
        self.weights_history.append(weights)
        self.ess_history.append(1.0 / np.sum(weights ** 2))
        self.cost_history.append(np.min(costs))
        action    = updated_controls[0].copy()
        action[0] = np.clip(action[0], cfg.v_min, cfg.v_max)
        action[1] = np.clip(action[1], -cfg.omega_max, cfg.omega_max)

        # 두 모드 모두 shift
        for m, controls in self.mode_controls.items():
            self.mode_controls[m] = np.vstack([controls[1:], controls[-1:]])
        self.controls = self.mode_controls[mode]
        return action


def avoidance_side(result):
    """
    로봇이 장애물 1의 x를 지날 때 그 시점 장애물 1보다 위/아래 어느 쪽인지
    (장애물이 움직이므로 같은 시각의 장애물 위치와 비교)
    """
    states, obstacles = result["states"], result["obstacles"]
    k = np.argmin(np.abs(states[:, 0] - obstacles[:, 0, 0]))
    return "upper" if states[k, 1] > obstacles[k, 0, 1] else "lower"


def make_controller(method, cfg):
    if method == "single":
        return MultiObstacleController(cfg)
    if method == "mm_select":
        return MultimodalController(cfg, update_all=False, bias_decay=False)
    return MultimodalController(cfg, update_all=True, bias_decay=True)


def decision_metrics(result):
    """
    회피 방향 결정 지표 (장애물 1 기준, 장애물이 움직이므로 매 스텝 현재 위치와 비교)
    - half_*   : 로봇 x가 그 시점 장애물 x의 절반에 도달한 순간 위/아래 어느 쪽인지
    - decision : 장애물을 지나기 전, 출발-목표 직선(y = goal y)에서 DECISION_Y 이상
                 벗어난 뒤 다시 돌아오지 않은 첫 스텝 (그 시점 로봇-장애물 x 간격도 기록)
                 장애물이 중심선 밖에서 출발해도 의미가 유지되도록 직선 기준으로 판단
                 (headon처럼 장애물 y = 0이면 장애물 중심 기준과 동일)
    """
    states, obstacles = result["states"], result["obstacles"]
    dy      = states[:, 1] - obstacles[:, 0, 1]
    dy_line = states[:, 1] - config.goal[1]

    reached = np.flatnonzero(states[:, 0] >= obstacles[:, 0, 0] / 2)
    k_half  = reached[0] if len(reached) else None
    if k_half is None:
        half_side, half_dy = "undecided", np.nan
    else:
        half_dy   = dy[k_half]
        half_side = ("upper" if half_dy > DECISION_Y else
                     "lower" if half_dy < -DECISION_Y else "undecided")

    passed = np.flatnonzero(states[:, 0] >= obstacles[:, 0, 0])
    decision_step, decision_gap = np.nan, np.nan
    if len(passed):
        k_pass    = passed[0]
        undecided = np.flatnonzero(np.abs(dy_line[:k_pass + 1]) < DECISION_Y)
        k_dec     = undecided[-1] + 1 if len(undecided) else 0
        if k_dec <= k_pass:
            decision_step = k_dec
            decision_gap  = obstacles[k_dec, 0, 0] - states[k_dec, 0]
    return {
        "half_side":     half_side,
        "half_dy":       half_dy,
        "decision_step": decision_step,
        "decision_gap":  decision_gap,
    }


def run_once(method, seed, env="B", dynamic=True):
    return run_env(method, seed, ENVS[env], dynamic)


def run_env(method, seed, exp, dynamic=True, make=make_controller):
    """임의의 환경 dict(obstacles, velocity)로 한 번 실행 (make: 컨트롤러 생성 함수)"""
    cfg        = make_cfg(obs_dynamic=dynamic, seed=seed)
    controller = make(method, cfg)

    # 계획 1회(command 호출)당 계산 시간 측정
    plan_ms = []
    command = controller.command
    def timed_command(state):
        start  = time.perf_counter()
        action = command(state)
        plan_ms.append((time.perf_counter() - start) * 1e3)
        return action
    controller.command = timed_command

    result = simulate(exp, cfg, controller)
    result["plan_ms"] = np.array(plan_ms)
    result["side"] = avoidance_side(result)
    result.update(decision_metrics(result))
    if method != "single":
        modes = controller.mode_history
        result["modes"]    = modes
        result["switches"] = sum(a != b for a, b in zip(modes, modes[1:]))
    return result


def summarize(runs):
    outcomes = np.array([r["outcome"] for r in runs])
    clear    = np.array([r["min_clear"] for r in runs])
    steps    = np.array([r["steps"] for r in runs if r["outcome"] == "success"])
    summary  = {
        "success_rate":   np.mean(outcomes == "success"),
        "collision_rate": np.mean(outcomes == "collision"),
        "clear_mean":     clear.mean(),
        "clear_std":      clear.std(),
        "steps_mean":     steps.mean() if len(steps) else np.nan,
        "steps_std":      steps.std() if len(steps) else np.nan,
    }
    half_sides = [r["half_side"] for r in runs]
    for side in ("upper", "lower", "undecided"):
        summary[f"half_{side}"] = half_sides.count(side)
    half_dy  = np.abs([r["half_dy"] for r in runs])
    dec_step = np.array([r["decision_step"] for r in runs], dtype=float)
    dec_gap  = np.array([r["decision_gap"] for r in runs], dtype=float)
    # 해당 실행이 하나도 없으면 (예: 모두 장애물 앞에서 timeout) nan
    nanmean = lambda a: np.nanmean(a) if np.any(~np.isnan(a)) else np.nan
    nanstd  = lambda a: np.nanstd(a) if np.any(~np.isnan(a)) else np.nan
    summary.update({
        "half_dy_mean":  nanmean(half_dy),
        "half_dy_std":   nanstd(half_dy),
        "dec_step_mean": nanmean(dec_step),
        "dec_step_std":  nanstd(dec_step),
        "dec_gap_mean":  nanmean(dec_gap),
        "dec_gap_std":   nanstd(dec_gap),
        "dec_n":         int(np.sum(~np.isnan(dec_step))),
    })
    plan_ms = np.concatenate([r["plan_ms"] for r in runs])   # 모든 시드의 계획 호출
    summary["plan_ms_mean"] = plan_ms.mean()
    summary["plan_ms_std"]  = plan_ms.std()
    if "switches" in runs[0]:
        switches = np.array([r["switches"] for r in runs])
        summary["switch_mean"] = switches.mean()
        summary["switch_std"]  = switches.std()
        summary["switch_max"]  = switches.max()
    return summary


# ── 시각화
INK      = "#52514e"
GRID     = "#e4e3df"
OBS_GRAY = "#8a8984"
FAIL     = "#a3a29c"
CROSS    = "#e34948"
PATH_COLORS   = {"single": "#2a78d6", "upper": "#008300", "lower": "#eb6834"}
METHOD_COLORS = {"single": "#2a78d6", "mm_select": "#e87ba4", "mm_both": "#6250d6"}
METHOD_LABELS = {
    "single":    "Single MPPI",
    "mm_select": "Multimodal (prev: selected-mode update)",
    "mm_both":   "Multimodal (new: both update + bias decay)",
}
SHORT_LABELS = {"single": "single", "mm_select": "mm prev", "mm_both": "mm new"}


def style_axis(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def draw_environment(ax, exp):
    for (x, y, r), (vx, vy) in zip(exp["obstacles"], exp["velocity"](0)):
        ax.add_patch(plt.Circle((x, y), r, color=OBS_GRAY, alpha=0.25, lw=0))
        ax.annotate("", xy=(x + vx, y + vy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=OBS_GRAY, lw=1.5))
    goal = np.array(config.goal)
    ax.add_patch(plt.Circle(goal, config.goal_tolerance, color=INK, alpha=0.08, lw=0))
    ax.scatter(*goal, marker="*", s=120, color=INK, zorder=5)
    ax.set_aspect("equal")
    style_axis(ax)


def plot_paths(ax, exp, runs, method, n_seeds, xlim):
    draw_environment(ax, exp)
    ax.set_xlim(*xlim)
    ax.set_ylim(-2.0, 2.0)
    for r in runs:
        s = r["states"]
        if r["outcome"] == "collision":
            ax.plot(s[:, 0], s[:, 1], color=FAIL, lw=1, ls="--")
            ax.scatter(s[-1, 0], s[-1, 1], marker="x", s=60, linewidths=2,
                       color=CROSS, zorder=6)
            continue
        key   = "single" if method == "single" else r["side"]
        alpha = 0.45 if r["outcome"] == "success" else 0.2   # timeout은 흐리게
        ax.plot(s[:, 0], s[:, 1], color=PATH_COLORS[key], lw=1, alpha=alpha)
    dec = [r["states"][int(r["decision_step"])] for r in runs
           if not np.isnan(r["decision_step"])]
    if dec:
        dec = np.array(dec)
        ax.scatter(dec[:, 0], dec[:, 1], s=10, color="#0b0b0b", zorder=6)

    n_ok  = sum(r["outcome"] == "success" for r in runs)
    n_col = sum(r["outcome"] == "collision" for r in runs)
    n_to  = sum(r["outcome"] == "timeout" for r in runs)
    title = (f"{METHOD_LABELS[method]}\nsuccess {n_ok}/{n_seeds}, "
             f"collision {n_col}/{n_seeds}, timeout {n_to}/{n_seeds}")
    if method != "single":
        n_up = sum(r["side"] == "upper" for r in runs if r["outcome"] != "collision")
        n_lo = sum(r["side"] == "lower" for r in runs if r["outcome"] != "collision")
        title += f"  (upper {n_up}, lower {n_lo})"
    ax.set_title(title, fontsize=9, color="#0b0b0b")

    if method == "single":
        handles = [plt.Line2D([], [], color=PATH_COLORS["single"], lw=1.5)]
        labels  = ["Success"]
    else:
        handles = [plt.Line2D([], [], color=PATH_COLORS["upper"], lw=1.5),
                   plt.Line2D([], [], color=PATH_COLORS["lower"], lw=1.5)]
        labels  = ["Upper pass", "Lower pass"]
    if any(r["outcome"] == "timeout" for r in runs):
        color    = PATH_COLORS["single"] if method == "single" else PATH_COLORS["upper"]
        handles += [plt.Line2D([], [], color=color, lw=1, alpha=0.3)]
        labels  += ["Timeout (faint)"]
    handles += [plt.Line2D([], [], color=FAIL, lw=1, ls="--", marker="x",
                           markeredgecolor=CROSS, markeredgewidth=2)]
    labels  += ["Collision"]
    handles += [plt.Line2D([], [], color="#0b0b0b", marker="o", ls="", markersize=3)]
    labels  += [f"Decision point (|dy| ≥ {DECISION_Y} m)"]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="lower left")


def plot_metrics(ax, summaries, methods):
    metrics = [("success_rate", "Success"), ("collision_rate", "Collision")]
    x     = np.arange(len(metrics))
    width = 0.78 / len(methods)
    for j, method in enumerate(methods):
        vals = [summaries[method][m] for m, _ in metrics]
        bars = ax.bar(x + (j - (len(methods) - 1) / 2) * (width + 0.02), vals, width,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8, color=INK)

    lines = []
    for key, label, unit in (("clear", "min clearance", " m"), ("steps", "steps (success)", "")):
        for i, m in enumerate(methods):
            s   = summaries[m]
            fmt = "{:+.3f} ± {:.3f}" if key == "clear" else "{:.1f} ± {:.1f}"
            val = fmt.format(s[f"{key}_mean"], s[f"{key}_std"])
            lines.append(f"{label if i == 0 else '':16s}{SHORT_LABELS[m]:8s} {val}{unit}")
    for i, m in enumerate(methods):
        s = summaries[m]
        lines.append(f"{'half-pt side' if i == 0 else '':16s}{SHORT_LABELS[m]:8s} "
                     f"U {s['half_upper']} / L {s['half_lower']} / ? {s['half_undecided']}"
                     f"  |dy| {s['half_dy_mean']:.3f} m")
    for i, m in enumerate(methods):
        s = summaries[m]
        lines.append(f"{'decision step' if i == 0 else '':16s}{SHORT_LABELS[m]:8s} "
                     f"{s['dec_step_mean']:.1f} ± {s['dec_step_std']:.1f}"
                     f"  gap {s['dec_gap_mean']:.2f} m")
    for i, m in enumerate(methods):
        s = summaries[m]
        lines.append(f"{'plan time' if i == 0 else '':16s}{SHORT_LABELS[m]:8s} "
                     f"{s['plan_ms_mean']:.1f} ± {s['plan_ms_std']:.1f} ms")
    for i, m in enumerate([m for m in methods if m != "single"]):
        s = summaries[m]
        lines.append(f"{'mode switches' if i == 0 else '':16s}{SHORT_LABELS[m]:8s} "
                     f"{s['switch_mean']:.1f} ± {s['switch_std']:.1f} (max {s['switch_max']})")
    ax.text(0.5, -0.12, "\n".join(lines),
            transform=ax.transAxes, ha="center", va="top", multialignment="left",
            fontsize=8, color=INK, family="monospace")

    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=1,
              bbox_to_anchor=(0.5, 1.0))
    style_axis(ax)
    ax.grid(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    env     = sys.argv[2] if len(sys.argv) > 2 else "B"
    exp     = ENVS[env]
    seeds   = list(range(n_seeds))
    methods = exp["methods"]

    jobs = [(method, seed, env) for method in methods for seed in seeds]
    with Pool() as pool:
        results = pool.starmap(run_once, jobs)

    runs      = {m: [r for (jm, _, _), r in zip(jobs, results) if jm == m] for m in methods}
    summaries = {m: summarize(runs[m]) for m in methods}
    for m in methods:
        s = summaries[m]
        line = (f"{METHOD_LABELS[m]:44s} success {s['success_rate']:6.1%}  "
                f"collision {s['collision_rate']:6.1%}  "
                f"min clear {s['clear_mean']:+.3f} ± {s['clear_std']:.3f} m  "
                f"steps {s['steps_mean']:.1f} ± {s['steps_std']:.1f}  "
                f"half-pt U/L/? {s['half_upper']}/{s['half_lower']}/{s['half_undecided']} "
                f"|dy| {s['half_dy_mean']:.3f} ± {s['half_dy_std']:.3f} m  "
                f"decision step {s['dec_step_mean']:.1f} ± {s['dec_step_std']:.1f} "
                f"(gap {s['dec_gap_mean']:.2f} ± {s['dec_gap_std']:.2f} m, n={s['dec_n']})  "
                f"plan {s['plan_ms_mean']:.1f} ± {s['plan_ms_std']:.1f} ms")
        if m != "single":
            line += (f"  switches {s['switch_mean']:.1f} ± {s['switch_std']:.1f}"
                     f" (max {s['switch_max']})")
        print(line)

    n_panels  = len(methods)
    fig, axes = plt.subplots(1, n_panels + 1, figsize=(5.6 * n_panels + 5.2, 6.5),
                             gridspec_kw={"width_ratios": [1.4] * n_panels + [1.1]})
    # 목표를 지나쳐 맴도는 timeout 경로까지 보이도록 x 범위를 맞춤
    x_max = max(5.5, max(r["states"][:, 0].max() for m in methods for r in runs[m]) + 0.2)
    for ax, m in zip(axes, methods):
        plot_paths(ax, exp, runs[m], m, n_seeds, (-0.3, x_max))
    plot_metrics(axes[-1], summaries, methods)
    fig.suptitle(
        f"Dynamic obstacles ({exp['label']}): single vs pre-split multimodal MPPI  "
        f"(K={config.K} = {K_UPPER}+{config.K - K_UPPER}, "
        f"bias=±{config.multimodal_bias} rad/s, decay within {2 * BIAS_DECAY_DIST}→{BIAS_DECAY_DIST} m, seeds 0~{n_seeds - 1})",
        fontsize=11, color="#0b0b0b",
    )
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / exp["output"]
    fig.savefig(output_path, dpi=130)
    print("그래프 저장:", output_path)


if __name__ == "__main__":
    main()
