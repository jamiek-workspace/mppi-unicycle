from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_summary(result, cfg, output_path):
    """실행 결과를 2x2 요약 그림으로 저장한다."""
    states   = result["states"]
    controls = result["controls"]
    ess      = np.asarray(result["ess"])
    goal     = np.array(cfg.goal)
    obstacle = np.array(cfg.obstacle)

    state_times = np.arange(len(states)) * cfg.dt
    goal_distances = np.linalg.norm(states[:, :2] - goal, axis=1)
    clearances = (
        np.linalg.norm(states[:, :2] - obstacle[:2], axis=1)
        - obstacle[2]
    )
    min_idx = np.argmin(clearances)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_path, ax_goal, ax_clearance, ax_input = axes.ravel()

    # A. 실제 이동 경로
    ax_path.add_patch(plt.Circle(
        obstacle[:2], obstacle[2],
        color="crimson", alpha=0.35, label="Obstacle",
    ))
    ax_path.add_patch(plt.Circle(
        obstacle[:2], obstacle[2] + cfg.margin,
        fill=False, edgecolor="crimson", linestyle=":",
        label="Proximity-cost boundary",
    ))
    ax_path.add_patch(plt.Circle(
        goal, cfg.goal_tolerance,
        color="purple", alpha=0.12, label="Goal tolerance",
    ))
    ax_path.plot(
        states[:, 0], states[:, 1],
        color="navy", linewidth=2.5, label="Executed trajectory",
    )
    ax_path.scatter(*states[0, :2], color="green", s=50,
                    label="Start", zorder=5)
    ax_path.scatter(*states[-1, :2], color="darkorange", s=50,
                    label="End", zorder=5)
    ax_path.scatter(*goal, color="purple", marker="*", s=150,
                    label="Goal", zorder=5)

    arrow_states = states[::10]
    ax_path.quiver(
        arrow_states[:, 0], arrow_states[:, 1],
        np.cos(arrow_states[:, 2]), np.sin(arrow_states[:, 2]),
        angles="xy", scale_units="xy", scale=5,
        color="royalblue", width=0.005,
    )
    ax_path.set_title("Executed path and heading")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True)
    ax_path.legend(fontsize=7, loc="upper left")

    # B. 목표까지의 거리
    ax_goal.plot(state_times, goal_distances, color="navy",
                 label="Goal distance")
    ax_goal.axhline(cfg.goal_tolerance, color="purple", linestyle="--",
                    label=f"Goal tolerance: {cfg.goal_tolerance} m")
    ax_goal.set_title("Distance to goal")
    ax_goal.set_xlabel("Simulation time [s]")
    ax_goal.set_ylabel("Distance [m]")
    ax_goal.grid(True)
    ax_goal.legend()

    # C. 장애물 여유 거리
    ax_clearance.plot(state_times, clearances, color="teal",
                      label="Obstacle clearance")
    ax_clearance.axhline(0.0, color="crimson", linestyle="--",
                         label="Collision boundary")
    ax_clearance.axhline(cfg.margin, color="darkorange", linestyle=":",
                         label="Proximity-cost threshold")
    ax_clearance.scatter(
        state_times[min_idx], clearances[min_idx],
        color="navy", zorder=5,
        label=f"Minimum: {clearances.min():.3f} m",
    )
    ax_clearance.set_title("Clearance at recorded states")
    ax_clearance.set_xlabel("Simulation time [s]")
    ax_clearance.set_ylabel("Clearance [m]")
    ax_clearance.grid(True)
    ax_clearance.legend(fontsize=8)

    # D. 적용 입력 (단위별 좌우 축)
    ax_omega = ax_input.twinx()
    v_line = ax_input.stairs(controls[:, 0], state_times,
                             color="navy", label="v")
    omega_line = ax_omega.stairs(controls[:, 1], state_times,
                                 color="teal", label="omega")
    ax_input.set_title(f"Applied controls (mean ESS={ess.mean():.1f})")
    ax_input.set_xlabel("Simulation time [s]")
    ax_input.set_ylabel("v [m/s]", color="navy")
    ax_omega.set_ylabel("omega [rad/s]", color="teal")
    ax_input.grid(True)
    ax_input.legend(
        handles=[v_line, omega_line],
        labels=["v [m/s]", "omega [rad/s]"],
        loc="best",
    )

    fig.suptitle(
        "MPPI receding horizon control\n"
        f"K={cfg.K}, H={cfg.H}, dt={cfg.dt} s, lambda={cfg.lam} | "
        f"Final goal distance={goal_distances[-1]:.3f} m | "
        f"Min clearance={clearances.min():.3f} m",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
