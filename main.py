from pathlib import Path

import numpy as np

import config as cfg
from core.controller import MPPIController
from core.dynamics   import step
from visualization   import plot_summary


def run():
    controller = MPPIController(cfg)
    goal       = np.array(cfg.goal)
    obstacle   = np.array(cfg.obstacle)
    state      = np.array(cfg.initial_state, dtype=float)

    states      = [state.copy()]
    controls    = []
    stop_reason = "최대 반복 횟수 도달"

    for k in range(cfg.max_steps):
        action = controller.command(state)
        state  = step(state, action, cfg.dt)

        states.append(state.copy())
        controls.append(action)

        goal_distance = np.linalg.norm(state[:2] - goal)
        clearance     = np.linalg.norm(state[:2] - obstacle[:2]) - obstacle[2]

        if k % 20 == 0:
            print(
                f"step={k + 1:3d}, "
                f"time={(k + 1) * cfg.dt:.2f} s, "
                f"goal_distance={goal_distance:.3f} m, "
                f"clearance={clearance:.3f} m, "
                f"ESS={controller.ess_history[-1]:.2f}"
            )

        if clearance <= 0.0:
            stop_reason = "실제 이동 상태에서 충돌 감지"
            break
        if goal_distance <= cfg.goal_tolerance:
            stop_reason = "목표 범위 도달"
            break

    return {
        "states":      np.asarray(states),
        "controls":    np.asarray(controls),
        "ess":         controller.ess_history,
        "stop_reason": stop_reason,
    }


def main():
    result   = run()
    states   = result["states"]
    goal     = np.array(cfg.goal)
    obstacle = np.array(cfg.obstacle)

    clearances = (
        np.linalg.norm(states[:, :2] - obstacle[:2], axis=1)
        - obstacle[2]
    )

    print("\n종료 이유:", result["stop_reason"])
    print("적용한 입력 수:", len(result["controls"]))
    print(f"시뮬레이션 시간: {len(result['controls']) * cfg.dt:.2f} s")
    print("최종 상태:", states[-1])
    print(f"최종 목표 거리: {np.linalg.norm(states[-1, :2] - goal):.4f} m")
    print(f"최소 여유 거리: {clearances.min():.4f} m")

    # controls 배열에서 v가 v_max에 걸린 횟수 출력
    clipped = np.sum(result["controls"][:, 0] >= cfg.v_max)
    print(f"action clip 횟수: {clipped} / {len(result['controls'])}")

    output_path = Path(__file__).resolve().parent / "mppi_summary.png"
    plot_summary(result, cfg, output_path)
    print("그래프 저장:", output_path)


if __name__ == "__main__":
    main()
