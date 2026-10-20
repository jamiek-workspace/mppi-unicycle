from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def unicycle_step(state, control, dt):
    """Step 1에서 검증한 Forward Euler 운동 모델."""
    x, y, theta = state
    v, omega = control

    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt,
    ])

def trajectory_cost(states, controls, goal, obstacle, dt):
    """한 궤적의 총비용, 항별 비용, 장애물 여유 거리를 계산한다."""
    q_goal = 1.0
    r_v = 0.1
    r_omega = 0.05
    q_obstacle = 100.0
    q_collision = 10000.0
    q_terminal = 10.0
    margin = 0.3

    # 각 입력을 적용한 다음 위치: 크기 (H, 2)
    positions = states[1:, :2]

    # 1. 목표 거리 비용
    goal_distance_sq = np.sum((positions - goal) ** 2, axis=1)
    goal_cost = q_goal * np.sum(goal_distance_sq) * dt

    # 2. 제어 입력 비용
    control_cost = np.sum(
        r_v * controls[:, 0] ** 2
        + r_omega * controls[:, 1] ** 2
    ) * dt

    # 3. 장애물 표면까지의 거리
    center = obstacle[:2]
    radius = obstacle[2]

    clearance = np.linalg.norm(positions - center, axis=1) - radius

    proximity = np.maximum(0.0, (margin - clearance) / margin) ** 2
    obstacle_cost = q_obstacle * np.sum(proximity) * dt

    collision_cost = q_collision * np.sum(clearance <= 0.0) * dt

    # 4. 예측 마지막 위치의 목표 거리 비용
    terminal_cost = q_terminal * goal_distance_sq[-1]

    parts = {
        "goal": goal_cost,
        "control": control_cost,
        "proximity": obstacle_cost,
        "collision": collision_cost,
        "terminal": terminal_cost,
    }

    # 최소 여유 거리는 초기 위치도 포함해 확인한다.
    all_clearance = (
        np.linalg.norm(states[:, :2] - center, axis=1) - radius
    )

    return sum(parts.values()), parts, np.min(all_clearance)

# 1. 설정
dt = 0.05
H = 100
K = 200
temperature = 1.0
noise_std = np.array([0.2, 0.8])

max_steps = 300                     # 최대 15초의 시뮬레이션
goal_tolerance = 0.2                # 목표에서 20 cm 이내면 종료

state = np.array([0.0, 0.0, 0.0])
goal = np.array([5.0, 0.0])
obstacle = np.array([2.5, 0.0, 0.5])

rng = np.random.default_rng(42)

# 처음에만 직진 계획으로 초기화
nominal_controls = np.zeros((H, 2))
nominal_controls[:, 0] = 1.0

# 실제로 적용한 입력과 이동한 상태를 기록
actual_states = [state.copy()]
applied_controls = []
ess_history = []
prediction_costs = []
prediction_clearances = []

stop_reason = "최대 반복 횟수 도달"

# 2. 반복 제어
for k in range(max_steps):
    # A. 현재 계획 주변에 새로운 후보 입력 생성
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(K, H, 2),
    )
    sampled_controls = nominal_controls[None, :, :] + noise

    # B. 모든 후보를 현재 실제 상태에서 예측
    trajectories = np.zeros((K, H + 1, 3))
    trajectories[:, 0, :] = state
    costs = np.zeros(K)

    for i in range(K):
        for t in range(H):
            trajectories[i, t + 1] = unicycle_step(
                trajectories[i, t],
                sampled_controls[i, t],
                dt,
            )

        costs[i], _, _ = trajectory_cost(
            trajectories[i],
            sampled_controls[i],
            goal,
            obstacle,
            dt,
        )

    # C. 비용 기반 가중치와 입력 업데이트
    rho = np.min(costs)
    weights = np.exp(-(costs - rho) / temperature)
    weights /= np.sum(weights)

    updated_controls = nominal_controls + np.sum(
        weights[:, None, None] * noise,
        axis=0,
    )

    ess = 1.0 / np.sum(weights ** 2)

    # D. 업데이트한 전체 계획의 예측 결과 확인
    updated_trajectory = np.zeros((H + 1, 3))
    updated_trajectory[0] = state

    for t in range(H):
        updated_trajectory[t + 1] = unicycle_step(
            updated_trajectory[t],
            updated_controls[t],
            dt,
        )

    predicted_cost, _, predicted_clearance = trajectory_cost(
        updated_trajectory,
        updated_controls,
        goal,
        obstacle,
        dt,
    )

    prediction_costs.append(predicted_cost)
    prediction_clearances.append(predicted_clearance)
    ess_history.append(ess)

    # E. 첫 입력만 적용해 실제 시뮬레이션 상태를 한 단계 진행
    action = updated_controls[0].copy()
    state = unicycle_step(state, action, dt)

    actual_states.append(state.copy())
    applied_controls.append(action)

    goal_distance = np.linalg.norm(state[:2] - goal)
    actual_clearance = (
        np.linalg.norm(state[:2] - obstacle[:2]) - obstacle[2]
    )

    if k % 20 == 0:
        print(
            f"step={k + 1:3d}, "
            f"time={(k + 1) * dt:.2f} s, "
            f"goal_distance={goal_distance:.3f} m, "
            f"clearance={actual_clearance:.3f} m, "
            f"ESS={ess:.2f}"
        )

    # F. 종료 조건: 이번에도 이산 위치에서 충돌 판정
    if actual_clearance <= 0.0:
        stop_reason = "실제 이동 상태에서 충돌 감지"
        break

    if goal_distance <= goal_tolerance:
        stop_reason = "목표 범위 도달"
        break

    # G. 残った計画を一つ前に移し、末尾の入力を複製
    nominal_controls = np.vstack([
        updated_controls[1:],
        updated_controls[-1:],
    ])

# 3. 結果の集計
actual_states = np.asarray(actual_states)
applied_controls = np.asarray(applied_controls)

actual_clearances = (
    np.linalg.norm(actual_states[:, :2] - obstacle[:2], axis=1)
    - obstacle[2]
)
final_goal_distance = np.linalg.norm(actual_states[-1, :2] - goal)

print("\n終了理由:", stop_reason)
print("適用した入力の数:", len(applied_controls))
print(f"シミュレーション時間: {len(applied_controls) * dt:.2f} s")
print("最終状態:", actual_states[-1])
print(f"最終目標距離: {final_goal_distance:.4f} m")
print(f"実際の離散状態の最小余裕距離: {actual_clearances.min():.4f} m")
print(
    "予測計画で衝突を含んだ回数:",
    np.count_nonzero(np.asarray(prediction_clearances) <= 0.0),
)

# 4. 実際の移動経路と最後の予測計画を描画
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    actual_states[:, 0],
    actual_states[:, 1],
    color="navy",
    linewidth=2.5,
    label="Executed trajectory",
)

ax.plot(
    updated_trajectory[:, 0],
    updated_trajectory[:, 1],
    color="teal",
    linestyle="--",
    alpha=0.8,
    label="Last prediction (before final action)",
)

ax.scatter(
    actual_states[0, 0],
    actual_states[0, 1],
    color="green",
    label="Start",
    zorder=5,
)
ax.scatter(
    actual_states[-1, 0],
    actual_states[-1, 1],
    color="darkorange",
    label="End",
    zorder=5,
)
ax.scatter(
    goal[0],
    goal[1],
    marker="*",
    s=150,
    color="purple",
    label="Goal",
    zorder=5,
)

ax.add_patch(plt.Circle(
    obstacle[:2],
    obstacle[2],
    color="crimson",
    alpha=0.35,
    label="Obstacle",
))

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Receding horizon control")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

output_path = Path(__file__).resolve().parent / "step4_receding.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("グラフ保存:", output_path)