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
H = 100                              # 예측 단계 수
K = 200                              # 후보 개수
initial_state = np.array([0.0, 0.0, 0.0])

# 같은 설정에서 같은 난수를 얻기 위한 시드
rng = np.random.default_rng(42)

# 2. 명목 제어 입력 시퀀스: 모든 단계에서 직진
nominal_controls = np.zeros((H, 2))
nominal_controls[:, 0] = 1.0          # v = 1.0 m/s

# 3. 가우시안 노이즈와 후보 입력 생성
noise_std = np.array([0.2, 0.8])      # 표준편차: [sigma_v, sigma_omega]

noise = rng.normal(
    loc=0.0,
    scale=noise_std,
    size=(K, H, 2),
)

sampled_controls = nominal_controls[None, :, :] + noise

# 4. 각 후보의 궤적 예측
trajectories = np.zeros((K, H + 1, 3))
trajectories[:, 0, :] = initial_state

for i in range(K):
    for t in range(H):
        trajectories[i, t + 1] = unicycle_step(
            trajectories[i, t],
            sampled_controls[i, t],
            dt,
        )

# 5. 비교용 명목 궤적 계산
nominal_trajectory = np.zeros((H + 1, 3))
nominal_trajectory[0] = initial_state

for t in range(H):
    nominal_trajectory[t + 1] = unicycle_step(
        nominal_trajectory[t],
        nominal_controls[t],
        dt,
    )

# 6. 검증용 출력
print("예측 시간 [s]:", H * dt)
print("명목 입력 크기:", nominal_controls.shape)
print("후보 입력 크기:", sampled_controls.shape)
print("후보 궤적 크기:", trajectories.shape)
print("명목 궤적 최종 상태:", nominal_trajectory[-1])

print("노이즈 평균 [v, omega]:", noise.mean(axis=(0, 1)))
print("노이즈 표준편차 [v, omega]:", noise.std(axis=(0, 1)))

# 목표점과 원형 장애물: [center_x, center_y, radius]
goal = np.array([5.0, 0.0])
obstacle = np.array([2.5, 0.0, 0.5])

costs = np.zeros(K)
min_clearances = np.zeros(K)

for i in range(K):
    costs[i], _, min_clearances[i] = trajectory_cost(
        trajectories[i],
        sampled_controls[i],
        goal,
        obstacle,
        dt,
    )

# 생성된 후보 중 총비용이 가장 작은 후보
best_index = np.argmin(costs)

best_cost, best_parts, best_clearance = trajectory_cost(
    trajectories[best_index],
    sampled_controls[best_index],
    goal,
    obstacle,
    dt,
)

nominal_cost, nominal_parts, nominal_clearance = trajectory_cost(
    nominal_trajectory,
    nominal_controls,
    goal,
    obstacle,
    dt,
)

print("\n명목 궤적 총비용:", nominal_cost)
print("명목 궤적 항별 비용:", nominal_parts)
print(f"명목 궤적 최소 여유 거리: {nominal_clearance:.4f} m")

print("\n최저 비용 후보 번호:", best_index)
print("최저 후보 비용:", best_cost)
print("최저 후보 항별 비용:", best_parts)
print(f"최저 후보 최소 여유 거리: {best_clearance:.4f} m")

print(
    "검사한 상태에서 장애물과 겹치지 않은 후보 수:",
    np.count_nonzero(min_clearances > 0.0),
    "/",
    K,
)
print("후보 비용이 모두 유한한가?:", np.all(np.isfinite(costs)))

# 7. 시각화
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(K):
    ax.plot(
        trajectories[i, :, 0],
        trajectories[i, :, 1],
        color="cornflowerblue",
        alpha=0.15,
        linewidth=0.8,
        label="Sampled predictions" if i == 0 else None,
    )

ax.plot(
    nominal_trajectory[:, 0],
    nominal_trajectory[:, 1],
    color="navy",
    linewidth=2.0,
    label="Nominal prediction",
)

ax.scatter(
    initial_state[0],
    initial_state[1],
    color="green",
    s=50,
    zorder=5,
    label="Start",
)

ax.plot(
    trajectories[best_index, :, 0],
    trajectories[best_index, :, 1],
    color="darkorange",
    linewidth=2.5,
    label="Lowest-cost sample",
)

ax.scatter(
    goal[0],
    goal[1],
    color="purple",
    marker="*",
    s=150,
    zorder=6,
    label="Goal",
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
ax.set_title("Candidate evaluation with goal and obstacle costs")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

output_path = Path(__file__).resolve().parent / "step3_cost.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("그래프 저장:", output_path)