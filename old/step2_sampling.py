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

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Control sampling: K={K}, H={H}, dt={dt} s")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

output_path = Path(__file__).resolve().parent / "step2_sampling.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("그래프 저장:", output_path)