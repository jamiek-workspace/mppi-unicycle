from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def unicycle_step(state, control, dt):
    """Forward Euler로 한 단계 뒤의 상태를 계산한다."""
    x, y, theta = state
    v, omega = control

    next_state = np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt,
    ])

    return next_state


# 1. 시뮬레이션 설정
dt = 0.01
num_steps = 1000
initial_state = np.array([0.0, 0.0, 0.0])
control = np.array([1.0, 0.5])

# 초기 상태까지 포함하므로 행이 num_steps + 1개 필요하다.
states = np.zeros((num_steps + 1, 3))
states[0] = initial_state

# 2. 상태 업데이트와 기록
for k in range(num_steps):
    states[k + 1] = unicycle_step(states[k], control, dt)

print("첫 단계 상태:", states[1])
print("최종 상태:", states[-1])

# 3. 궤적 시각화
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(states[:, 0], states[:, 1], label="Euler trajectory")
ax.scatter(states[0, 0], states[0, 1], color="green", label="Start")
ax.scatter(states[-1, 0], states[-1, 1], color="red", label="End")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Unicycle: v = 1.0 m/s, omega = 0.5 rad/s")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

output_path = Path(__file__).resolve().parent / "step1_trajectory.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("그래프 저장:", output_path)

total_time = num_steps * dt
v, omega = control

# 초기 상태가 [0, 0, 0]이고 일정한 omega != 0인 이번 실험의 정확해
exact_position = np.array([
    (v / omega) * np.sin(omega * total_time),
    (v / omega) * (1 - np.cos(omega * total_time)),
])

position_error = np.linalg.norm(states[-1, :2] - exact_position)

print("정확한 최종 위치:", exact_position)
print(f"최종 위치 오차: {position_error:.6f} m")