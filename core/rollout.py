import numpy as np
from core.dynamics import step_batch

def rollout(state, controls, noise, dt,
            v_min=0.0, v_max=1.5, omega_max=2.0):
    K, H, _ = noise.shape
    trajectories = np.zeros((K, H + 1, 3))
    trajectories[:, 0, :] = state
    perturbed_controls = controls[None, :, :] + noise

    # 예측도 실제와 같은 제한 적용
    perturbed_controls[:, :, 0] = np.clip(
        perturbed_controls[:, :, 0], v_min, v_max,
    )
    perturbed_controls[:, :, 1] = np.clip(
        perturbed_controls[:, :, 1], -omega_max, omega_max,
    )

    for i in range(H):
        trajectories[:, i + 1, :] = step_batch(
            trajectories[:, i, :],
            perturbed_controls[:, i, :],
            dt,
        )

    # clip된 실효 노이즈 반환
    # clipped_controls - controls = 실제로 적용된 노이즈
    effective_noise = perturbed_controls - controls[None, :, :]

    return trajectories, effective_noise
