# MPPI Unicycle

Model Predictive Path Integral (MPPI) control
implementation for unicycle robot.

Based on: Williams et al. 2017
"Model Predictive Path Integral Control:
From Theory to Parallel Computation"

## Structure

- `core/`
  - `dynamics.py` — Unicycle dynamics
  - `sampling.py` — Control noise sampling
  - `rollout.py` — Trajectory rollout
  - `cost.py` — Cost computation
  - `update.py` — Weighted control update
  - `controller.py` — MPPI controller
- `config.py` — Parameters
- `main.py` — Single obstacle, entry point
- `analyze.py` — Avoidance mode analysis
- `compare_multimodal.py` — Standard vs Multimodal MPPI
- `visualization.py` — Plotting
- `NOTES.md` — Parameter tuning notes

## Branches

- `main` — Single obstacle environment
- `feature/multi-obstacle` — Two obstacle environment

## Requirements

```
pip install numpy matplotlib
```

## Run

```
# 단일 장애물 실행
python main.py

# 회피 모드 분석
python analyze.py

# Multimodal MPPI 비교
python compare_multimodal.py
```

## Results

- 장애물 회피 + 목표 도달 ✅
- 파라미터 튜닝 (λ, noise_std 등) ✅
- 회피 모드 분석 (위/아래 분리) ✅
- Multimodal MPPI 첫 시도 ✅

## Future Work

- k-means 클러스터링으로 모드 자동 분류
- ROS2 Jazzy 연동
- VLA + Multimodal MPPI 연동
