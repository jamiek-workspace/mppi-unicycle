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

## Gazebo Harmonic 실험

Gazebo에서 LIMO를 월드 원점 (x=0, y=0, yaw=0)에 스폰하고 재생한다.
각 터미널에서 ROS 2 Jazzy 환경을 source하고 저장소 루트에서 실행한다.
기존 LIMO `/cmd_vel`, `/model/limo_diff/odometry` 및 `/clock` 브리지가 필요하다.
`empty`는 실제 월드 이름으로 바꾼다.

### B · 정적 다중 장애물

반경 0.4 m의 원통 두 개를 (2.0, +0.5), (3.5, −0.5)에 고정한다.
노드는 이 좌표를 사용하므로 원통 odometry 브리지는 필요 없다.

```bash
python3 ros2_limo/spawn_dynamic_obstacles.py --world empty --scenario B
python3 ros2_limo/single_mppi_node.py --ros-args \
  -p use_sim_time:=true -p obstacle_scenario:=B
```

### CORRIDOR · 좁아지는 통로

정적 원통 두 개와 중앙 쪽으로 0.5 m/s로 이동하는 원통 두 개를 배치한다.
정지 경계와 롤아웃의 제한된 위치 예측을 2D 통로 코드와 공유한다.
지원 폭은 0.30, 0.20, 0.15, 0.10 m이며 두 명령에 같은 폭을 지정한다.
원통이 서로 겹치는 2D 배치를 재현하기 위해 통로 원통은 시각 모델로
스폰하며 로봇 중심과 원통 반경으로 충돌을 판정한다.

```bash
# 터미널 1 — 스폰 후 두 이동 원통의 브리지 유지
python3 ros2_limo/spawn_dynamic_obstacles.py --world empty --scenario CORRIDOR --gap 0.2

# 터미널 2
python3 ros2_limo/single_mppi_node.py --ros-args \
  -p use_sim_time:=true -p obstacle_scenario:=CORRIDOR -p corridor_gap:=0.2
```

두 시나리오의 MPPI는 2D 설정을 사용한다. `/mppi/clearance_m`은
로봇 중심에서 가장 가까운 원통 표면까지의 부호 있는 거리다.
Gazebo의 제어 및 odometry 타이밍에 따라 2D 결과와 차이가 날 수 있다.
별도 장애물 옵션 없이 실행하면 기존 단일 정적 장애물 모드를 사용한다.

Reset 이후에는 기존 노드를 종료하고 LIMO와 원통이 올바른 초기 위치에
있는지 확인한다. 원통을 새로 스폰하기 전에 이전 모델을 제거한다.
통로 스폰 스크립트를 다시 실행할 때는 이전 장애물 브리지도 종료한다.
시작 조건이 충족되지 않으면 노드가 기다리는 토픽을 로그에 표시한다.
