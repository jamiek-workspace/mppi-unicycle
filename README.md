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

## Gazebo Harmonic에서 동적 장애물 비교

ROS 2 Jazzy 환경에서 기존 LIMO 모델과 `/cmd_vel`,
`/model/limo_diff/odometry` 브리지를 사용한다. 정적 원기둥이 없는
Gazebo 월드를 열고 LIMO를 월드 원점 `(x=0, y=0, yaw=0)`에 스폰한다.
실험에서는 LIMO odom의 XY 원점과 방향이 Gazebo 월드와 일치해야 한다.
실험 목표는 2D와 동일하게 `(5.0, 0.0)`이다.

| 시나리오 | 장애물 움직임 |
| --- | --- |
| A | 원기둥 1개가 경로를 횡단 |
| B | 원기둥 2개가 반대 방향으로 이동 |
| C | 원기둥 1개가 이동 도중 방향을 반전 |

각 터미널에서 ROS 2 환경을 source한 뒤, 저장소 루트에서 실행한다.
`empty`는 실행 중인 Gazebo의 월드 이름으로 바꿀 수 있다.
이미 `/clock`을 브리지 중이면 첫 번째 명령을 생략한다.
Gazebo에서 시뮬레이션을 재생한 상태로 노드를 실행한다.

```bash
# 터미널 1 — 시뮬레이션 시간
ros2 run ros_gz_bridge parameter_bridge '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'

# 터미널 2 — 원기둥 스폰 및 장애물 odometry / cmd_vel 브리지
python3 ros2_limo/spawn_dynamic_obstacles.py --world empty --scenario A

# 터미널 3 — LIMO 제어
python3 ros2_limo/single_mppi_node.py --ros-args \
  -p use_sim_time:=true -p obstacle_scenario:=A \
  -p predict_obstacles:=true -p seed:=42 -p obstacle_speed_scale:=0.15
```

비교 실행은 월드를 초기 상태에서 다시 시작한 뒤 같은 시나리오, 시드,
속도 배율로 실행한다. `predict_obstacles:=false`는 매 제어 주기에 최신
장애물 위치를 사용하되, 롤아웃 전체에 그 위치를 유지한다.
`true`는 Gazebo odometry로 관측한 속도로 롤아웃 중 장애물 위치를 예측한다.
두 모드에서 장애물의 이동 명령은 동일하다. 별도 파라미터 없는 기존
노드 실행에서는 정적 장애물 시나리오가 사용된다.

원기둥 중심과 반경은 2D와 동일하다. 기본 속도 배율 `0.15`는 LIMO
속도 제한(0.2 m/s)에 맞춰 Gazebo에서 횡단을 관찰하기 위한 값이다.
C의 방향 반전 시각도 같은 배율로 늘어나 기본 약 6.67 시뮬레이션 초에
반전한다. `obstacle_speed_scale:=1.0`에서 2D의 속도와 반전 시각을
사용한다. Gazebo와 2D는 제어 주기와 로봇 속도 제한이 달라서 결과를 비교할
때 사용한 설정을 함께 기록한다.

`/mppi/clearance_m`은 가장 가까운 장애물의 표면과 LIMO 근사 원형
footprint(반경 0.17 m) 사이 여유 거리다. 노드는 목표 도달, 추정
footprint 겹침, 시간 초과 등 종료 사유와 최소 여유 거리, 경과 시간을
기록한다. 물리 접촉 여부는 Gazebo에서 별도로 확인한다.
기존 RViz 경로와 `/mppi/compute_time_ms`도 함께 비교할 수 있다.
