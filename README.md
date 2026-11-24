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
  -p predict_obstacles:=true -p seed:=42 -p obstacle_speed_scale:=1.0
```

비교 실행은 월드를 초기 상태에서 다시 시작한 뒤 같은 시나리오, 시드,
속도 배율로 실행한다. `predict_obstacles:=false`는 매 제어 주기에 최신
장애물 위치를 사용하되, 롤아웃 전체에 그 위치를 유지한다.
`true`는 Gazebo odometry로 관측한 속도로 롤아웃 중 장애물 위치를 예측한다.
두 모드에서 장애물의 이동 명령은 동일하다. 별도 파라미터 없는 기존
노드 실행에서는 정적 장애물 시나리오가 사용된다.

동적 시나리오는 2D와 같은 초기 좌표, 원기둥 반경, 장애물 속도,
MPPI 설정을 사용한다. 기본 속도 배율은 `1.0`이고 C는 약 1초에
반전한다. MPPI의 모델 제어 주기는 0.02초, 예측 구간은 1초,
목표 허용 오차는 0.20 m, 시뮬레이션 시간 제한은 6초다.

`/mppi/clearance_m`은 2D 실험과 동일하게 LIMO 중심에서 가장 가까운
장애물 표면까지의 부호 있는 거리다. Gazebo의 실제 차체 접촉은 이
거리와 별개다. Gazebo에서는 로봇의 물리 응답, odometry 및 명령
전달 시점에 따라 결과가 달라진다. 20 ms 시뮬레이션 제어 주기를
따라가는지는 `/mppi/compute_time_ms`와 제어 명령 간격으로 확인한다.
