# ── 알고리즘 구조
K   = 1000
H   = 50
dt  = 0.02

# ── MPPI 핵심 파라미터
lam       = 3.0
R_v       = 0.1
R_omega   = 0.05
noise_std = [0.1, 0.8]

# ── 제어 입력 제한
v_min     = 0.0
v_max     = 1.5
omega_max = 2.0

# ── 비용 함수
q_goal      = 2.0
q_terminal  = 20.0
q_obstacle  = 100.0
q_collision = 10000.0
margin      = 0.3

# ── 환경
goal           = [5.0, 0.0]
obstacle       = [2.5, 0.0, 0.5]
initial_state  = [0.0, 0.0, 0.0]
goal_tolerance = 0.2

# ── 동적 장애물 (등속도 모델)
obs_velocity = [0.0, -0.4]   # 장애물 속도 [vx, vy] m/s
obs_dynamic  = False         # False면 기존 정적 동작 그대로

# ── Multimodal MPPI (사전 샘플 분할)
multimodal_bias = 0.5        # 모드별 omega 노이즈 평균 오프셋 [rad/s]

# ── 시뮬레이션
max_steps = 300
seed      = 42
