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
obstacles      = [
    [1.8, 0.0, 0.4],   # 배치 2로 복원
    [3.2, 0.0, 0.4],
]
initial_state  = [0.0, 0.0, 0.0]
goal_tolerance = 0.2

# ── 시뮬레이션
max_steps = 300
seed      = 42
