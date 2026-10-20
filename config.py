# ── 알고리즘 구조
K   = 1000
H   = 50
dt  = 0.02

# ── MPPI 핵심 파라미터
lam       = 1.0
R         = 1.0
noise_std = [0.2, 0.8]

# ── 비용 함수
q_goal      = 1.0
q_terminal  = 10.0
q_obstacle  = 100.0
q_collision = 10000.0
margin      = 0.3

# ── 환경
goal           = [5.0, 0.0]
obstacle       = [2.5, 0.0, 0.5]
initial_state  = [0.0, 0.0, 0.0]
goal_tolerance = 0.2

# ── 시뮬레이션
max_steps = 300
seed      = 42
