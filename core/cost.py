import numpy as np

def running_cost(positions, goal, obstacles, cfg):
    goal_cost = cfg.q_goal * np.sum(
        (positions - goal) ** 2, axis=2,
    )                                       # shape (K, H)

    obstacle_cost  = np.zeros_like(goal_cost)
    collision_cost = np.zeros_like(goal_cost)

    for obs in obstacles:
        center    = np.array(obs[:2])
        radius    = obs[2]
        clearance = np.linalg.norm(
            positions - center, axis=2,
        ) - radius                          # shape (K, H)
        proximity = np.maximum(
            0.0,
            (cfg.margin - clearance) / cfg.margin,
        ) ** 2
        obstacle_cost  += cfg.q_obstacle * proximity
        collision_cost += cfg.q_collision * (clearance <= 0.0)

    return goal_cost + obstacle_cost + collision_cost

def control_cost(controls, noise, cfg):
    u    = controls[None, :, :]
    du   = noise
    R    = np.array([cfg.R_v, cfg.R_omega])  # shape (2,)
    term1 = (1 - 1 / cfg.lam) / 2 * np.sum(du * R * du, axis=2)
    term2 = np.sum(u * R * du, axis=2)
    term3 = 0.5 * np.sum(u * R * u, axis=2)
    return term1 + term2 + term3

def terminal_cost(final_states, goal, cfg):
    return cfg.q_terminal * np.sum(
        (final_states[:, :2] - goal) ** 2, axis=1,
    )

def compute_costs(trajectories, controls, noise, goal, obstacles, cfg):
    positions = trajectories[:, 1:, :2]
    r_cost    = running_cost(positions, goal, obstacles, cfg)
    c_cost    = control_cost(controls, noise, cfg)
    t_cost    = terminal_cost(trajectories[:, -1, :], goal, cfg)
    return np.sum((r_cost + c_cost) * cfg.dt, axis=1) + t_cost
