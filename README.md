# MPPI Unicycle

Model Predictive Path Integral (MPPI) control
implementation for unicycle robot.

Based on: Williams et al. 2017
"Model Predictive Path Integral Control:
From Theory to Parallel Computation"

## Structure

- `mppi/`
  - `core/`
    - `dynamics.py` — Unicycle dynamics
    - `sampling.py` — Control noise sampling
    - `rollout.py` — Trajectory rollout
    - `cost.py` — Cost computation
    - `update.py` — Weighted control update
    - `controller.py` — MPPI controller
  - `config.py` — Parameters
  - `main.py` — Entry point
  - `visualization.py` — Plotting

## Requirements
pip install numpy matplotlib

## Run
cd mppi
python3 main.py
