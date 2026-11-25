"""Compare Gazebo and wheel odometry at the same simulation timestamps."""
import argparse
from pathlib import Path

import numpy as np
import yaml


def read_xy(path):
    rows = {}
    with path.open() as stream:
        for msg in yaml.safe_load_all(stream):
            if not msg or not {"header", "pose"} <= msg.keys():
                continue
            stamp = msg["header"]["stamp"]
            p = msg["pose"]["pose"]["position"]
            rows[stamp["sec"] + 1e-9 * stamp["nanosec"]] = (p["x"], p["y"])
    return np.array([(t, *rows[t]) for t in sorted(rows)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path)
    args = parser.parse_args()
    gazebo = read_xy(args.trial_dir / "gazebo-odometry.log")
    wheel = read_xy(args.trial_dir / "limo-odometry.log")
    if len(gazebo) < 2 or len(wheel) < 2:
        parser.error("Both odometry logs must contain at least two poses.")
    gazebo = gazebo[(gazebo[:, 0] >= wheel[0, 0]) & (gazebo[:, 0] <= wheel[-1, 0])]
    if not len(gazebo):
        parser.error("The pose timestamps do not overlap.")
    wheel_xy = np.column_stack([
        np.interp(gazebo[:, 0], wheel[:, 0], wheel[:, column])
        for column in (1, 2)
    ])
    error = np.linalg.norm(gazebo[:, 1:3] - wheel_xy, axis=1)
    print(f"XY separation (max): {error.max():.3f} m")
    crossings = np.flatnonzero((gazebo[:-1, 1] < 2.5) & (gazebo[1:, 1] >= 2.5))
    if not crossings.size:
        print("Gazebo did not cross x=2.50 m in the shared recording.")
        return
    i = crossings[0]
    window = slice(i, i + 2)
    crossing_t = np.interp(2.5, gazebo[window, 1], gazebo[window, 0])
    gazebo_y = np.interp(crossing_t, gazebo[window, 0], gazebo[window, 2])
    wheel_x, wheel_y = (np.interp(crossing_t, wheel[:, 0], wheel[:, k]) for k in (1, 2))
    print(f"At Gazebo x=2.50 m (t={crossing_t:.3f}s): "
          f"Gazebo y={gazebo_y:+.3f} m, wheel XY=({wheel_x:.3f}, {wheel_y:+.3f}) m")


if __name__ == "__main__":
    main()
