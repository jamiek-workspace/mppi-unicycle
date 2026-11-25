"""Spawn the selected cylinders and run their Gazebo Transport bridge."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dynamic_obstacles import gazebo_experiment


def obstacle_topic(index, suffix):
    return f"/model/mppi_obstacle_{index}/{suffix}"


def cylinder_sdf(index, radius, moving, physical_collision):
    geometry = f"<geometry><cylinder><radius>{radius}</radius><length>0.6</length></cylinder></geometry>"
    collision = f'<collision name="body">{geometry}</collision>' if physical_collision else ""
    plugins = """
        <plugin filename="gz-sim-velocity-control-system"
                name="gz::sim::systems::VelocityControl"/>
        <plugin filename="gz-sim-odometry-publisher-system"
                name="gz::sim::systems::OdometryPublisher">
          <odom_frame>world</odom_frame>
          <odom_publish_frequency>50</odom_publish_frequency>
        </plugin>
    """ if moving else ""
    static = "false" if moving else "true"
    return f"""<sdf version="1.9">
      <model name="mppi_obstacle_{index}">
        <static>{static}</static>
        <link name="body">
          <gravity>false</gravity>
          {collision}
          <visual name="body">{geometry}
            <material><ambient>0.85 0.28 0.1 1</ambient></material>
          </visual>
        </link>
        {plugins}
      </model>
    </sdf>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="empty")
    parser.add_argument("--scenario", type=str.upper, choices=("B", "CORRIDOR", "CORRIDOR_PHYSICAL"), default="B")
    parser.add_argument("--gap", type=float, default=0.2)
    args = parser.parse_args()
    corridor = args.scenario != "B"
    obstacles = gazebo_experiment(args.scenario, args.gap)["obstacles"]

    bridge = ["ros2", "run", "ros_gz_bridge", "parameter_bridge"]
    for index, (x, y, radius) in enumerate(obstacles):
        moving = corridor and index < 2
        subprocess.run(
            ["ros2", "run", "ros_gz_sim", "create", "-world", args.world,
             "-name", f"mppi_obstacle_{index}",
             "-string", cylinder_sdf(index, radius, moving, physical_collision=args.scenario != "CORRIDOR"),
             "-x", str(x), "-y", str(y), "-z", "0.3"],
            check=True,
        )
        if moving:
            bridge.extend([
                obstacle_topic(index, "cmd_vel") + "@geometry_msgs/msg/Twist]gz.msgs.Twist",
                obstacle_topic(index, "odometry") + "@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            ])
    if corridor:
        os.execvp(bridge[0], bridge)


if __name__ == "__main__":
    main()
