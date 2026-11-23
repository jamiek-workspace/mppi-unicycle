"""Spawn the selected cylinders and run their Gazebo Transport bridge."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dynamic_obstacles import EXPERIMENTS


def obstacle_topic(index, suffix):
    return f"/model/mppi_obstacle_{index}/{suffix}"


def cylinder_sdf(index, radius):
    geometry = f"<geometry><cylinder><radius>{radius}</radius><length>0.6</length></cylinder></geometry>"
    return f"""<sdf version="1.9">
      <model name="mppi_obstacle_{index}">
        <link name="body">
          <gravity>false</gravity>
          <collision name="body">{geometry}</collision>
          <visual name="body">{geometry}
            <material><ambient>0.85 0.28 0.1 1</ambient></material>
          </visual>
        </link>
        <plugin filename="gz-sim-velocity-control-system"
                name="gz::sim::systems::VelocityControl"/>
        <plugin filename="gz-sim-odometry-publisher-system"
                name="gz::sim::systems::OdometryPublisher">
          <odom_frame>world</odom_frame>
          <odom_publish_frequency>50</odom_publish_frequency>
        </plugin>
      </model>
    </sdf>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="empty")
    parser.add_argument("--scenario", choices=EXPERIMENTS, default="A")
    args = parser.parse_args()

    bridge = ["ros2", "run", "ros_gz_bridge", "parameter_bridge"]
    for index, (x, y, radius) in enumerate(EXPERIMENTS[args.scenario]["obstacles"]):
        subprocess.run(
            ["ros2", "run", "ros_gz_sim", "create", "-world", args.world,
             "-string", cylinder_sdf(index, radius),
             "-x", str(x), "-y", str(y), "-z", "0.3"],
            check=True,
        )
        bridge.extend([
            obstacle_topic(index, "cmd_vel") + "@geometry_msgs/msg/Twist]gz.msgs.Twist",
            obstacle_topic(index, "odometry") + "@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ])
    os.execvp(bridge[0], bridge)


if __name__ == "__main__":
    main()
