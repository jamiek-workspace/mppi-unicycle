"""Spawn a copy of the LIMO SDF with a simulator-world odometry publisher."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def add_world_odometry(root):
    models = root.findall("model")
    if len(models) != 1:
        raise ValueError("Expected a single top-level LIMO model in the SDF")
    plugin = ET.SubElement(
        models[0], "plugin",
        filename="gz-sim-odometry-publisher-system",
        name="gz::sim::systems::OdometryPublisher",
    )
    for key, value in {
        "dimensions": "3",
        "odom_frame": "world",
        "odom_topic": "/mppi/gazebo_odometry",
        "odom_publish_frequency": "50",
    }.items():
        ET.SubElement(plugin, key).text = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    source = args.model.resolve()
    tree = ET.parse(source)
    add_world_odometry(tree.getroot())

    # Keep relative mesh and include paths rooted at the source SDF directory.
    with tempfile.NamedTemporaryFile(suffix=".sdf", prefix=".mppi-limo-", dir=source.parent) as copy:
        tree.write(copy, encoding="utf-8", xml_declaration=True)
        copy.flush()
        subprocess.run(
            ["ros2", "run", "ros_gz_sim", "create", "-world", "empty",
             "-name", "limo_diff", "-file", copy.name, "-x", "0", "-y", "0",
             "-z", "0.15", "-Y", "0"],
            check=True,
        )

        # An accepted create request can still be queued in Gazebo. Keep the
        # SDF available until the model is present in the simulation.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                models = subprocess.run(
                    ["gz", "model", "--list"], capture_output=True,
                    text=True, timeout=2.0,
                )
            except subprocess.TimeoutExpired:
                continue
            if models.returncode == 0 and any(
                line.strip() == "- limo_diff" for line in models.stdout.splitlines()
            ):
                print("Gazebo registered limo_diff.", flush=True)
                return
            time.sleep(0.2)
        raise SystemExit("Gazebo accepted the LIMO spawn request but never listed limo_diff.")


if __name__ == "__main__":
    main()
