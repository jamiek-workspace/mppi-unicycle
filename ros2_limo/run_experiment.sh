#!/usr/bin/env bash
# Start and repeat a Gazebo LIMO trial from one terminal.
set -eo pipefail

case "${1:-B}" in
  B|b) scenario=B ;;
  CORRIDOR|corridor) scenario=CORRIDOR ;;
  *) echo "Usage: bash ros2_limo/run_experiment.sh [B|CORRIDOR] [gap]" >&2; exit 2 ;;
esac
gap=${2:-0.2}
if [[ $scenario == CORRIDOR && ! $gap =~ ^0\.(30?|20?|15|10?)$ ]]; then
  echo "Corridor gap must be 0.30, 0.20, 0.15 or 0.10." >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /opt/ros/jazzy/setup.bash
model=${LIMO_MODEL_SDF:-"$HOME/limo_pro_ws/sim_harmonic/models/limo_diff/model.sdf"}
if [[ ! -r $model ]]; then
  echo "LIMO SDF not found: $model. Set LIMO_MODEL_SDF to the model.sdf path." >&2
  exit 1
fi

# Isolate this simulator from manually started Gazebo and ROS sessions.
export GZ_PARTITION="mppi-limo-$$"
export ROS_DOMAIN_ID=${MPPI_ROS_DOMAIN_ID:-42}
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

logs=$(mktemp -d "${TMPDIR:-/tmp}/mppi-limo.XXXXXX")
session_pids=()
trial_pids=()

stop_groups() {
  local pid
  for pid in "$@"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for _ in {1..20}; do
    local alive=false
    for pid in "$@"; do
      if kill -0 -- "-$pid" 2>/dev/null; then alive=true; fi
    done
    if [[ $alive == false ]]; then break; fi
    sleep 0.1
  done
  for pid in "$@"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}

cleanup() {
  stop_groups "${trial_pids[@]}" "${session_pids[@]}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_trial_process() {
  local logfile=$1
  shift
  setsid "$@" > "$logfile" 2>&1 &
  trial_pids+=("$!")
}

wait_for_topic() {
  local topic=$1 message_type=$2
  if ! timeout 12s ros2 topic echo "$topic" "$message_type" --once --qos-reliability best_effort >/dev/null 2>&1; then
    echo "No messages received on $topic. See $trial_logs." >&2
    return 1
  fi
}

prepare_trial() {
  start_trial_process "$trial_logs/gazebo-server.log" gz sim -s -r -v 4 empty.sdf
  local server_pid=${trial_pids[0]}
  local ready=false
  for _ in {1..20}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    local services
    services=$(timeout 2s gz service -l) || services=""
    if [[ $'\n'$services$'\n' == *$'\n/world/empty/create\n'* ]]; then
      ready=true
      break
    fi
    sleep 0.25
  done
  if [[ $ready == false ]]; then
    echo "Gazebo world empty did not become ready. See $trial_logs/gazebo-server.log." >&2
    return 1
  fi

  if ! timeout 15s ros2 run ros_gz_sim create -world empty -name limo_diff \
      -file "$model" -x 0 -y 0 -z 0.15 -Y 0 > "$trial_logs/limo-spawn.log" 2>&1; then
    echo "LIMO spawn failed. See $trial_logs/limo-spawn.log." >&2
    return 1
  fi
  start_trial_process "$trial_logs/bridge.log" ros2 run ros_gz_bridge parameter_bridge \
    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist' \
    '/model/limo_diff/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
  start_trial_process "$trial_logs/gazebo-gui.log" gz sim -g -v 4

  local -a spawn=(python3 ros2_limo/spawn_dynamic_obstacles.py --world empty --scenario "$scenario")
  if [[ $scenario == CORRIDOR ]]; then
    start_trial_process "$trial_logs/obstacles.log" "${spawn[@]}" --gap "$gap"
  elif ! timeout 25s "${spawn[@]}" > "$trial_logs/obstacles.log" 2>&1; then
    echo "Obstacle spawn failed. See $trial_logs/obstacles.log." >&2
    return 1
  fi

  wait_for_topic /clock rosgraph_msgs/msg/Clock || return 1
  wait_for_topic /model/limo_diff/odometry nav_msgs/msg/Odometry || return 1
  if [[ $scenario == CORRIDOR ]]; then
    wait_for_topic /model/mppi_obstacle_0/odometry nav_msgs/msg/Odometry || return 1
    wait_for_topic /model/mppi_obstacle_1/odometry nav_msgs/msg/Odometry || return 1
  fi
}

rviz_args=()
if [[ -n ${MPPI_RVIZ_CONFIG:-} ]]; then
  rviz_args=(-d "$MPPI_RVIZ_CONFIG")
fi
setsid rviz2 "${rviz_args[@]}" --ros-args -p use_sim_time:=true > "$logs/rviz.log" 2>&1 &
session_pids+=("$!")
setsid ros2 run plotjuggler plotjuggler > "$logs/plotjuggler.log" 2>&1 &
session_pids+=("$!")

echo "Logs: $logs"
trial=0
while true; do
  trial=$((trial + 1))
  trial_logs="$logs/trial-$trial"
  mkdir "$trial_logs"
  echo "Starting $scenario (trial $trial)"
  if prepare_trial; then
    setsid python3 -u ros2_limo/single_mppi_node.py --ros-args \
      -p use_sim_time:=true -p obstacle_scenario:="$scenario" -p corridor_gap:="$gap" \
      > >(tee "$trial_logs/mppi.log") 2>&1 &
    trial_pids+=("$!")
  fi
  if ! read -r -p "Enter: fresh trial | q then Enter: quit > " repeat || [[ $repeat == q ]]; then
    break
  fi
  stop_groups "${trial_pids[@]}"
  trial_pids=()
done
