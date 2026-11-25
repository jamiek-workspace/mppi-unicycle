#!/usr/bin/env bash
# Start and repeat a Gazebo LIMO trial from one terminal.
set -eo pipefail

case "${1:-B}" in
  B|b) scenario=B ;;
  CORRIDOR|corridor) scenario=CORRIDOR ;;
  CORRIDOR_PHYSICAL|corridor_physical) scenario=CORRIDOR_PHYSICAL ;;
  *) echo "Usage: bash ros2_limo/run_experiment.sh [B|CORRIDOR|CORRIDOR_PHYSICAL] [gap]" >&2; exit 2 ;;
esac
default_gap=0.2
if [[ $scenario == CORRIDOR_PHYSICAL ]]; then default_gap=0.50; fi
gap=${2:-$default_gap}
trace_pose=${MPPI_TRACE_POSE:-0}
if [[ $scenario == CORRIDOR && ! $gap =~ ^0\.(30?|20?|15|10?)$ ]]; then
  echo "Corridor gap must be 0.30, 0.20, 0.15 or 0.10." >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /opt/ros/jazzy/setup.bash
if [[ $scenario == CORRIDOR_PHYSICAL ]]; then
  python3 -c 'import sys; from core.dynamic_obstacles import gazebo_experiment; gazebo_experiment("CORRIDOR_PHYSICAL", float(sys.argv[1]))' "$gap"
fi
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
  local world_file=empty.sdf
  if [[ $scenario == CORRIDOR ]]; then
    world_file="$PWD/ros2_limo/corridor_world.sdf"
  fi
  start_trial_process "$trial_logs/gazebo-server.log" gz sim -s -r -v 4 "$world_file"
  local server_pid=${trial_pids[0]}
  local ready=false
  local services_log="$trial_logs/gazebo-services.log"
  for _ in {1..12}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    timeout 4s gz service -l > "$services_log" 2>&1 || true
    if grep -Eq '^/world/empty/(create|create_multiple)(/blocking)?[[:space:]]*$' "$services_log"; then
      ready=true
      break
    fi
    sleep 0.25
  done
  if [[ $ready == false ]]; then
    echo "Gazebo world empty did not become ready." >&2
    echo "Server log: $trial_logs/gazebo-server.log" >&2
    echo "Service discovery log: $services_log" >&2
    return 1
  fi

  local -a limo_spawn=(ros2 run ros_gz_sim create -world empty -name limo_diff
    -file "$model" -x 0 -y 0 -z 0.15 -Y 0)
  local -a pose_bridge=()
  if [[ $scenario == CORRIDOR_PHYSICAL || $trace_pose == 1 ]]; then
    limo_spawn=(python3 ros2_limo/spawn_limo_world_pose.py "$model")
    pose_bridge=('/mppi/gazebo_odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry')
  fi
  if ! timeout 15s "${limo_spawn[@]}" > "$trial_logs/limo-spawn.log" 2>&1; then
    echo "LIMO spawn failed. See $trial_logs/limo-spawn.log." >&2
    return 1
  fi
  start_trial_process "$trial_logs/bridge.log" ros2 run ros_gz_bridge parameter_bridge \
    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist' \
    '/model/limo_diff/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' "${pose_bridge[@]}"

  local -a spawn=(python3 ros2_limo/spawn_dynamic_obstacles.py --world empty --scenario "$scenario")
  if [[ $scenario != B ]]; then
    start_trial_process "$trial_logs/obstacles.log" "${spawn[@]}" --gap "$gap"
  elif ! timeout 25s "${spawn[@]}" > "$trial_logs/obstacles.log" 2>&1; then
    echo "Obstacle spawn failed. See $trial_logs/obstacles.log." >&2
    return 1
  fi

  local -a expected_models=(limo_diff mppi_obstacle_0 mppi_obstacle_1)
  if [[ $scenario != B ]]; then
    expected_models+=(mppi_obstacle_2 mppi_obstacle_3)
  fi
  local model_log="$trial_logs/gazebo-models.log"
  local -a missing_models
  local name
  local model_deadline=$((SECONDS + 25))
  while true; do
    missing_models=()
    timeout 4s gz model --list > "$model_log" 2>&1 || true
    for name in "${expected_models[@]}"; do
      if ! grep -Eq "^[[:space:]]*-[[:space:]]+$name[[:space:]]*$" "$model_log"; then
        missing_models+=("$name")
      fi
    done
    if (( ${#missing_models[@]} == 0 || SECONDS >= model_deadline )); then break; fi
    sleep 0.25
  done
  if (( ${#missing_models[@]} > 0 )); then
    echo "Missing Gazebo models: ${missing_models[*]}. See $model_log and $trial_logs/obstacles.log." >&2
    return 1
  fi

  echo "Gazebo confirmed models: ${expected_models[*]}"
  start_trial_process "$trial_logs/gazebo-gui.log" gz sim -g -v 4
  wait_for_topic /clock rosgraph_msgs/msg/Clock || return 1
  wait_for_topic /model/limo_diff/odometry nav_msgs/msg/Odometry || return 1
  if [[ $scenario == CORRIDOR_PHYSICAL || $trace_pose == 1 ]]; then
    wait_for_topic /mppi/gazebo_odometry nav_msgs/msg/Odometry || return 1
  fi
  if [[ $scenario != B ]]; then
    wait_for_topic /model/mppi_obstacle_0/odometry nav_msgs/msg/Odometry || return 1
    wait_for_topic /model/mppi_obstacle_1/odometry nav_msgs/msg/Odometry || return 1
  fi
}

rviz_args=()
if [[ -n ${MPPI_RVIZ_CONFIG:-} ]]; then
  rviz_args=(-d "$MPPI_RVIZ_CONFIG")
fi
if [[ $scenario == CORRIDOR_PHYSICAL ]]; then
  rviz_config=${MPPI_RVIZ_CONFIG:-"$PWD/rviz/single_mppi.rviz"}
  sed -E 's/^([[:space:]]*Fixed Frame:).*/\1 world/' "$rviz_config" > "$logs/physical-corridor.rviz"
  rviz_args=(-d "$logs/physical-corridor.rviz")
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
    if ! read -r -p "Gazebo view ready? Enter: start MPPI | q then Enter: quit > " start || [[ $start == q ]]; then
      break
    fi
    if [[ $trace_pose == 1 ]]; then
      start_trial_process "$trial_logs/gazebo-odometry.log" \
        env PYTHONUNBUFFERED=1 ros2 topic echo /mppi/gazebo_odometry nav_msgs/msg/Odometry \
        --qos-reliability best_effort
      start_trial_process "$trial_logs/limo-odometry.log" \
        env PYTHONUNBUFFERED=1 ros2 topic echo /model/limo_diff/odometry nav_msgs/msg/Odometry \
        --qos-reliability best_effort
      echo "Pose traces: $trial_logs/gazebo-odometry.log and $trial_logs/limo-odometry.log"
    fi
    speed_args=()
    if [[ $scenario == B ]]; then
      speed_args=(-p b_v_max:="${MPPI_B_V_MAX:-0.30}" -p b_omega_max:="${MPPI_B_OMEGA_MAX:-0.50}")
    elif [[ $scenario == CORRIDOR_PHYSICAL ]]; then
      speed_args=(-p pose_topic:=/mppi/gazebo_odometry -p path_frame:=world)
    fi
    setsid python3 -u ros2_limo/single_mppi_node.py --ros-args \
      -p use_sim_time:=true -p obstacle_scenario:="$scenario" -p corridor_gap:="$gap" "${speed_args[@]}" \
      > >(tee "$trial_logs/mppi.log") 2>&1 &
    trial_pids+=("$!")
  fi
  if ! read -r -p "Enter: fresh trial | q then Enter: quit > " repeat || [[ $repeat == q ]]; then
    break
  fi
  stop_groups "${trial_pids[@]}"
  trial_pids=()
done
