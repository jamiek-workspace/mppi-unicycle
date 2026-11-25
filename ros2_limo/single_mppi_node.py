
import sys
import time
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.impl.implementation_singleton import rclpy_implementation
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32

# 기존 MPPI 코드 가져오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as base_cfg
from core.controller import MPPIController
from core.dynamics import step_batch
from core.dynamic_obstacles import (
    MultiObstacleController, clipped_prediction, gazebo_experiment,
)

from spawn_dynamic_obstacles import obstacle_topic

CorridorController = clipped_prediction(MultiObstacleController)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class GuidedCorridorController(CorridorController):
    lane_y = None

    def compute_costs(self, trajectories, effective_noise):
        costs = super().compute_costs(trajectories, effective_noise)
        x, y, yaw = np.moveaxis(trajectories[:, 1:, :], -1, 0)
        if self.lane_y is None:
            dx, dy = self.cfg.goal[0] - x, self.cfg.goal[1] - y
            heading_error = wrap_angle(np.arctan2(dy, dx) - yaw)
            weight = np.minimum(np.hypot(dx, dy) / 0.30, 1.0)
            tracking = 4.0 * weight * heading_error ** 2
        else:
            tracking = 4.0 * wrap_angle(yaw) ** 2 + 8.0 * (y - self.lane_y) ** 2
        return costs + self.cfg.dt * np.sum(tracking, axis=1)


def quaternion_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )

def cost_to_color(normalized_cost):
    """
    Normalized cost [0, 1] -> RGB
    Low cost: light green
    High cost: dark blue
    """
    colors = np.array([
        [0.84, 0.96, 0.60],
        [0.63, 0.87, 0.52],
        [0.30, 0.75, 0.61],
        [0.14, 0.55, 0.69],
        [0.09, 0.24, 0.58],
    ])

    t = float(np.clip(normalized_cost, 0.0, 1.0))
    pos = t * (len(colors) - 1)

    i = min(int(pos), len(colors) - 2)
    fraction = pos - i

    rgb = (
        colors[i] * (1.0 - fraction)
        + colors[i + 1] * fraction
    )

    return rgb

class SingleMPPILimo(Node):
    def __init__(self):
        super().__init__("single_mppi_limo")

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        self.compute_time_pub = self.create_publisher(
            Float32,
            "/mppi/compute_time_ms",
            10
        )

        self.pose_topic = self.declare_parameter("pose_topic", "/model/limo_diff/odometry").value
        self.path_frame = self.declare_parameter("path_frame", "odom").value
        self.odom_sub = self.create_subscription(
            Odometry,
            self.pose_topic,
            self.odom_callback,
            10
        )

        viz_qos = QoSProfile(depth=1)
        viz_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.goal_pub = self.create_publisher(
            Marker, "/mppi/goal_marker", viz_qos
        )
        self.tolerance_pub = self.create_publisher(
            Marker, "/mppi/goal_tolerance", viz_qos
        )
        self.path_pub = self.create_publisher(
            NavPath, "/mppi/actual_path", viz_qos
        )

        self.predicted_pub = self.create_publisher(
            NavPath, "/mppi/predicted_path", 10
        )
        
        self.samples_pub = self.create_publisher(
            MarkerArray, "/mppi/sample_trajectories", 10
        )

        self.particles_pub = self.create_publisher(
            Marker, "/mppi/sample_particles", 10
        )

        self.actual_path = NavPath()
        self.actual_path.header.frame_id = self.path_frame

        # 기존 설정을 복사하여 테스트용으로 변경
        self.cfg = SimpleNamespace(
            **{
                key: value
                for key, value in vars(base_cfg).items()
                if not key.startswith("_")
                and not callable(value)
                and not isinstance(value, type)
            }
        )

        scenario = self.declare_parameter("obstacle_scenario", "").value.upper()
        self.physical_corridor = scenario == "CORRIDOR_PHYSICAL"
        self.corridor = scenario in ("CORRIDOR", "CORRIDOR_PHYSICAL")
        self.static_multi = scenario == "B"
        self.footprint_radius = 0.17 if self.static_multi or self.physical_corridor else 0.0
        gap = self.declare_parameter("corridor_gap", 0.2).value
        self.experiment = gazebo_experiment(scenario, gap) if scenario else None
        predict = self.declare_parameter("predict_obstacles", True).value
        self.cfg.obs_dynamic = self.corridor and predict
        self.cfg.seed = self.declare_parameter("seed", base_cfg.seed).value
        self.speed_scale = self.declare_parameter("obstacle_speed_scale", 1.0).value
        if not math.isfinite(self.speed_scale) or self.speed_scale <= 0:
            raise ValueError("obstacle_speed_scale must be finite and positive")

        if not self.corridor or self.physical_corridor:
            self.cfg.dt = 0.1
            self.cfg.v_max = 0.2
            self.cfg.omega_max = 0.5
            self.cfg.noise_std = [0.08, 0.3]
            self.cfg.goal_tolerance = 0.15
            self.cfg.obs_dynamic = False

        if self.physical_corridor:
            self.cfg.v_max = 0.40
            self.cfg.obs_dynamic = predict

        if self.static_multi:
            self.cfg.v_max = self.declare_parameter("b_v_max", 0.30).value
            self.cfg.omega_max = self.declare_parameter("b_omega_max", 0.50).value
            for limit in (self.cfg.v_max, self.cfg.omega_max):
                if not math.isfinite(limit) or limit <= 0.0:
                    raise ValueError("B speed limits must be finite and positive")

        self.obstacle_states = {}
        self.obstacle_cmds = []
        self.stopped_obstacles = set()
        self.min_clearance = math.inf
        if self.experiment:
            self.clearance_pub = self.create_publisher(Float32, "/mppi/clearance_m", 10)
            for index in range(2 if self.corridor else 0):
                self.obstacle_cmds.append(
                    self.create_publisher(Twist, obstacle_topic(index, "cmd_vel"), 10)
                )
                self.create_subscription(
                    Odometry, obstacle_topic(index, "odometry"),
                    lambda msg, index=index: self.obstacle_callback(index, msg), 10,
                )
            self.get_logger().info(
                f"Scenario {scenario}, prediction={self.cfg.obs_dynamic}, "
                f"seed={self.cfg.seed}, v_max={self.cfg.v_max:.2f}m/s, "
                f"omega_max={self.cfg.omega_max:.2f}rad/s"
                + (f", footprint radius={self.footprint_radius:.2f}m" if self.footprint_radius else "")
                + (f", corridor gap={gap:.2f}m, speed scale={self.speed_scale:.3f}" if self.corridor else "")
            )

        self.state = None
        self.last_odom_time = None
        self.controller = None
        self.goal = None
        self.corridor_waypoints = []
        self.aligning_at_entrance = False
        self.previous_lane_xy = None
        if self.physical_corridor:
            moving, _, wall, _ = self.experiment["obstacles"]
            stop_y = self.experiment["y_limits"][0][0]
            lane_y = (stop_y + moving[2] + wall[1] - wall[2]) / 2.0
            x = moving[0]
            self.corridor_waypoints = [
                np.array([x - 1.0, lane_y]),
                np.array([x, lane_y]),
                np.array([x + 0.9, lane_y]),
            ]
            self.get_logger().info(f"Guided corridor: positive-y lane at y={lane_y:.3f}m")
        self.started_at = None
        self.finished = False
        self.step_count = 0

        self.timer = self.create_timer(
            self.cfg.dt, self.control_step
        )

        self.last_clock_check = self.get_clock().now().nanoseconds
        self.startup_timer = self.create_timer(
            2.0, self.report_startup_status, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def obstacle_callback(self, index, msg):
        pose = msg.pose.pose
        velocity = msg.twist.twist.linear
        yaw = quaternion_to_yaw(pose.orientation)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        # Odometry twist is expressed in the child frame. Spawn LIMO at
        # world (0, 0, yaw=0) so world XY and the LIMO odom XY coincide.
        vx = cos_yaw * velocity.x - sin_yaw * velocity.y
        vy = sin_yaw * velocity.x + cos_yaw * velocity.y
        radius = self.experiment["obstacles"][index][2]
        self.obstacle_states[index] = (
            [pose.position.x, pose.position.y, radius], [vx, vy], time.monotonic()
        )

    def report_startup_status(self):
        if self.controller is not None or self.finished:
            self.startup_timer.cancel()
            return
        now = time.monotonic()
        missing = []
        clock_now = self.get_clock().now().nanoseconds
        if self.get_parameter("use_sim_time").value and clock_now <= self.last_clock_check:
            missing.append("/clock (simulation time is not advancing)")
        self.last_clock_check = clock_now
        if self.last_odom_time is None or now - self.last_odom_time > 0.5:
            missing.append(self.pose_topic)
        for index in range(len(self.obstacle_cmds)):
            sample = self.obstacle_states.get(index)
            if sample is None or now - sample[2] > 0.5:
                missing.append(obstacle_topic(index, "odometry"))
        if missing:
            self.get_logger().warning("Waiting for: " + ", ".join(missing))

    def command_obstacles(self):
        velocities = self.experiment["velocity"](0)
        for index, publisher in enumerate(self.obstacle_cmds):
            position = self.obstacle_states[index][0][1]
            vy = velocities[index, 1] * self.speed_scale
            lo, hi = self.experiment["y_limits"][index]
            stop = lo if vy < 0.0 else hi
            direction = math.copysign(1.0, vy)
            remaining = direction * (stop - position)
            if index not in self.stopped_obstacles and remaining <= 0.005:
                self.stopped_obstacles.add(index)
                self.get_logger().info(
                    f"Obstacle {index} stop latched: y={position:+.3f}m, target={stop:+.3f}m"
                )
            cmd = Twist()
            if index not in self.stopped_obstacles:
                cmd.linear.y = direction * min(abs(vy), remaining / self.cfg.dt)
            publisher.publish(cmd)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.state = np.array(
            [p.x, p.y, quaternion_to_yaw(q)],
            dtype=float
        )
        self.last_odom_time = time.monotonic()

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = self.path_frame
        pose.pose = msg.pose.pose

        self.actual_path.header.stamp = msg.header.stamp
        self.actual_path.poses.append(pose)

        # RViz2 표시용 최근 5000개 위치 유지
        if len(self.actual_path.poses) > 5000:
            self.actual_path.poses.pop(0)

        self.path_pub.publish(self.actual_path)

    def publish_goal_markers(self):
        if self.goal is None:
            return

        stamp = self.get_clock().now().to_msg()

        goal = Marker()
        goal.header.frame_id = self.path_frame
        goal.header.stamp = stamp
        goal.ns = "mppi_goal"
        goal.id = 0
        goal.type = Marker.SPHERE
        goal.action = Marker.ADD

        goal.pose.position.x = float(self.goal[0])
        goal.pose.position.y = float(self.goal[1])
        goal.pose.position.z = 0.08
        goal.pose.orientation.w = 1.0

        goal.scale.x = 0.12
        goal.scale.y = 0.12
        goal.scale.z = 0.12

        goal.color.r = 0.1
        goal.color.g = 0.9
        goal.color.b = 0.2
        goal.color.a = 1.0

        self.goal_pub.publish(goal)

        circle = Marker()
        circle.header.frame_id = self.path_frame
        circle.header.stamp = stamp
        circle.ns = "mppi_tolerance"
        circle.id = 0
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD

        circle.pose.orientation.w = 1.0
        circle.scale.x = 0.025

        circle.color.r = 0.1
        circle.color.g = 0.9
        circle.color.b = 0.2
        circle.color.a = 0.55

        radius = float(self.cfg.goal_tolerance)

        for i in range(65):
            angle = 2.0 * math.pi * i / 64
            point = Point()
            point.x = float(
                self.goal[0] + radius * math.cos(angle)
            )
            point.y = float(
                self.goal[1] + radius * math.sin(angle)
            )
            point.z = 0.03
            circle.points.append(point)

        self.tolerance_pub.publish(circle)

    def publish_mppi_visualization(self, state):
        controller = self.controller
        if controller is None:
            return

        stamp = self.get_clock().now().to_msg()

        # 1. 최적화된 제어 시퀀스로 미래 상태 예측
        controls = controller.last_updated_controls
        predicted = np.zeros((self.cfg.H + 1, 3))
        predicted[0] = state

        for i in range(self.cfg.H):
            control = controls[i].copy()
            control[0] = np.clip(
                control[0], self.cfg.v_min, self.cfg.v_max
            )
            control[1] = np.clip(
                control[1],
                -self.cfg.omega_max,
                self.cfg.omega_max
            )

            predicted[i + 1] = step_batch(
                predicted[i:i + 1],
                control.reshape(1, 2),
                self.cfg.dt
            )[0]

        path = NavPath()
        path.header.frame_id = self.path_frame
        path.header.stamp = stamp

        for x, y, yaw in predicted:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
            pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
            path.poses.append(pose)

        self.predicted_pub.publish(path)

        # 2. Cost-colored sampled trajectories
        trajectories = controller.last_trajectories
        costs = np.asarray(controller.last_costs)

        # 우선 100개 표시
        count = min(100, trajectories.shape[0])

        indices = np.linspace(
            0,
            trajectories.shape[0] - 1,
            count,
            dtype=int
        )

        # 전체 샘플의 Cost 기준으로 정규화
        finite_costs = costs[np.isfinite(costs)]

        if finite_costs.size > 0:
            # 개선안: 충돌 outlier를 무시하고 95th percentile을 최대값으로 사용
            cost_min = float(np.percentile(finite_costs, 5))
            cost_max = float(np.percentile(finite_costs, 95))
        else:
            cost_min = 0.0
            cost_max = 1.0

        cost_range = max(cost_max - cost_min, 1e-9)

        marker_array = MarkerArray()

        for marker_id, sample_idx in enumerate(indices):
            marker = Marker()
            marker.header.frame_id = self.path_frame
            marker.header.stamp = stamp

            marker.ns = "mppi_samples"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0

            # 선 두께
            marker.scale.x = 0.008

            # Cost -> RGB
            cost = float(costs[sample_idx])

            if np.isfinite(cost):
                normalized_cost = (
                    (cost - cost_min) / cost_range
                )
            else:
                normalized_cost = 1.0

            r, g, b = cost_to_color(normalized_cost)

            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 0.65

            for x, y, _ in trajectories[sample_idx]:
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.025
                marker.points.append(point)

            marker_array.markers.append(marker)

        # 이전 프레임에서 남은 Marker 제거
        delete_marker = Marker()
        delete_marker.header.frame_id = self.path_frame
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.insert(0, delete_marker)

        self.samples_pub.publish(marker_array)

        # 미래 2초 시점의 모든 샘플 상태를 Particle로 표시
        future_step = min(
            round(2.0 / self.cfg.dt),
            trajectories.shape[1] - 1
        )
        particles = trajectories[:, future_step, :2]

        marker = Marker()
        marker.header.frame_id = self.path_frame
        marker.header.stamp = stamp
        marker.ns = "mppi_particles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.018
        marker.scale.y = 0.018

        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.65

        for x, y in particles:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.05
            marker.points.append(point)

        self.particles_pub.publish(marker)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        for publisher in self.obstacle_cmds:
            publisher.publish(Twist())

    def run_time(self):
        if self.experiment:
            return self.get_clock().now().nanoseconds / 1e9
        return time.monotonic()

    def finish(self, reason):
        if not self.finished:
            self.finished = True
            self.stop_robot()
            if self.experiment and self.started_at is not None:
                elapsed = self.run_time() - self.started_at
                reason += (f" | elapsed={elapsed:.2f}s | "
                           f"minimum clearance={self.min_clearance:+.3f}m")
            self.get_logger().info(reason)

    def control_step(self):
        if self.finished:
            self.stop_robot()
            return

        if self.state is None:
            return

        now = time.monotonic()

        # Odometry 수신 중단 감지
        if now - self.last_odom_time > 0.5:
            if self.controller is not None:
                self.finish("Odometry timeout. Robot stopped.")
            return

        if self.experiment and (
            len(self.obstacle_states) != len(self.obstacle_cmds)
            or any(now - sample[2] > 0.5 for sample in self.obstacle_states.values())
        ):
            if self.controller is not None:
                self.finish("Obstacle odometry timeout. Robot stopped.")
            return

        # 초기 위치 기준 목표점 설정
        if self.controller is None:
            x, y, yaw = self.state

            if self.experiment:
                self.goal = np.array(base_cfg.goal)
                if self.corridor:
                    controller_type = GuidedCorridorController if self.physical_corridor else CorridorController
                    self.controller = controller_type(self.cfg, self.experiment["y_limits"])
                else:
                    self.controller = MultiObstacleController(self.cfg)
            else:
                # 로봇 초기 자세 기준 전방 1.0m, 왼쪽 0.7m.
                self.goal = np.array([
                    x + math.cos(yaw) - 0.7 * math.sin(yaw),
                    y + math.sin(yaw) + 0.7 * math.cos(yaw),
                ])
                # Gazebo 정적 원기둥: 실제 반경 0.20m + LIMO 근사 반경 0.17m.
                self.cfg.obstacle = [0.50, 0.35, 0.37]
                self.get_logger().info("Obstacle: (0.50, 0.35), effective radius=0.37m")
                self.controller = MPPIController(self.cfg)

            self.cfg.goal = self.goal.tolist()
            self.publish_goal_markers()

            if not self.corridor or self.physical_corridor:
                self.controller.controls[:, 0] = 0.1
                self.controller.controls[:, 1] = 0.0

            self.started_at = self.run_time()

            self.get_logger().info(
                f"Start: ({x:.2f}, {y:.2f}) "
                f"Goal: ({self.goal[0]:.2f}, "
                f"{self.goal[1]:.2f})"
            )
            if not self.experiment:
                return

        elapsed = self.run_time() - self.started_at
        if elapsed < 0.0:
            self.finish("Clock reset. Restart the controller for a new run.")
            return

        if self.experiment:
            obstacles = np.array(self.experiment["obstacles"])
            velocities = np.zeros((len(obstacles), 2))
            for index in range(len(self.obstacle_cmds)):
                obstacles[index], velocities[index], _ = self.obstacle_states[index]
            obstacles[:, 2] += self.footprint_radius
            self.controller.obstacles = obstacles
            self.controller.velocities = velocities
            clearance = float(np.min(
                np.linalg.norm(self.state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2]
            ))
            self.min_clearance = min(self.min_clearance, clearance)
            self.clearance_pub.publish(Float32(data=clearance))
            if clearance <= 0.0:
                self.finish("Estimated footprint overlap." if self.footprint_radius
                            else "Robot center entered an obstacle radius.")
                return

        distance = np.linalg.norm(
            self.state[:2] - self.goal
        )

        if not self.corridor_waypoints and distance <= self.cfg.goal_tolerance:
            self.finish(
                f"Goal reached! Distance: {distance:.3f} m"
            )
            return

        time_limit = self.cfg.max_steps * self.cfg.dt if self.corridor else 30.0
        if self.physical_corridor:
            time_limit = 90.0
        elif self.static_multi:
            time_limit = 60.0
        if elapsed >= time_limit:
            self.finish("Maximum test duration reached.")
            return

        if self.physical_corridor:
            points = self.corridor_waypoints
            if len(points) == 3 and np.linalg.norm(self.state[:2] - points[0]) <= self.cfg.goal_tolerance:
                reached = points.pop(0)
                self.aligning_at_entrance = True
                self.get_logger().info(f"Corridor entrance reached: {reached.tolist()}; aligning to +x")

            if self.aligning_at_entrance:
                error = float(wrap_angle(-self.state[2]))
                self.command_obstacles()
                cmd = Twist()
                if abs(error) > math.radians(8):
                    cmd.angular.z = float(np.clip(1.5 * error, -self.cfg.omega_max, self.cfg.omega_max))
                    self.cmd_pub.publish(cmd)
                    return
                self.cmd_pub.publish(cmd)
                self.controller.controls[:, 0] = 0.1
                self.controller.controls[:, 1] = 0.0
                self.aligning_at_entrance = False
                self.previous_lane_xy = self.state[:2].copy()
                self.get_logger().info(f"Corridor aligned: heading_error={math.degrees(error):+.1f}deg")
                return

            if 0 < len(points) <= 2:
                x, y = self.state[:2]
                section_x, lane_y = points[0]
                if x >= section_x:
                    previous_x, previous_y = self.previous_lane_xy
                    section_y = y
                    if previous_x < section_x:
                        fraction = (section_x - previous_x) / (x - previous_x)
                        section_y = previous_y + fraction * (y - previous_y)
                    cross_track = section_y - lane_y
                    if abs(cross_track) > 0.15:
                        self.finish(f"Corridor section missed: x={section_x:.2f}m, lateral_error={cross_track:+.3f}m")
                        return
                    points.pop(0)
                    self.get_logger().info(
                        f"Corridor section passed: x={section_x:.2f}m, lateral_error={cross_track:+.3f}m"
                    )
                self.previous_lane_xy = self.state[:2].copy()

            # Track toward the exit continuously. The midpoint is a crossing
            # checkpoint, so the controller never brakes to reach that point.
            target = points[-1] if 0 < len(points) <= 2 else (points[0] if points else self.goal)
            self.cfg.goal = target.tolist()
            self.controller.lane_y = float(target[1]) if 0 < len(points) <= 2 else None
            dx, dy = target - self.state[:2]
            heading = math.atan2(dy, dx) if self.controller.lane_y is None else 0.0
            heading_error = float(wrap_angle(heading - self.state[2]))
            if abs(heading_error) > math.pi / 4:
                # Give the sampler a turn-in-place nominal sequence.
                turn_steps = min(self.cfg.H, math.ceil(
                    abs(heading_error) / (self.cfg.omega_max * self.cfg.dt)
                ))
                self.controller.controls[:turn_steps, 0] = 0.0
                self.controller.controls[:turn_steps, 1] = math.copysign(
                    self.cfg.omega_max, heading_error
                )

        start_compute = time.monotonic()

        try:
            action = self.controller.command(
                self.state.copy()
            )
        except Exception as exc:
            self.finish(f"MPPI error: {exc}")
            return

        compute_time = time.monotonic() - start_compute

        compute_msg = Float32()
        compute_msg.data = float(compute_time * 1000.0)
        self.compute_time_pub.publish(compute_msg)

        # 계산이 지나치게 오래 걸리면 정지
        if compute_time > 0.3:
            self.finish(
                f"MPPI computation too slow: "
                f"{compute_time:.3f} s"
            )
            return

        # 계산 도중 Odometry가 오래되었는지 재확인
        if time.monotonic() - self.last_odom_time > 0.5:
            self.finish("Stale odometry after computation.")
            return

        cmd = Twist()
        cmd.linear.x = float(
            np.clip(action[0], 0.0, self.cfg.v_max)
        )
        cmd.angular.z = float(
            np.clip(
                action[1],
                -self.cfg.omega_max,
                self.cfg.omega_max
            )
        )

        if self.corridor:
            self.command_obstacles()
        self.cmd_pub.publish(cmd)
        self.step_count += 1

        # 매 3번째 제어 주기마다 RViz2 시각화 갱신
        if self.step_count % 3 == 0:
            self.publish_mppi_visualization(self.state.copy())

        if self.step_count % 10 == 0:
            self.get_logger().info(
                f"distance={distance:.3f} m | "
                f"v={cmd.linear.x:.3f} | "
                f"omega={cmd.angular.z:.3f} | "
                f"compute={compute_time * 1000:.1f} ms"
                + (f" | active_goal=({target[0]:.2f}, {target[1]:.2f})"
                   f" | xy=({self.state[0]:.3f}, {self.state[1]:.3f})"
                   f" | active_distance={math.hypot(dx, dy):.3f}m"
                   f" | heading_error={math.degrees(heading_error):+.1f}deg"
                   + (f" | lateral_error={self.state[1] - target[1]:+.3f}m"
                      if self.controller.lane_y is not None else "")
                   if self.physical_corridor else "")
            )


def main():
    rclpy.init()
    node = SingleMPPILimo()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError:
        if rclpy.ok():
            raise
    finally:
        try:
            if rclpy.ok():
                node.stop_robot()
        except rclpy_implementation.RCLError:
            if rclpy.ok():
                raise
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
