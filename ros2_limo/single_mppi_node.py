
import sys
import time
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy

# 기존 MPPI 코드 가져오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as base_cfg
from core.controller import MPPIController
from core.dynamics import step_batch


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
        self.odom_sub = self.create_subscription(
            Odometry,
            "/model/limo_diff/odometry",
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
        self.actual_path.header.frame_id = "odom"

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

        self.cfg.dt = 0.1
        self.cfg.v_max = 0.2
        self.cfg.omega_max = 0.5
        self.cfg.noise_std = [0.08, 0.3]
        self.cfg.goal_tolerance = 0.15
        self.cfg.obs_dynamic = False

        self.state = None
        self.last_odom_time = None
        self.controller = None
        self.goal = None
        self.started_at = None
        self.finished = False
        self.step_count = 0

        self.timer = self.create_timer(
            self.cfg.dt, self.control_step
        )

        self.get_logger().info(
            "Single MPPI LIMO node ready. Waiting for odometry."
        )

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
        pose.header.frame_id = "odom"
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
        goal.header.frame_id = "odom"
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
        circle.header.frame_id = "odom"
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
        path.header.frame_id = "odom"
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
            cost_min = float(np.min(finite_costs))
            cost_max = float(np.max(finite_costs))
        else:
            cost_min = 0.0
            cost_max = 1.0

        cost_range = max(cost_max - cost_min, 1e-9)

        marker_array = MarkerArray()

        for marker_id, sample_idx in enumerate(indices):
            marker = Marker()
            marker.header.frame_id = "odom"
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
        delete_marker.header.frame_id = "odom"
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
        marker.header.frame_id = "odom"
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

    def finish(self, reason):
        if not self.finished:
            self.finished = True
            self.stop_robot()
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
            self.finish("Odometry timeout. Robot stopped.")
            return

        # 초기 위치 기준 목표점 설정
        if self.controller is None:
            x, y, yaw = self.state

            # 로봇 초기 자세 기준:
            # 전방 1.0m, 왼쪽 0.7m
            forward = 1.0
            left = 0.7

            self.goal = np.array([
                x + forward * math.cos(yaw) - left * math.sin(yaw),
                y + forward * math.sin(yaw) + left * math.cos(yaw)
            ])

            self.cfg.goal = self.goal.tolist()

            # Gazebo 정적 원기둥 장애물
            # 실제 반경 0.20m + LIMO 근사 반경 0.17m
            self.cfg.obstacle = [
                0.50,
                0.35,
                0.37
            ]

            self.get_logger().info(
                f"Obstacle: ({self.cfg.obstacle[0]:.2f}, "
                f"{self.cfg.obstacle[1]:.2f}), "
                f"effective radius={self.cfg.obstacle[2]:.2f}m"
            )

            self.controller = MPPIController(self.cfg)
            self.publish_goal_markers()

            # 초기 제어 시퀀스 제한
            self.controller.controls[:, 0] = 0.1
            self.controller.controls[:, 1] = 0.0

            self.started_at = now

            self.get_logger().info(
                f"Start: ({x:.2f}, {y:.2f}) "
                f"Goal: ({self.goal[0]:.2f}, "
                f"{self.goal[1]:.2f})"
            )
            return

        if now - self.started_at > 30.0:
            self.finish("Maximum test duration reached.")
            return

        distance = np.linalg.norm(
            self.state[:2] - self.goal
        )

        if distance <= self.cfg.goal_tolerance:
            self.finish(
                f"Goal reached! Distance: {distance:.3f} m"
            )
            return

        start_compute = time.monotonic()

        try:
            action = self.controller.command(
                self.state.copy()
            )
        except Exception as exc:
            self.finish(f"MPPI error: {exc}")
            return

        compute_time = time.monotonic() - start_compute

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
            )

    def destroy_node(self):
        self.stop_robot()
        return super().destroy_node()


def main():
    rclpy.init()
    node = SingleMPPILimo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
