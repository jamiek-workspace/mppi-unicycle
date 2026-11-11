
import sys
import time
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# 기존 MPPI 코드 가져오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as base_cfg
from core.controller import MPPIController


def quaternion_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


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

            # 첫 실험에서는 장애물 비용이 영향을 주지 않도록
            # 가상 장애물을 목표점에서 충분히 멀리 배치
            self.cfg.obstacle = [
                float(x + 100.0),
                float(y + 100.0),
                0.1
            ]

            self.controller = MPPIController(self.cfg)

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
