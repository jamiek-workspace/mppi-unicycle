
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class VelocityErrorNode(Node):
    def __init__(self):
        super().__init__("velocity_error_node")

        self.last_cmd = None
        self.last_cmd_time = None

        self.create_subscription(
            Twist, "/cmd_vel", self.cmd_callback, 10
        )

        self.create_subscription(
            Odometry,
            "/model/limo_diff/odometry",
            self.odom_callback,
            10
        )

        self.linear_error_pub = self.create_publisher(
            Float32, "/mppi/linear_velocity_error", 10
        )

        self.angular_error_pub = self.create_publisher(
            Float32, "/mppi/angular_velocity_error", 10
        )

    def cmd_callback(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def odom_callback(self, msg):
        if self.last_cmd is None:
            return

        # 너무 오래된 명령은 비교하지 않음
        age = (
            self.get_clock().now() - self.last_cmd_time
        ).nanoseconds / 1e9

        if age > 0.2 or age < 0:
            return

        linear_error = (
            self.last_cmd.linear.x
            - msg.twist.twist.linear.x
        )

        angular_error = (
            self.last_cmd.angular.z
            - msg.twist.twist.angular.z
        )

        linear_msg = Float32()
        linear_msg.data = float(linear_error)

        angular_msg = Float32()
        angular_msg.data = float(angular_error)

        self.linear_error_pub.publish(linear_msg)
        self.angular_error_pub.publish(angular_msg)


def main():
    rclpy.init()
    node = VelocityErrorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
