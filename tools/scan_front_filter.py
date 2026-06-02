#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import LaserScan


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class ScanFrontFilter(Node):
    """
    Input:
      /scan, frame_id = laser_frame_raw

    Output:
      /scan_front, frame_id = laser_frame_raw

    Geometry rule:
      - Output scan remains in the LiDAR frame.
      - We do NOT rotate the scan data itself.
      - We only invalidate rays that are not in the robot-front sector.
      - Front sector is decided after applying lidar_yaw_deg.
    """

    def __init__(self):
        super().__init__("scan_front_filter")

        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/scan_front")

        # Robot-front sector, in base_link convention.
        self.declare_parameter("min_angle_deg", -90.0)
        self.declare_parameter("max_angle_deg", 90.0)

        # LiDAR is mounted 180 deg reversed relative to base_link.
        self.declare_parameter("lidar_yaw_deg", 180.0)

        # Remove robot body / arm / too-close noise.
        self.declare_parameter("min_keep_range", 0.45)
        self.declare_parameter("max_keep_range", 6.0)

        # Fixed output scan length for slam_toolbox stability.
        self.declare_parameter("fixed_bins", 720)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.min_angle = math.radians(float(self.get_parameter("min_angle_deg").value))
        self.max_angle = math.radians(float(self.get_parameter("max_angle_deg").value))
        self.lidar_yaw = math.radians(float(self.get_parameter("lidar_yaw_deg").value))
        self.min_keep_range = float(self.get_parameter("min_keep_range").value)
        self.max_keep_range = float(self.get_parameter("max_keep_range").value)
        self.fixed_bins = int(self.get_parameter("fixed_bins").value)

        if self.fixed_bins < 2:
            raise ValueError("fixed_bins must be >= 2")

        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self.cb_scan,
            sub_qos,
        )
        self.pub = self.create_publisher(
            LaserScan,
            self.output_topic,
            pub_qos,
        )

        self.get_logger().info(
            f"scan_front_filter: {self.input_topic} -> {self.output_topic}, "
            f"front=[{math.degrees(self.min_angle):.1f}, {math.degrees(self.max_angle):.1f}] deg, "
            f"lidar_yaw={math.degrees(self.lidar_yaw):.1f} deg, "
            f"range=[{self.min_keep_range:.2f}, {self.max_keep_range:.2f}], "
            f"fixed_bins={self.fixed_bins}"
        )

    def interpolate_range(self, msg: LaserScan, angle: float) -> float:
        if msg.angle_increment == 0.0:
            return float("inf")

        idx_float = (angle - msg.angle_min) / msg.angle_increment
        idx = int(round(idx_float))

        if idx < 0 or idx >= len(msg.ranges):
            return float("inf")

        r = float(msg.ranges[idx])
        if not math.isfinite(r):
            return float("inf")

        return r

    def cb_scan(self, msg: LaserScan):
        out = LaserScan()
        out.header = msg.header
        out.header.frame_id = msg.header.frame_id

        # Keep output geometry as a full 360-degree scan in the LiDAR frame.
        # This avoids lying about the frame while still removing rear/body rays.
        out.angle_min = -math.pi
        out.angle_max = math.pi
        out.angle_increment = (out.angle_max - out.angle_min) / float(self.fixed_bins - 1)

        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time

        out.range_min = self.min_keep_range
        out.range_max = self.max_keep_range

        ranges: List[float] = []

        for i in range(self.fixed_bins):
            angle_lidar = out.angle_min + i * out.angle_increment

            # Convert ray direction to base_link convention only for filtering.
            angle_base = wrap_pi(angle_lidar + self.lidar_yaw)

            in_front = self.min_angle <= angle_base <= self.max_angle

            if not in_front:
                ranges.append(float("inf"))
                continue

            r = self.interpolate_range(msg, angle_lidar)

            if r < self.min_keep_range or r > self.max_keep_range:
                ranges.append(float("inf"))
            else:
                ranges.append(r)

        out.ranges = ranges
        out.intensities = []

        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFrontFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
