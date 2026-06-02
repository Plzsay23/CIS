#!/usr/bin/env python3
"""Send a one-lap waypoint route to Nav2 for the generated LeKiwi map."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses
from rclpy.action import ActionClient
from rclpy.node import Node


DEFAULT_WAYPOINTS = [
    (-38.2463264465332, -3.923205614089966),
    (58.1522102355957, -3.837080478668213),
    (57.93247604370117, -2.078298330307007),
    (-38.328094482421875, -2.054600238800049),
]


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_to_next(points: list[tuple[float, float]], index: int) -> float:
    x0, y0 = points[index]
    x1, y1 = points[(index + 1) % len(points)]
    return math.atan2(y1 - y0, x1 - x0)


class LoopWaypointClient(Node):
    def __init__(self, action_name: str):
        super().__init__("lekiwi_loop_waypoint_client")
        self.client = ActionClient(self, NavigateThroughPoses, action_name)
        self.goal_handle = None

    def make_pose(self, x: float, y: float, yaw: float, frame_id: str) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def send_route(self, points: list[tuple[float, float]], frame_id: str) -> bool:
        self.get_logger().info("Waiting for Nav2 /navigate_through_poses action server...")
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available. Is bt_navigator active?")
            return False

        goal = NavigateThroughPoses.Goal()
        goal.poses = [
            self.make_pose(x, y, yaw_to_next(points, i), frame_id)
            for i, (x, y) in enumerate(points)
        ]

        self.get_logger().info(f"Sending {len(goal.poses)} waypoints in frame '{frame_id}'")
        future = self.client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()
        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().error("Waypoint goal was rejected")
            return False

        self.get_logger().info("Waypoint goal accepted")
        result_future = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        status = result.status if result is not None else None
        self.get_logger().info(f"Waypoint route finished with status={status}")
        return status == 4  # GoalStatus.STATUS_SUCCEEDED

    def feedback_callback(self, msg):
        feedback = msg.feedback
        current = getattr(feedback, "current_waypoint", None)
        if current is not None:
            self.get_logger().info(f"Current waypoint index: {current}", throttle_duration_sec=2.0)

    def cancel(self):
        if self.goal_handle is not None:
            self.get_logger().warn("Canceling waypoint route...")
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)


def repeated_route(points: Iterable[tuple[float, float]], loops: int) -> list[tuple[float, float]]:
    base = list(points)
    if loops < 1:
        raise ValueError("--loops must be >= 1")
    route: list[tuple[float, float]] = []
    for _ in range(loops):
        route.extend(base)
    route.append(base[0])
    return route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send LeKiwi one-lap waypoints to Nav2.")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--action-name", default="/navigate_through_poses")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = LoopWaypointClient(args.action_name)
    points = repeated_route(DEFAULT_WAYPOINTS, args.loops)

    try:
        ok = node.send_route(points, args.frame_id)
    except KeyboardInterrupt:
        node.cancel()
        ok = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
