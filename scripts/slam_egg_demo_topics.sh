#!/usr/bin/env bash

source /opt/ros/humble/setup.bash
source /home/lerobot/CIS/ros2_ws/install/setup.bash 2>/dev/null || true
source /home/lerobot/robot_ws/install/setup.bash 2>/dev/null || true
set -u

case "${1:-help}" in
    egg)
        ros2 topic echo /egg_detection
        ;;
    locations)
        ros2 topic echo /egg_locations
        ;;
    markers)
        ros2 topic info /egg_markers --verbose
        ;;
    auto)
        ros2 topic echo /auto/cmd_vel
        ;;
    safe)
        ros2 topic echo /safe_cmd_vel
        ;;
    scan)
        ros2 topic hz /scan_front
        ;;
    map)
        ros2 topic echo /map --once
        ;;
    nodes)
        ros2 node list | sort
        ;;
    topics)
        ros2 topic list | sort
        ;;
    logs)
        grep -ERai 'error|exception|traceback|failed|fatal' \
            /home/lerobot/CIS/.slam_egg_demo/logs 2>/dev/null | tail -n 50
        ;;
    *)
        cat <<'EOF'
Usage:
  slam_egg_demo_topics.sh egg        # YOLO provisional egg detection
  slam_egg_demo_topics.sh locations  # stored egg positions in map frame
  slam_egg_demo_topics.sh markers    # RViz marker topic information
  slam_egg_demo_topics.sh auto       # Nav2 velocity before mux
  slam_egg_demo_topics.sh safe       # velocity sent to base driver
  slam_egg_demo_topics.sh scan       # /scan_front rate
  slam_egg_demo_topics.sh map        # one SLAM map message
  slam_egg_demo_topics.sh nodes      # all ROS nodes
  slam_egg_demo_topics.sh topics     # all ROS topics
  slam_egg_demo_topics.sh logs       # recent error lines from demo logs
EOF
        ;;
esac
