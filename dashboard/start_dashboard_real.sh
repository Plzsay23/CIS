#!/usr/bin/env bash
set -eo pipefail

cd /home/lerobot/CIS

if [ -f /home/lerobot/CIS/.venv/bin/activate ]; then
  source /home/lerobot/CIS/.venv/bin/activate
fi

# ROS setup.bash can break under nounset, so keep nounset disabled while sourcing ROS.
set +u
source /opt/ros/humble/setup.bash

if [ -f /home/lerobot/CIS/ros2_ws/install/setup.bash ]; then
  source /home/lerobot/CIS/ros2_ws/install/setup.bash
fi
set -u

export DASHBOARD_REAL_MAP_YAML="${DASHBOARD_REAL_MAP_YAML:-/home/lerobot/CIS/nav_maps/generated/lekiwi_map_v8.yaml}"
export DASHBOARD_REAL_POSE_TOPIC="${DASHBOARD_REAL_POSE_TOPIC:-/odom}"
export DASHBOARD_REAL_PORT="${DASHBOARD_REAL_PORT:-8082}"

exec python3 /home/lerobot/CIS/dashboard/dashboard_real_server.py
