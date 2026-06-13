#!/usr/bin/env bash
set -eo pipefail

cd /home/lerobot/CIS

# venv 먼저 활성화
source /home/lerobot/CIS/.venv/bin/activate

# ROS setup은 set -u 상태에서 AMENT_TRACE_SETUP_FILES 에러가 날 수 있으므로
# source 하는 동안만 nounset을 끈다.
set +u
source /opt/ros/humble/setup.bash

if [ -f /home/lerobot/CIS/ros2_ws/install/setup.bash ]; then
  source /home/lerobot/CIS/ros2_ws/install/setup.bash
fi
set -u

export DASHBOARD_V2_MAP_YAML="${DASHBOARD_V2_MAP_YAML:-/home/lerobot/CIS/nav_maps/generated/lekiwi_map_v8.yaml}"

exec python3 /home/lerobot/CIS/dashboard/dashboard_v2_server.py
