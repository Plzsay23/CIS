#!/usr/bin/env bash
set -eo pipefail

cd /home/lerobot/CIS

set +u
source /opt/ros/humble/setup.bash
source /home/lerobot/CIS/.venv/bin/activate
if [ -f /home/lerobot/CIS/ros2_ws/install/setup.bash ]; then
  source /home/lerobot/CIS/ros2_ws/install/setup.bash
fi
set -u

exec python3 /home/lerobot/CIS/scripts/lekiwi_base_driver_odom_node.py \
  --port /dev/follower \
  --calibration-json /home/lerobot/CIS/config/lekiwi.json \
  --arm-acceleration 160 \
  --arm-home-return-seconds 0.8 \
  --arm-home-return-fps 50
