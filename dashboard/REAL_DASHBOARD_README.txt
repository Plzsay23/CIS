REAL dashboard run order

1) cmd_vel mux
cd ~/CIS
source /opt/ros/humble/setup.bash
source ~/CIS/.venv/bin/activate
source ~/CIS/ros2_ws/install/setup.bash
python3 ~/CIS/scripts/cmd_vel_mux_node.py

2) odom/base driver
~/CIS/dashboard/start_lekiwi_odom_real.sh

Equivalent manual command:
cd ~/CIS
source /opt/ros/humble/setup.bash
source ~/CIS/.venv/bin/activate
source ~/CIS/ros2_ws/install/setup.bash
python3 ~/CIS/scripts/lekiwi_base_driver_odom_node.py \
  --port /dev/follower \
  --calibration-json ~/CIS/config/lekiwi.json \
  --arm-acceleration 160 \
  --arm-home-return-seconds 0.8 \
  --arm-home-return-fps 50

3) real dashboard
~/CIS/dashboard/start_dashboard_real.sh

Open:
http://localhost:8082

Check:
ros2 topic echo /dashboard/cmd_vel
ros2 topic echo /safe_cmd_vel
ros2 topic echo /odom --once
curl http://localhost:8082/health | python3 -m json.tool
