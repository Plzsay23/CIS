#!/usr/bin/env bash
set -u

CIS_DIR="${CIS_DIR:-/home/lerobot/CIS}"
RUNTIME_DIR="${CIS_DIR}/.slam_egg_demo"
PID_DIR="${RUNTIME_DIR}/pids"

echo "Stopping LeKiwi SLAM egg demo..."

declare -a stopped_pids=()

for pid_file in "${PID_DIR}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    name="$(basename "${pid_file}" .pid)"
    pid="$(cat "${pid_file}" 2>/dev/null || true)"

    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        echo "[STOP] ${name} (PID ${pid})"
        stopped_pids+=("${pid}")
    else
        echo "[SKIP] ${name} is not running"
    fi
    rm -f "${pid_file}"
done

sleep 2

for pid in "${stopped_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        echo "[KILL] remaining process group ${pid}"
    fi
done

# Clean up demo processes that may have outlived their launcher shell.
patterns=(
    "${CIS_DIR}/scripts/lekiwi_base_driver_odom_node.py"
    "${CIS_DIR}/tools/scan_front_filter.py"
    "${CIS_DIR}/scripts/cmd_vel_mux_node.py"
    "${CIS_DIR}/tools/lekiwi_camera_obs_stream.py"
    "${CIS_DIR}/tools/yolo_coco_proxy_egg_from_lekiwi_obs.py"
    "ros2 launch ydlidar_ros2_driver ydlidar_launch.py"
    "ros2 launch lekiwi_nav lekiwi_slam_navigation.launch.py"
    "ydlidar_ros2_driver_node"
    "static_tf_pub_laser"
    "async_slam_toolbox_node"
    "/opt/ros/humble/lib/nav2_"
    "/opt/ros/humble/lib/slam_toolbox/"
    "/home/lerobot/robot_ws/install/lekiwi_nav/"
    "rviz2"
)

for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
        pkill -TERM -f "${pattern}" 2>/dev/null || true
        echo "[CLEAN] ${pattern}"
    fi
done

sleep 2

for pattern in "${patterns[@]}"; do
    pkill -KILL -f "${pattern}" 2>/dev/null || true
done

if command -v ros2 >/dev/null 2>&1 || [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash 2>/dev/null || true
    set -u
    timeout 3 ros2 daemon stop >/dev/null 2>&1 || true
fi

echo "Done. Demo ROS nodes, RViz, and hardware holders have been stopped."
