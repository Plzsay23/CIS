#!/usr/bin/env bash
set -u

CIS_DIR="${CIS_DIR:-/home/lerobot/CIS}"
ROBOT_WS="${ROBOT_WS:-/home/lerobot/robot_ws}"
RUNTIME_DIR="${CIS_DIR}/.slam_egg_demo"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"

BASE_PORT="${BASE_PORT:-/dev/follower}"
YOLO_MODEL="${YOLO_MODEL:-${CIS_DIR}/yolov10n.pt}"
YOLO_CAMERA="${YOLO_CAMERA:-top}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_VIEW="${YOLO_VIEW:-false}"
STOP_ON_EGG_DETECTION="${STOP_ON_EGG_DETECTION:-false}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.55}"
CAMERA_PITCH_DEG="${CAMERA_PITCH_DEG:-50.0}"
CAMERA_FORWARD_OFFSET="${CAMERA_FORWARD_OFFSET:-0.0}"
CAMERA_HORIZONTAL_FOV_DEG="${CAMERA_HORIZONTAL_FOV_DEG:-70.0}"
CAMERA_VERTICAL_FOV_DEG="${CAMERA_VERTICAL_FOV_DEG:-42.5}"
ZMQ_ADDRESS="${ZMQ_ADDRESS:-tcp://127.0.0.1:5556}"
USE_SIM_TIME="${USE_SIM_TIME:-false}"
ENABLE_EGG_APPROACH="${ENABLE_EGG_APPROACH:-true}"
EGG_STAND_OFF_DISTANCE="${EGG_STAND_OFF_DISTANCE:-0.2}"
ENABLE_RVIZ="${ENABLE_RVIZ:-true}"
RVIZ_CONFIG="${RVIZ_CONFIG:-/home/lerobot/.rviz2/default.rviz}"
ENABLE_CAMERA_STREAM="${ENABLE_CAMERA_STREAM:-true}"

find_realsense_color_camera() {
    local dev product fmt
    for dev in /dev/video*; do
        [[ -c "${dev}" ]] || continue
        product="$(udevadm info --query=property --name="${dev}" 2>/dev/null | awk -F= '/^ID_V4L_PRODUCT=/ {print $2; exit}')"
        [[ "${product}" == *"RealSense"* ]] || continue
        fmt="$(v4l2-ctl -d "${dev}" --get-fmt-video 2>/dev/null | awk -F"'" '/Pixel Format/ {print $2; exit}')"
        case "${fmt}" in
            YUYV|MJPG|RGB3|BGR3)
                echo "${dev}"
                return 0
                ;;
        esac
    done
    return 1
}

DEFAULT_TOP_CAMERA_DEVICE="auto"
TOP_CAMERA_DEVICE="${TOP_CAMERA_DEVICE:-${DEFAULT_TOP_CAMERA_DEVICE}}"
if [[ "${TOP_CAMERA_DEVICE}" != "auto" && -n "${TOP_CAMERA_DEVICE}" ]]; then
    TOP_CAMERA_DEVICE="$(readlink -f "${TOP_CAMERA_DEVICE}" 2>/dev/null || echo "${TOP_CAMERA_DEVICE}")"
fi

mkdir -p "${LOG_DIR}" "${PID_DIR}"

if [[ "${ENABLE_CAMERA_STREAM}" == "true" ]] && [[ "${TOP_CAMERA_DEVICE}" != "auto" ]] && { [[ -z "${TOP_CAMERA_DEVICE}" ]] || [[ ! -c "${TOP_CAMERA_DEVICE}" ]]; }; then
    echo "[FAIL] RealSense color camera device was not found."
    echo "       /dev/top: $(ls -l /dev/top 2>/dev/null || echo 'not found')"
    echo "       available cameras:"
    v4l2-ctl --list-devices 2>/dev/null || true
    echo "       Reconnect the RealSense camera, then run this script again."
    exit 1
fi

start_process() {
    local name="$1"
    local command="$2"
    local pid_file="${PID_DIR}/${name}.pid"
    local log_file="${LOG_DIR}/${name}.log"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[SKIP] ${name} already running (PID ${old_pid})"
            return 0
        fi
        rm -f "${pid_file}"
    fi

    : > "${log_file}"
    nohup setsid bash -lc "${command}" > "${log_file}" 2>&1 < /dev/null &
    local pid=$!
    echo "${pid}" > "${pid_file}"
    sleep 1

    if kill -0 "${pid}" 2>/dev/null; then
        echo "[ OK ] ${name} started (PID ${pid})"
    else
        echo "[FAIL] ${name} exited. Log: ${log_file}"
        tail -n 20 "${log_file}" 2>/dev/null || true
        return 1
    fi
}

echo "============================================================"
echo "LeKiwi SLAM + sports-ball-as-egg demo start"
echo "logs: ${LOG_DIR}"
echo "============================================================"

start_process "base_driver" \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/.venv/bin/activate';
     source '${CIS_DIR}/ros2_ws/install/setup.bash';
     export PYTHONNOUSERSITE=1;
     export PYTHONPATH='${CIS_DIR}/src':\${PYTHONPATH:-};
     python3 '${CIS_DIR}/scripts/lekiwi_base_driver_odom_node.py' --port '${BASE_PORT}'"

start_process "lidar" \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/ros2_ws/install/setup.bash';
     ros2 launch ydlidar_ros2_driver ydlidar_launch.py"

start_process "scan_filter" \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/ros2_ws/install/setup.bash';
     python3 '${CIS_DIR}/tools/scan_front_filter.py' --ros-args -p input_topic:=/scan -p output_topic:=/scan_front -p min_angle_deg:=-90.0 -p max_angle_deg:=90.0 -p lidar_yaw_deg:=0.0 -p min_keep_range:=0.45 -p max_keep_range:=6.0 -p fixed_bins:=720"

start_process "cmd_vel_mux" \
    "source /opt/ros/humble/setup.bash;
     python3 '${CIS_DIR}/scripts/cmd_vel_mux_node.py'"

start_process "slam_nav" \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/ros2_ws/install/setup.bash';
     source '${ROBOT_WS}/install/setup.bash';
     ros2 launch lekiwi_nav lekiwi_slam_navigation.launch.py use_sim_time:='${USE_SIM_TIME}' enable_egg_approach:='${ENABLE_EGG_APPROACH}' egg_stand_off_distance:='${EGG_STAND_OFF_DISTANCE}'"

if [[ "${ENABLE_CAMERA_STREAM}" == "true" ]]; then
    start_process "camera_stream" \
        "source '${CIS_DIR}/.venv/bin/activate';
         python3 '${CIS_DIR}/tools/lekiwi_camera_obs_stream.py' --device '${TOP_CAMERA_DEVICE}' --camera-key top --address 'tcp://*:5556' --rotate-180"
else
    echo "[SKIP] camera stream disabled (ENABLE_CAMERA_STREAM=${ENABLE_CAMERA_STREAM})"
fi

start_process "egg_detector" \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/.venv/bin/activate';
     source '${ROBOT_WS}/install/setup.bash';
     python3 '${CIS_DIR}/tools/yolo_sports_ball_egg_detection.py' --address '${ZMQ_ADDRESS}' --model '${YOLO_MODEL}' --cam '${YOLO_CAMERA}' --camera-height '${CAMERA_HEIGHT}' --camera-pitch-deg '${CAMERA_PITCH_DEG}' --camera-forward-offset '${CAMERA_FORWARD_OFFSET}' --horizontal-fov-deg '${CAMERA_HORIZONTAL_FOV_DEG}' --vertical-fov-deg '${CAMERA_VERTICAL_FOV_DEG}' --conf '${YOLO_CONF}' --device '${YOLO_DEVICE}' $([[ '${STOP_ON_EGG_DETECTION}' == 'true' ]] && echo '--stop-on-detection') $([[ '${YOLO_VIEW}' == 'true' ]] && echo '--view')"

if [[ "${ENABLE_RVIZ}" == "true" ]]; then
    start_process "rviz" \
        "source /opt/ros/humble/setup.bash;
         source '${CIS_DIR}/ros2_ws/install/setup.bash';
         source '${ROBOT_WS}/install/setup.bash';
         if [[ -z \"\${DISPLAY:-}\" ]]; then
             echo 'DISPLAY is empty. Reconnect SSH with: ssh -X <user>@<host>';
             exit 1;
         fi;
         rviz2 -d '${RVIZ_CONFIG}'"
else
    echo "[SKIP] rviz disabled (ENABLE_RVIZ=${ENABLE_RVIZ})"
fi

echo
echo "Start commands submitted. Waiting for ROS nodes to initialize..."
sleep 8
echo
"${CIS_DIR}/scripts/slam_egg_demo_status.sh" || true
echo
echo "RViz config: ${RVIZ_CONFIG}"
echo "For RViz, add MarkerArray topic: /egg_markers"
echo "For manual exploration, publish to: /dashboard/cmd_vel"
echo "Top camera device: ${TOP_CAMERA_DEVICE}"
echo "YOLO view window: ${YOLO_VIEW}"
echo "Stop on egg detection: ${STOP_ON_EGG_DETECTION}"
echo "Automatic egg approach: ${ENABLE_EGG_APPROACH}"
echo "Egg stand-off distance: ${EGG_STAND_OFF_DISTANCE} m"
echo "Camera ground projection: height=${CAMERA_HEIGHT} m, pitch=${CAMERA_PITCH_DEG} deg"
