#!/usr/bin/env bash
set -u

CIS_DIR="${CIS_DIR:-/home/lerobot/CIS}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/top}"
RUNTIME_DIR="${CIS_DIR}/.camera_yolo_topic_test"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

start_process() {
    local name="$1"
    local command="$2"
    local log_file="${LOG_DIR}/${name}.log"
    local pid_file="${PID_DIR}/${name}.pid"

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

echo "Camera + YOLO ROS topic test only. No base driver or Nav2 will be started."

start_process camera \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/.venv/bin/activate';
     python3 '${CIS_DIR}/tools/top_camera_ros_publisher.py' --device '${CAMERA_DEVICE}' --rotate-180"

start_process yolo \
    "source /opt/ros/humble/setup.bash;
     source '${CIS_DIR}/.venv/bin/activate';
     python3 '${CIS_DIR}/tools/yolo_coco_proxy_from_ros_image.py'"

echo
echo "RViz Image topics:"
echo "  /camera/top/image_raw"
echo "  /camera/top/yolo_annotated"
echo "Detection topic:"
echo "  /coco_proxy_detection"
echo "Logs:"
echo "  ${LOG_DIR}"
