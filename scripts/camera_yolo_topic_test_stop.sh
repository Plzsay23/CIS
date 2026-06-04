#!/usr/bin/env bash
set -u

CIS_DIR="${CIS_DIR:-/home/lerobot/CIS}"
PID_DIR="${CIS_DIR}/.camera_yolo_topic_test/pids"

for pid_file in "${PID_DIR}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    name="$(basename "${pid_file}" .pid)"
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        echo "[STOP] ${name} (PID ${pid})"
    fi
    rm -f "${pid_file}"
done

sleep 1
pkill -TERM -f "${CIS_DIR}/tools/top_camera_ros_publisher.py" 2>/dev/null || true
pkill -TERM -f "${CIS_DIR}/tools/yolo_sports_ball_from_ros_image.py" 2>/dev/null || true
echo "Camera + YOLO topic test stopped."
