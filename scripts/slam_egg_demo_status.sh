#!/usr/bin/env bash

CIS_DIR="${CIS_DIR:-/home/lerobot/CIS}"
RUNTIME_DIR="${CIS_DIR}/.slam_egg_demo"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"

source /opt/ros/humble/setup.bash
source "${CIS_DIR}/ros2_ws/install/setup.bash" 2>/dev/null || true
source /home/lerobot/robot_ws/install/setup.bash 2>/dev/null || true
set -u

OK=0
WARN=0
FAIL=0

ok() { echo "[ OK ] $*"; OK=$((OK + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }

has_node() {
    timeout 4 ros2 node list 2>/dev/null | grep -Fxq "$1"
}

topic_counts() {
    timeout 4 ros2 topic info "$1" 2>/dev/null | awk '
        /Publisher count:/ {publisher_count=$3}
        /Subscription count:/ {subscriber_count=$3}
        END {printf "%s %s", publisher_count+0, subscriber_count+0}'
}

check_topic() {
    local topic="$1"
    local need_pub="${2:-1}"
    local counts pub sub
    counts="$(topic_counts "${topic}")"
    pub="${counts%% *}"
    sub="${counts##* }"
    if [[ "${pub}" -ge "${need_pub}" ]]; then
        ok "${topic}: publishers=${pub}, subscribers=${sub}"
    else
        fail "${topic}: publishers=${pub}, subscribers=${sub}"
    fi
}

check_lifecycle() {
    local node="$1"
    local state
    state="$(timeout 4 ros2 lifecycle get "${node}" 2>/dev/null || true)"
    if [[ "${state}" == active* ]]; then
        ok "${node}: ${state}"
    else
        fail "${node}: ${state:-no response}"
    fi
}

echo "============================================================"
echo "LeKiwi SLAM egg demo status"
echo "============================================================"

echo
echo "[Processes]"
for name in base_driver lidar scan_filter cmd_vel_mux slam_nav camera_stream egg_detector rviz; do
    pid_file="${PID_DIR}/${name}.pid"
    if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            ok "${name}: running (PID ${pid})"
        else
            fail "${name}: stopped"
        fi
    else
        warn "${name}: no PID file"
    fi
done

echo
echo "[ROS Nodes]"
for node in \
    /lekiwi_base_driver_odom \
    /ydlidar_ros2_driver_node \
    /scan_front_filter \
    /cmd_vel_mux_node \
    /slam_toolbox \
    /planner_server \
    /controller_server \
    /bt_navigator \
    /egg_map_marker \
    /egg_approach \
    /sports_ball_egg_detector \
    /rviz
do
    if has_node "${node}"; then
        ok "${node}"
    else
        fail "${node}: missing"
    fi
done

echo
echo "[Lifecycle]"
check_lifecycle /planner_server
check_lifecycle /controller_server
check_lifecycle /bt_navigator

echo
echo "[Topics]"
check_topic /scan
check_topic /scan_front
check_topic /odom
check_topic /map
check_topic /egg_detection
check_topic /egg_markers
check_topic /egg_locations
check_topic /auto/cmd_vel
check_topic /safe_cmd_vel

echo
echo "[Rates]"
scan_rate="$(timeout 5 ros2 topic hz /scan_front --window 10 2>/dev/null | grep -m1 'average rate:' || true)"
if [[ "${scan_rate}" == *"average rate:"* ]]; then
    ok "/scan_front rate: ${scan_rate}"
else
    fail "/scan_front has no measured rate"
fi

echo
echo "[TF]"
if timeout 4 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q "Translation:"; then
    ok "map -> base_link available"
else
    fail "map -> base_link unavailable"
fi
if timeout 4 ros2 run tf2_ros tf2_echo base_link laser_frame_raw 2>/dev/null | grep -q "Translation:"; then
    ok "base_link -> laser_frame_raw available"
elif timeout 4 ros2 run tf2_ros tf2_echo base_link laser_frame 2>/dev/null | grep -q "Translation:"; then
    ok "base_link -> laser_frame available"
else
    fail "base_link -> laser frame unavailable"
fi

echo
echo "[Likely Problems]"
if ! has_node /sports_ball_egg_detector; then
    echo "- YOLO detector is not running. Check ${LOG_DIR}/egg_detector.log"
fi
if [[ ! -f "${PID_DIR}/camera_stream.pid" ]] || ! kill -0 "$(cat "${PID_DIR}/camera_stream.pid" 2>/dev/null || true)" 2>/dev/null; then
    echo "- Camera stream is not running. Check ${LOG_DIR}/camera_stream.log"
fi
if ! has_node /rviz; then
    echo "- RViz is not running. For SSH GUI forwarding, reconnect with: ssh -X <user>@<host>"
    echo "  Then check ${LOG_DIR}/rviz.log"
fi
if [[ "$(topic_counts /scan_front)" == 0* ]]; then
    echo "- /scan_front is missing. Check LiDAR and scan_filter logs."
fi
if ! timeout 4 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q "Translation:"; then
    echo "- SLAM TF is missing. Check /odom, /scan_front, and slam_nav.log."
fi
if [[ "$(topic_counts /safe_cmd_vel)" == 0* ]]; then
    echo "- /safe_cmd_vel has no publisher. cmd_vel_mux_node is required."
fi
if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | grep -q ':5556 '; then
        ok "ZMQ camera stream port 5556 appears to be listening"
    else
        warn "No local listener found on TCP 5556. The detector may wait forever without the LeKiwi camera observation stream."
    fi
fi

echo
echo "[Error Logs]"
if [[ "${FAIL}" -eq 0 ]]; then
    echo "No active failures. Earlier startup warnings may remain in the log files."
else
    for log in "${LOG_DIR}"/*.log; do
        [[ -e "${log}" ]] || continue
        matches="$(grep -Eai 'error|exception|traceback|failed|fatal' "${log}" 2>/dev/null | tail -n 3 || true)"
        if [[ -n "${matches}" ]]; then
            echo "--- $(basename "${log}")"
            echo "${matches}"
        fi
    done
fi

echo
echo "============================================================"
echo "Summary: OK=${OK}, WARN=${WARN}, FAIL=${FAIL}"
if [[ "${FAIL}" -eq 0 ]]; then
    echo "Result: core demo components look healthy."
else
    echo "Result: fix FAIL items before running the demo."
fi
echo "Logs: ${LOG_DIR}"
echo "============================================================"

exit "${FAIL}"
