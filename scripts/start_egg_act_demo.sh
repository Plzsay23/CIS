#!/usr/bin/env bash
set -eo pipefail

: "${HF_USER:?export HF_USER=plzsay 처럼 먼저 설정해야 합니다}"
: "${TASK_NAME:?export TASK_NAME=pick_egg_act 처럼 먼저 설정해야 합니다}"
: "${TOP_REALSENSE_SERIAL:?export TOP_REALSENSE_SERIAL=리얼센스_시리얼 처럼 먼저 설정해야 합니다}"

ROOT="${CIS_ROOT:-$HOME/CIS}"
LOG_DIR="${LOG_DIR:-/tmp/cis_act_logs}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [ ! -f "$ROOT/tools/top_realsense_ros_publisher.py" ]; then
  echo "[ERROR] tools/top_realsense_ros_publisher.py 가 없습니다."
  echo "        RealSense serial 방식 top camera publisher 파일을 먼저 만들어야 합니다."
  exit 1
fi

PIDS=()

shutdown() {
  echo
  echo "[STOP] stopping CIS ACT demo..."

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  wait 2>/dev/null || true

  echo "[STOP] done"
}

trap shutdown INT TERM EXIT

run_node() {
  local name="$1"
  shift

  echo "[START] $name"

  (
    cd "$ROOT"
    source .venv/bin/activate

    if [ -f /opt/ros/humble/setup.bash ]; then
      source /opt/ros/humble/setup.bash
    fi

    exec "$@"
  ) > >(sed -u "s/^/[$name] /" | tee -a "$LOG_DIR/${name}.log") \
    2> >(sed -u "s/^/[$name][ERR] /" | tee -a "$LOG_DIR/${name}.log" >&2) &

  local pid=$!
  PIDS+=("$pid")

  sleep 0.4

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] $name failed to start"
    exit 1
  fi
}

echo "===================================================="
echo " CIS Egg ACT Demo"
echo " ROOT=$ROOT"
echo " HF_USER=$HF_USER"
echo " TASK_NAME=$TASK_NAME"
echo " TOP_REALSENSE_SERIAL=$TOP_REALSENSE_SERIAL"
echo " LOG_DIR=$LOG_DIR"
echo "===================================================="
echo

rm -f "$LOG_DIR"/*.log 2>/dev/null || true

run_node mux \
  python -u scripts/cmd_vel_mux_node_act_guard.py

run_node base_driver \
  python -u scripts/lekiwi_base_driver_odom_act_node.py \
    --port /dev/follower \
    --cmd-topic /safe_cmd_vel \
    --act-action-space auto

run_node top_camera \
  python -u tools/top_realsense_ros_publisher.py \
    --serial "$TOP_REALSENSE_SERIAL" \
    --topic /camera/top/image_raw \
    --frame-id top_camera_optical_frame \
    --width 640 \
    --height 480 \
    --fps 30

run_node yolo_egg \
  python -u tools/yolo_sports_ball_egg_from_ros_image_act.py \
    --image-topic /camera/top/image_raw \
    --output-topic /egg_detection \
    --model "$ROOT/yolov10n.pt" \
    --device 0 \
    --repeat

run_node act_bridge \
  python -u tools/act_policy_bridge_ros.py \
    --policy-type act \
    --pretrained-name-or-path "${HF_USER}/${TASK_NAME}" \
    --policy-device cuda \
    --task "${TASK_NAME}" \
    --wrist-device /dev/wrist \
    --width 640 \
    --height 480 \
    --fps 25

run_node mission \
  python -u tools/egg_act_mission_manager.py \
    --enable-simple-align

echo
echo "===================================================="
echo " Started all nodes."
echo " Logs are also saved in: $LOG_DIR"
echo " Press Ctrl+C here to stop everything."
echo "===================================================="
echo

while true; do
  sleep 1

  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] one process exited. Stopping all."
      exit 1
    fi
  done
done
