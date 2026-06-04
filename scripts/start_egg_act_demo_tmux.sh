#!/usr/bin/env bash
set -eo pipefail

: "${HF_USER:?export HF_USER=plzsay 처럼 먼저 설정해야 합니다}"
: "${TASK_NAME:?export TASK_NAME=pick_egg_act 처럼 먼저 설정해야 합니다}"
: "${TOP_REALSENSE_SERIAL:?export TOP_REALSENSE_SERIAL=리얼센스시리얼 먼저 설정해야 합니다}"

ROOT="${CIS_ROOT:-$HOME/CIS}"
SESSION="${SESSION_NAME:-cis_act}"

cd "$ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux가 없습니다. 설치: sudo apt install tmux"
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

BASE_CMD="cd $ROOT && source .venv/bin/activate && source /opt/ros/humble/setup.bash"

tmux new-session -d -s "$SESSION" -n mux \
  "$BASE_CMD && python scripts/cmd_vel_mux_node_act_guard.py"

tmux split-window -t "$SESSION":0 -h \
  "$BASE_CMD && python scripts/lekiwi_base_driver_odom_act_node.py --port /dev/follower --cmd-topic /safe_cmd_vel --act-action-space auto"

tmux split-window -t "$SESSION":0 -v \
  "$BASE_CMD && python tools/top_realsense_ros_publisher.py --serial $TOP_REALSENSE_SERIAL --topic /camera/top/image_raw --frame-id top_camera_optical_frame --width 640 --height 480 --fps 30"

tmux select-pane -t "$SESSION":0.0
tmux split-window -t "$SESSION":0 -v \
  "$BASE_CMD && python tools/yolo_sports_ball_egg_from_ros_image_act.py --image-topic /camera/top/image_raw --output-topic /egg_detection --model $ROOT/yolov10n.pt --device 0 --repeat"

tmux select-pane -t "$SESSION":0.1
tmux split-window -t "$SESSION":0 -v \
  "$BASE_CMD && python tools/act_policy_bridge_ros.py --policy-type act --pretrained-name-or-path ${HF_USER}/${TASK_NAME} --device cuda --camera-topic /camera/wrist/image_raw --image-key observation.images.wrist --state-key observation.state --action-keys arm_shoulder_pan,arm_shoulder_lift,arm_elbow_flex,arm_wrist_flex,arm_wrist_roll,arm_gripper --publish-topic /act/arm_action"

tmux select-pane -t "$SESSION":0.2
tmux split-window -t "$SESSION":0 -v \
  "$BASE_CMD && python tools/egg_act_mission_manager.py --egg-topic /egg_detection --act-enabled-topic /act/enabled --auto-cmd-topic /auto/cmd_vel --target-distance 0.28"

tmux select-layout -t "$SESSION":0 tiled

echo "[OK] tmux session started: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
