#!/usr/bin/env python3
"""Experimental ROS bridge for ACT policy -> /act/arm_action.

This process intentionally DOES NOT open /dev/follower. The motor bus remains owned by
scripts/lekiwi_base_driver_odom_act_node.py.

It loads the policy once and keeps it resident. While /act/enabled is true, it captures
wrist images, combines them with /act/arm_state, runs ACT, and publishes JSON actions to
/act/arm_action.

Because LeRobot feature key names vary across local branches/datasets, this file exposes
several CLI options for observation and action key mapping. If preprocessing raises a key
error, print the policy config/features and adjust --image-key/--state-key/--action-keys.
"""

import argparse
import json
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Bool, String

from lerobot.policies import get_policy_class, make_pre_post_processors


ARM_RANGES = {
    "arm_shoulder_pan": (695, 3379),
    "arm_shoulder_lift": (841, 3237),
    "arm_elbow_flex": (928, 3076),
    "arm_wrist_flex": (980, 3258),
    "arm_wrist_roll": (0, 4095),
    "arm_gripper": (2046, 3100),
}

DEFAULT_ACTION_KEYS = [
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def raw_to_norm_m100_100(name: str, raw: float) -> float:
    lo, hi = ARM_RANGES[name]
    if hi == lo:
        return 0.0
    alpha = (float(raw) - lo) / (hi - lo)
    return float(clamp(alpha * 200.0 - 100.0, -100.0, 100.0))


def bgr_to_model_image(frame_bgr: np.ndarray, rgb: bool) -> np.ndarray:
    if rgb:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr


class ActPolicyBridge(Node):
    def __init__(self, args):
        super().__init__("act_policy_bridge_ros")
        self.args = args
        self.enabled = False
        self.last_arm_state: Dict = {}
        self.action_queue = deque()
        self.last_infer_time = 0.0
        self.last_pub_time = 0.0

        self.action_keys = [s.strip() for s in args.action_keys.split(",") if s.strip()]
        if not self.action_keys:
            self.action_keys = DEFAULT_ACTION_KEYS

        self.action_pub = self.create_publisher(String, args.arm_action_topic, 10)
        self.status_pub = self.create_publisher(String, args.bridge_status_topic, 10)
        self.create_subscription(Bool, args.act_enabled_topic, self.on_enabled, 10)
        self.create_subscription(String, args.arm_state_topic, self.on_arm_state, 10)

        self.capture = cv2.VideoCapture(args.wrist_device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, args.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open wrist camera: {args.wrist_device}")

        self.get_logger().warn("Loading ACT policy. This should happen only once.")
        policy_class = get_policy_class(args.policy_type)
        self.policy = policy_class.from_pretrained(args.pretrained_name_or_path)
        self.policy.to(args.policy_device)
        self.policy.eval()

        device_override = {"device": args.policy_device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=args.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        self.timer = self.create_timer(1.0 / max(args.fps, 1.0), self.on_timer)
        self.get_logger().info(
            f"ACT bridge ready. image_key={args.image_key}, state_key={args.state_key}, "
            f"action_keys={self.action_keys}, publish={args.arm_action_topic}"
        )

    def on_enabled(self, msg: Bool):
        self.enabled = bool(msg.data)
        self.action_queue.clear()
        self.last_infer_time = 0.0
        self.last_pub_time = 0.0
        self.publish_status("enabled" if self.enabled else "disabled")

    def on_arm_state(self, msg: String):
        try:
            self.last_arm_state = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad arm state JSON: {e}")

    def publish_status(self, state: str, extra: Optional[dict] = None):
        payload = {
            "state": state,
            "enabled": self.enabled,
            "queue_size": len(self.action_queue),
            "time": time.time(),
        }
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def get_state_vector(self) -> np.ndarray:
        raw = self.last_arm_state.get("raw_position", {}) if isinstance(self.last_arm_state, dict) else {}
        vals: List[float] = []
        for name in self.action_keys:
            if name == "arm_gripper":
                # If raw gripper is absent, keep neutral/open-ish value.
                v = raw.get(name, 0.0)
                if abs(float(v)) > 100.0:
                    # Map gripper raw to UI-ish 0..100 then keep in 0..100.
                    lo, hi = ARM_RANGES[name]
                    vals.append(float(clamp((float(v) - lo) / max(1, hi - lo) * 100.0, 0.0, 100.0)))
                else:
                    vals.append(float(v))
            else:
                vals.append(raw_to_norm_m100_100(name, raw.get(name, 0.0)))
        return np.asarray(vals, dtype=np.float32)

    def capture_image(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("wrist camera read failed")
        if self.args.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
            frame = cv2.resize(frame, (self.args.width, self.args.height))
        return bgr_to_model_image(frame, rgb=self.args.rgb_image)

    def build_observation(self) -> Dict:
        image = self.capture_image()
        state = self.get_state_vector()

        obs = {
            self.args.state_key: torch.from_numpy(state).float(),
            self.args.image_key: torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "task": self.args.task,
        }
        return obs

    def run_inference(self):
        obs = self.build_observation()

        # Most LeRobot preprocessors accept an unbatched dict and add batch internally.
        # If your local branch expects a different image/state key, change CLI args.
        with torch.no_grad():
            proc_obs = self.preprocessor(obs)
            if hasattr(self.policy, "predict_action_chunk"):
                action_tensor = self.policy.predict_action_chunk(proc_obs)
                if action_tensor.ndim == 3:
                    action_tensor = action_tensor.squeeze(0)
            else:
                action_tensor = self.policy.select_action(proc_obs)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)

            processed = []
            for i in range(min(action_tensor.shape[0], self.args.actions_per_chunk)):
                single = action_tensor[i]
                if single.ndim == 1:
                    single = single.unsqueeze(0)
                out = self.postprocessor(single)
                if isinstance(out, torch.Tensor):
                    out = out.detach().cpu().squeeze(0).tolist()
                processed.append(out)

        self.action_queue.extend(processed)
        self.last_infer_time = time.monotonic()
        self.publish_status("inference_ok", {"added_actions": len(processed)})

    def publish_next_action(self):
        if not self.action_queue:
            return
        values = self.action_queue.popleft()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().flatten().tolist()
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(f"Unsupported action output type: {type(values)}")

        action = {}
        for i, key in enumerate(self.action_keys):
            if i >= len(values):
                break
            action[key] = float(values[i])

        msg = String()
        msg.data = json.dumps({"action": action}, ensure_ascii=False)
        self.action_pub.publish(msg)
        self.last_pub_time = time.monotonic()

    def on_timer(self):
        if not self.enabled:
            return

        now = time.monotonic()
        try:
            if len(self.action_queue) <= self.args.refill_threshold:
                if now - self.last_infer_time >= self.args.min_infer_interval_s:
                    self.run_inference()

            if now - self.last_pub_time >= 1.0 / max(self.args.fps, 1.0):
                self.publish_next_action()
        except Exception as e:
            self.publish_status("error", {"error": str(e)})
            self.get_logger().error(f"ACT bridge error: {e}")

    def close(self):
        try:
            self.capture.release()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-type", default="act")
    p.add_argument("--pretrained-name-or-path", required=True)
    p.add_argument("--policy-device", default="cuda")
    p.add_argument("--task", required=True)

    p.add_argument("--wrist-device", default="/dev/wrist")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--rotate-180", action="store_true")
    p.add_argument("--bgr-image", dest="rgb_image", action="store_false")
    p.set_defaults(rgb_image=True)

    p.add_argument("--image-key", default="observation.images.wrist")
    p.add_argument("--state-key", default="observation.state")
    p.add_argument("--action-keys", default=",".join(DEFAULT_ACTION_KEYS))
    p.add_argument("--actions-per-chunk", type=int, default=50)
    p.add_argument("--refill-threshold", type=int, default=10)
    p.add_argument("--min-infer-interval-s", type=float, default=0.02)

    p.add_argument("--act-enabled-topic", default="/act/enabled")
    p.add_argument("--arm-state-topic", default="/act/arm_state")
    p.add_argument("--arm-action-topic", default="/act/arm_action")
    p.add_argument("--bridge-status-topic", default="/act/policy_bridge_status")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ActPolicyBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
