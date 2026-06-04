#!/usr/bin/env python3

import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"환경 변수 {name} 이(가) 설정되지 않았습니다.")
    return value


def main() -> None:
    hf_user = require_env("HF_USER")
    task_name = require_env("TASK_NAME")

    repo_id = f"{hf_user}/{task_name}"

    # 기본 LeRobot 캐시 경로
    root = Path(
        os.environ.get(
            "LEROBOT_DATASET_ROOT",
            f"/home/lerobot/.cache/huggingface/lerobot/{repo_id}",
        )
    )

    print("Loading dataset...")
    print(f"repo_id: {repo_id}")
    print(f"root: {root}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    print("Dataset loaded.")
    print("num episodes:", dataset.num_episodes)
    print("num frames:", dataset.num_frames)
    print("fps:", dataset.fps)

    print("Pushing to hub...")
    dataset.push_to_hub(
        private=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()