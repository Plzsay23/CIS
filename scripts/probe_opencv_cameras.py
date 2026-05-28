from pathlib import Path
import cv2
import time


def probe_device(dev: str, width: int = 640, height: int = 480, fps: int = 15) -> None:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"{dev}: OPEN_FAIL")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok_count = 0
    shape = None

    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_count += 1
            shape = frame.shape
        time.sleep(0.03)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

    cap.release()

    status = "OK" if ok_count > 0 else "READ_FAIL"
    print(
        f"{dev}: {status} "
        f"ok_count={ok_count}/20 "
        f"shape={shape} "
        f"actual={actual_w:.0f}x{actual_h:.0f}@{actual_fps:.1f} "
        f"fourcc={fourcc_str!r}"
    )


def main() -> None:
    devices = sorted(
        Path("/dev").glob("video*"),
        key=lambda p: int(str(p.name).replace("video", "")),
    )

    if not devices:
        print("No /dev/video* devices found.")
        return

    for path in devices:
        probe_device(str(path))


if __name__ == "__main__":
    main()