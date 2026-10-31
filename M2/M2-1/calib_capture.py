import os

import cv2
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_imgs")
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
PREVIEW_MAX_WIDTH = 1280


def next_index():
    """从已有文件名接着编号，避免重跑时覆盖之前的照片。"""
    used = []
    for name in os.listdir(SAVE_DIR):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".jpg" and stem.startswith("img_"):
            try:
                used.append(int(stem[4:]))
            except ValueError:
                pass
    return max(used, default=-1) + 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头（WSL 下需要先用 usbipd 把摄像头挂进来）")
        return

    # USB/IP 下必须走 MJPG：YUYV 带宽不够会丢包，画面下半截会变成绿色。
    # 1080p 只能跑 15 fps，30 fps 会超过稳定带宽导致帧只收到一部分。
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"@{int(cap.get(cv2.CAP_PROP_FPS))}fps")

    index = next_index()
    print("按空格拍照，按 q 或 Esc 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败")
            break

        # 预览缩小显示，保存仍用原始分辨率
        scale = min(1.0, PREVIEW_MAX_WIDTH / frame.shape[1])
        preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("camera", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            path = os.path.join(SAVE_DIR, f"img_{index:02d}.jpg")
            cv2.imwrite(path, frame)
            print("已保存:", path)
            index += 1
        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
