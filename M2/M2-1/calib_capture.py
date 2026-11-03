import os

import cv2
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_imgs")
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
PREVIEW_MAX_WIDTH = 1280


def existing_indices():
    """返回 calib_imgs 里已有 img_XX.jpg 的序号（升序）。"""
    used = []
    for name in os.listdir(SAVE_DIR):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".jpg" and stem.startswith("img_"):
            try:
                used.append(int(stem[4:]))
            except ValueError:
                pass
    return sorted(used)


def max_index():
    """当前最大序号，没有图片时返回 -1。"""
    used = existing_indices()
    return used[-1] if used else -1


def next_index():
    """下一张自动编号。"""
    return max_index() + 1


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

    max_n = max_index()
    next_n = max_n + 1
    typed = ""

    def save_frame(n, frame):
        """保存为 img_<n>.jpg，覆盖或新增都报告当前最大序号。"""
        path = os.path.join(SAVE_DIR, f"img_{n:02d}.jpg")
        existed = os.path.exists(path)
        if not cv2.imwrite(path, frame):
            print(f"保存失败: {path}")
            return max_index()
        print(f"已{'覆盖' if existed else '新增'}: {path}")
        current = max_index()
        print(f"当前最大序号: {current:02d}，下一张自动编号: {current + 1:02d}")
        return current

    print("空格 = 存为下一张；输入序号后回车 = 覆盖/新增该序号")
    print("Esc = 清空输入（无输入时退出），q 退出")
    print(f"当前最大序号: {max_n:02d}，下一张自动编号: {next_n:02d}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败")
            break

        # 预览缩小显示，保存仍用原始分辨率
        scale = min(1.0, PREVIEW_MAX_WIDTH / frame.shape[1])
        preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hint = f"MAX {max_n:02d}  NEXT {next_n:02d}  INPUT {typed or '_'}"
        cv2.putText(preview, hint, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("camera", preview)
        key = cv2.waitKey(1) & 0xFF

        if ord("0") <= key <= ord("9"):
            typed += chr(key)
            continue
        if key in (8, 127):
            typed = typed[:-1]
            continue
        if key in (13, 10):
            if not typed:
                continue
            max_n = save_frame(int(typed), frame)
            next_n = max_n + 1
            typed = ""
            continue
        if key == ord(" "):
            if typed:
                continue  # 输入序号时不响应空格，避免误存
            max_n = save_frame(next_n, frame)
            next_n = max_n + 1
        elif key == ord("q"):
            break
        elif key == 27:
            if typed:
                typed = ""
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
