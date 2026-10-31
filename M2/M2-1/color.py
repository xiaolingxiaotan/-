"""颜色采样辅助：画面中间画圆圈，把红/绿/黄灯放进圈内按空格，输出 HSV。

库接口（供 detect.py 等外部调用）：
    sample_hsv(frame)      -> (N, 3) 圆圈内像素的 HSV 数组
    hsv_summary(hsv)       -> 统计结果（中位 HSV、H 圆周均值、阈值数组）
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(HERE, "hsv_samples.json")

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
PREVIEW_MAX_WIDTH = 1280
CIRCLE_RADIUS = 80          # 圆圈半径（像素，按 1920x1080 取）

HUE_PERIOD = 180.0          # OpenCV 8-bit H 的范围是 0~179
PCT_LOW = 5                 # 取 5% / 95% 分位，避开反光和边缘离群点
PCT_HIGH = 95


def sample_hsv(frame, center=None, radius=CIRCLE_RADIUS):
    """返回圆圈内像素的 HSV 数组，形状 (N, 3)，H∈[0,179]、S/V∈[0,255]。"""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center
    r = int(radius)

    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return hsv[mask]


def hsv_summary(hsv):
    """统计 HSV 数组。色相按圆周处理，红色跨 0 也不会算错。"""
    h = hsv[:, 0].astype(np.float64)
    s = hsv[:, 1].astype(np.float64)
    v = hsv[:, 2].astype(np.float64)

    # 色相圆周均值
    ang = h * (2 * np.pi / HUE_PERIOD)
    sin_m, cos_m = np.sin(ang).mean(), np.cos(ang).mean()
    mean_h = (np.arctan2(sin_m, cos_m) * HUE_PERIOD / (2 * np.pi)) % HUE_PERIOD

    # 以圆周均值为中心展开，再取分位，避免 179/0 被拆到两端
    offset = (h - mean_h + HUE_PERIOD / 2) % HUE_PERIOD - HUE_PERIOD / 2
    h_lo = (mean_h + np.percentile(offset, PCT_LOW)) % HUE_PERIOD
    h_hi = (mean_h + np.percentile(offset, PCT_HIGH)) % HUE_PERIOD

    return {
        "mean_hsv": [float(mean_h), float(s.mean()), float(v.mean())],
        "h_lo": float(h_lo),
        "h_hi": float(h_hi),
        "s_lo": float(np.percentile(s, PCT_LOW)),
        "s_hi": float(np.percentile(s, PCT_HIGH)),
        "v_lo": float(np.percentile(v, PCT_LOW)),
        "v_hi": float(np.percentile(v, PCT_HIGH)),
        "count": int(h.size),
    }


def threshold_array(summary):
    """整理成 [H_min, S_min, V_min, H_max, S_max, V_max]，可直接给 inRange。

    H_min > H_max 表示色相跨过 0，需要拆成 [0, H_max] 和 [H_min, 179] 两段。
    """
    return [
        summary["h_lo"], summary["s_lo"], summary["v_lo"],
        summary["h_hi"], summary["s_hi"], summary["v_hi"],
    ]


def save_samples(path, samples, frame_shape):
    """把每次采样的统计写入 JSON，图片不保存。"""
    h, w = frame_shape[:2]
    data = {
        "camera": {
            "resolution": [w, h],
            "circle_center": [w // 2, h // 2],
            "circle_radius": CIRCLE_RADIUS,
            "hsv_percentiles": [PCT_LOW, PCT_HIGH],
        },
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头（WSL 下需要先用 usbipd 把摄像头挂进来）")
        return

    # 与 calib_capture.py 保持一致：USB/IP 下走 MJPG，1080p 只能 15 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"@{int(cap.get(cv2.CAP_PROP_FPS))}fps")
    print("把灯放进圆圈，按空格采样，按 q 或 Esc 退出")
    print(f"采样数据保存到: {SAVE_PATH}")

    samples = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败")
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 预览缩小显示，采样仍用原始分辨率
        scale = min(1.0, PREVIEW_MAX_WIDTH / w)
        preview = cv2.resize(frame, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.circle(preview, (int(cx * scale), int(cy * scale)),
                   int(CIRCLE_RADIUS * scale), (0, 255, 0), 2)
        cv2.imshow("color", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            hsv = sample_hsv(frame, (cx, cy), CIRCLE_RADIUS)
            s = hsv_summary(hsv)
            count += 1
            thr = threshold_array(s)
            print(f"\n采样 {count}: {s['count']} 个像素")
            print(f"  平均 HSV: [{s['mean_hsv'][0]:.1f}, "
                  f"{s['mean_hsv'][1]:.1f}, {s['mean_hsv'][2]:.1f}]")
            print(f"  阈值数组: [{thr[0]:.0f}, {thr[1]:.0f}, "
                  f"{thr[2]:.0f}, {thr[3]:.0f}, {thr[4]:.0f}, {thr[5]:.0f}]")
            if s["h_lo"] > s["h_hi"]:
                print("  注意: H 跨过 0，需要拆成两段阈值")

            samples.append({
                "index": count,
                "pixels": s["count"],
                "mean_hsv": [round(x, 2) for x in s["mean_hsv"]],
                "h_lo": round(s["h_lo"], 2),
                "h_hi": round(s["h_hi"], 2),
                "s_lo": round(s["s_lo"], 2),
                "s_hi": round(s["s_hi"], 2),
                "v_lo": round(s["v_lo"], 2),
                "v_hi": round(s["v_hi"], 2),
                "threshold": [round(x, 2) for x in thr],
                "hue_wraps": s["h_lo"] > s["h_hi"],
            })
            save_samples(SAVE_PATH, samples, frame.shape)
            print(f"  已保存: {SAVE_PATH}（{len(samples)} 条）")
        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
