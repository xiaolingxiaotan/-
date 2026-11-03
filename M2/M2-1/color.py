"""颜色采样辅助：画面中间画圆圈，把红/绿/黄灯放进圈内按空格，输出反射率 R。

库接口（供 detect.py 等外部调用）：
    sample_hsv(frame)      -> (N, 3) 圆圈内像素的 HSV 数组
    hsv_summary(hsv)       -> 统计结果（平均 HSV、H 圆周均值、阈值数组）
    reflectance(frame)     -> SSR 反射率 R = V / L（只分解 V，H/S 不变）
    r_summary(r)           -> 圆圈内 R 的均值/分位
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R_SAVE_PATH = os.path.join(HERE, "r_baseline.json")
SCREW_SAVE_PATH = os.path.join(HERE, "screw_samples.json")

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
PREVIEW_MAX_WIDTH = 1280
CIRCLE_RADIUS = 80          # 圆圈半径（像素，按 1920x1080 取）

# SSR：低分辨率上做小核，等效于全分辨率的大核
RETINEX_SCALE = 8           # 下采样倍数
RETINEX_SIGMA = 12.0        # 低分辨率上的高斯 sigma
RETINEX_EPS = 1.0           # 防止除零

REPEATS = 5                 # 每种灯态采集张数

HUE_PERIOD = 180.0          # OpenCV 8-bit H 的范围是 0~179
PCT_LOW = 5                 # 取 5% / 95% 分位，避开反光和边缘离群点
PCT_HIGH = 95

# 螺丝采集仍按四种光照；R 基准采集按灯态各重复 REPEATS 张
SCENES = [
    ("normal", "正常室内光"),
    ("strong", "强光直射"),
    ("backlight", "逆光/背光"),
    ("dusk", "暗光/黄昏"),
]
STATES = [
    ("red_on", "红灯亮"),
    ("red_off", "红灯灭"),
    ("green_on", "绿灯亮"),
    ("green_off", "绿灯灭"),
    ("yellow_on", "黄灯亮"),
    ("yellow_off", "黄灯灭"),
]


def build_tasks():
    """6 种灯态各 REPEATS 张，按灯态分组，方便一次只切换一种状态。"""
    return [
        (state, state_cn, i + 1)
        for state, state_cn in STATES
        for i in range(REPEATS)
    ]


def _circle_roi(shape, center=None, radius=CIRCLE_RADIUS):
    """返回 (切片, 圆形掩膜)，所有圆圈采样统一走这里。"""
    h, w = shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)

    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return (slice(y0, y1), slice(x0, x1)), mask


def sample_hsv(frame, center=None, radius=CIRCLE_RADIUS):
    """返回圆圈内像素的 HSV 数组，形状 (N, 3)，H∈[0,179]、S/V∈[0,255]。"""
    roi, mask = _circle_roi(frame.shape, center, radius)
    return cv2.cvtColor(frame[roi], cv2.COLOR_BGR2HSV)[mask]


def reflectance(frame, scale=RETINEX_SCALE, sigma=RETINEX_SIGMA,
                eps=RETINEX_EPS):
    """SSR：只分解 V 通道，R = V / L；H、S 保持原图，不参与分解。"""
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    h, w = v.shape

    small = cv2.resize(
        v,
        (max(1, w // scale), max(1, h // scale)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    L = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return v / np.maximum(L, eps)


def r_summary(r, center=None, radius=CIRCLE_RADIUS):
    """统计圆圈内 R：均值 + 5%/95% 分位。"""
    roi, mask = _circle_roi(r.shape, center, radius)
    vals = r[roi][mask]
    return {
        "r_mean": float(vals.mean()),
        "r_lo": float(np.percentile(vals, PCT_LOW)),
        "r_hi": float(np.percentile(vals, PCT_HIGH)),
        "count": int(vals.size),
    }


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


def build_baseline(samples):
    """把每种灯态的 5 张 R 统计取平均，得到该状态的基准。"""
    baseline = {}
    for state, state_cn in STATES:
        rows = [s for s in samples if s["state"] == state]
        if not rows:
            continue
        baseline[state] = {
            "state_name": state_cn,
            "images": len(rows),
            "r_mean": round(float(np.mean([r["r_mean"] for r in rows])), 4),
            "r_lo": round(float(np.mean([r["r_lo"] for r in rows])), 4),
            "r_hi": round(float(np.mean([r["r_hi"] for r in rows])), 4),
        }
    return baseline


def save_r_samples(path, samples, frame_shape):
    """保存每张图的 R 统计和按灯态汇总的基准；不写 hsv_samples.json。"""
    h, w = frame_shape[:2]
    data = {
        "camera": {
            "resolution": [w, h],
            "circle_center": [w // 2, h // 2],
            "circle_radius": CIRCLE_RADIUS,
            "percentiles": [PCT_LOW, PCT_HIGH],
        },
        "retinex": {
            "method": "SSR on V",
            "formula": "R = V / max(L, eps)",
            "scale": RETINEX_SCALE,
            "sigma_lowres": RETINEX_SIGMA,
            "eps": RETINEX_EPS,
        },
        "baseline": build_baseline(samples),
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
    print("模式: 1=R 基准采集, 2=螺丝采集; 空格采样, q 或 Esc 退出")
    print(f"R 基准保存到: {R_SAVE_PATH}")
    print(f"螺丝数据保存到: {SCREW_SAVE_PATH}")

    mode = "color"

    tasks = build_tasks()
    total = len(tasks)
    samples = []
    pos = 0
    last_state = None

    screw_scenes = SCENES
    screw_total = len(screw_scenes)
    screw_samples = []
    screw_pos = 0

    shown = None

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
        # 两种模式都用画面中心的圆圈采样，螺丝模式不做识别
        cv2.circle(preview, (int(cx * scale), int(cy * scale)),
                   int(CIRCLE_RADIUS * scale), (0, 255, 0), 2)

        if mode == "color":
            if pos < total:
                state, state_cn, repeat = tasks[pos]
                label = (f"R  {state.upper().replace('_', ' ')}  "
                         f"{repeat}/{REPEATS}  ({pos + 1}/{total})")
            else:
                label = "R  ALL DONE"
        else:
            if screw_pos < screw_total:
                scene, scene_cn = screw_scenes[screw_pos]
                label = f"SCREW  {scene.upper()}  ({screw_pos + 1}/{screw_total})"
            else:
                label = "SCREW  ALL DONE"

        # 窗口上显示当前该拍哪一组
        cv2.putText(preview, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # 任务变化时在终端提示一次，不每帧刷屏
        state_key = (mode, pos if mode == "color" else screw_pos)
        if state_key != shown:
            if mode == "color":
                if pos < total:
                    state, state_cn, repeat = tasks[pos]
                    if state != last_state:
                        print(f"\n=== 灯态: {state_cn}"
                              f"（保持该状态拍 {REPEATS} 张）===")
                        last_state = state
                    print(f"当前: {state_cn} 第 {repeat}/{REPEATS} 张"
                          "（空格采集 R，按 2 切到螺丝模式）")
                else:
                    print(f"\nR 基准采集完成（{total} 张）"
                          "，按 1/2 切换模式，q 退出")
            else:
                if screw_pos < screw_total:
                    scene, scene_cn = screw_scenes[screw_pos]
                    print(f"\n=== 螺丝采集: {scene_cn}"
                          "（空格采集，按 1 切回颜色模式）===")
                else:
                    print(f"\n螺丝采集完成（{screw_total} 组）"
                          "，按 1/2 切换模式，q 退出")
            shown = state_key

        cv2.imshow("color", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            mode = "color"
            shown = None
            print("\n切换到颜色采样模式")
        elif key == ord("2"):
            mode = "screw"
            shown = None
            print("\n切换到螺丝采集模式")
        elif key == ord(" ") and mode == "color" and pos < total:
            state, state_cn, repeat = tasks[pos]
            r = reflectance(frame)
            s = r_summary(r, (cx, cy), CIRCLE_RADIUS)
            print(f"\n已采集: {state_cn} 第 {repeat}/{REPEATS} 张"
                  f"（{s['count']} 个像素）")
            print(f"  R: mean={s['r_mean']:.4f}  "
                  f"5%={s['r_lo']:.4f}  95%={s['r_hi']:.4f}")

            samples.append({
                "state": state,
                "state_name": state_cn,
                "repeat": repeat,
                "pixels": s["count"],
                "r_mean": round(s["r_mean"], 4),
                "r_lo": round(s["r_lo"], 4),
                "r_hi": round(s["r_hi"], 4),
            })
            save_r_samples(R_SAVE_PATH, samples, frame.shape)
            print(f"  已保存: {R_SAVE_PATH}（已完成 {len(samples)}/{total}）")
            pos += 1
        elif key == ord(" ") and mode == "screw" and screw_pos < screw_total:
            scene, scene_cn = screw_scenes[screw_pos]
            hsv = sample_hsv(frame, (cx, cy), CIRCLE_RADIUS)
            s = hsv_summary(hsv)
            thr = threshold_array(s)
            print(f"\n已采集: {scene_cn}（{s['count']} 个像素）")
            print(f"  平均 HSV: [{s['mean_hsv'][0]:.1f}, "
                  f"{s['mean_hsv'][1]:.1f}, {s['mean_hsv'][2]:.1f}]")
            print(f"  阈值数组: [{thr[0]:.0f}, {thr[1]:.0f}, "
                  f"{thr[2]:.0f}, {thr[3]:.0f}, {thr[4]:.0f}, {thr[5]:.0f}]")
            if s["h_lo"] > s["h_hi"]:
                print("  注意: H 跨过 0，需要拆成两段阈值")

            screw_samples.append({
                "scene": scene,
                "scene_name": scene_cn,
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
            save_samples(SCREW_SAVE_PATH, screw_samples, frame.shape)
            print(f"  已保存: {SCREW_SAVE_PATH}"
                  f"（已完成 {len(screw_samples)}/{screw_total}）")
            screw_pos += 1
        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
