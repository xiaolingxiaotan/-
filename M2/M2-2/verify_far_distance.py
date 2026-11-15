"""verify_far_distance：远距离可检测性 + 拍摄台两套流程的合成图验证（只读）。

用 M2-1 的真实内参与棋盘格尺寸，按四个距离生成"正视棋盘格"合成图，
加轻微模糊/噪声模拟摄像头，再逐级跑主程序的 detect_corners / process_frame，
回答两个问题：
  1) 1.20 m 时棋盘格还剩多少像素、能不能检出；
  2) 卡住的是检测尺度还是重投影阈值、要不要开全分辨率兜底。

运行结果只打印，不写任何文件，不改动 pose_estimate.py 的常量。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_estimate as pe  # noqa: E402


DISTANCES = (0.30, 0.50, 0.80, 1.20)
BLURS = (0.0, 0.8, 1.4)
CONFIGS = (
    ("现状（兜底全关）", False, False),
    ("开全分辨率经典", True, False),
    ("全兜底（图片模式）", True, True),
)


def render_board(calib: pe.Calibration, distance: float, blur: float) -> np.ndarray:
    """render_board：按距离算每格像素，画一张 1920×1080 的正视棋盘格。"""
    square_px = calib.K[0, 0] * calib.square_m / distance
    step = max(2, int(round(square_px)))
    # 内角点 (cols, rows) 对应外格 (cols+1, rows+1)
    ncol, nrow = calib.cols + 1, calib.rows + 1
    board_w, board_h = ncol * step, nrow * step
    img = np.full((calib.height, calib.width), 128, dtype=np.uint8)
    x0 = (calib.width - board_w) // 2
    y0 = (calib.height - board_h) // 2
    for r in range(nrow):
        for c in range(ncol):
            color = 0 if (r + c) % 2 == 0 else 255  # 顶左黑格
            x1, y1 = x0 + c * step, y0 + r * step
            img[y1:y1 + step, x1:x1 + step] = color
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    rng = np.random.default_rng(20261115)
    noise = rng.normal(0.0, 3.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def render_board_realistic(
    calib: pe.Calibration,
    distance: float,
    tilt_deg: float,
    blur: float,
    noise_sigma: float,
    jpeg_quality: int,
    offset_x: float,
) -> np.ndarray:
    """render_board_realistic：带透视、径向畸变、JPEG 与噪声的近似实拍合成图。

    外格顶点按真实针孔模型投影（含 calib.dist），所以边缘位置的畸变是真的，
    不是人为缩放；比正视合成图更接近现场。
    """
    ncol, nrow = calib.cols + 1, calib.rows + 1
    s = calib.square_m
    verts = []
    for v in range(nrow + 1):
        for u in range(ncol + 1):
            verts.append(((u - ncol / 2.0) * s, (v - nrow / 2.0) * s, 0.0))
    obj = np.asarray(verts, dtype=np.float32).reshape(-1, 1, 3)
    rvec = np.array([[0.0], [np.radians(tilt_deg)], [0.0]], dtype=np.float64)
    tvec = np.array([offset_x, 0.0, distance], dtype=np.float64)
    imgpts, _ = cv2.projectPoints(obj, rvec, tvec, calib.K, calib.dist)
    imgpts = imgpts.reshape(nrow + 1, ncol + 1, 2)

    img = np.full((calib.height, calib.width), 128, dtype=np.uint8)
    for v in range(nrow):
        for u in range(ncol):
            quad = np.array(
                [imgpts[v, u], imgpts[v, u + 1],
                 imgpts[v + 1, u + 1], imgpts[v + 1, u]],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [quad], 0 if (u + v) % 2 == 0 else 255)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    rng = np.random.default_rng(7)
    img = np.clip(
        img.astype(np.float32)
        + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32),
        0, 255,
    ).astype(np.uint8)
    if jpeg_quality:
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if ok:
            img = cv2.cvtColor(
                cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY
            )
    return img


def main() -> int:
    calib = pe.load_calibration(pe.DEFAULT_CALIB, None)
    print("标定: %.1fx%.1f  fx=%.2f  内角点=%dx%d  方格=%.1f mm" % (
        calib.width, calib.height, calib.K[0, 0],
        calib.cols, calib.rows, calib.square_m * 1000))
    print("MAX_REPROJ_PX=%.1f  SB@%.1f / classic@%.1f / SB@%.1f" % (
        pe.MAX_REPROJ_PX, pe.SB_PRIMARY_SCALE,
        pe.CLASSIC_SCALE, pe.SB_FALLBACK_SCALE))
    print()
    print("距离  每格px  SB@0.4格px  内角点跨度px  整板px")
    for z in DISTANCES:
        sq = calib.K[0, 0] * calib.square_m / z
        span_w = (calib.cols - 1) * sq
        span_h = (calib.rows - 1) * sq
        print("%4.2fm  %6.2f  %10.2f  %5.0fx%-5.0f  %4.0fx%-4.0f" % (
            z, sq, sq * pe.SB_PRIMARY_SCALE, span_w, span_h,
            (calib.cols + 1) * sq, (calib.rows + 1) * sq))
    print()

    print("检测命中（每格 3 个模糊档：清晰 / 0.8 / 1.4）")
    print("%-22s %8s %8s %8s %8s" % ("配置", "0.30m", "0.50m", "0.80m", "1.20m"))
    for name, fc, fs in CONFIGS:
        cells = []
        for z in DISTANCES:
            hits = 0
            for blur in BLURS:
                gray = render_board(calib, z, blur)
                corners, detector = pe.detect_corners(
                    gray, calib.cols, calib.rows, fc, fs)
                if corners is not None and len(corners) == calib.cols * calib.rows:
                    hits += 1
            cells.append("%d/3" % hits)
        print("%-22s %8s %8s %8s %8s" % (name, *cells))
    print()

    print("1.20 m 完整 process_frame（带畸变，refine 开）")
    for name, fc, fs in CONFIGS:
        gray = render_board(calib, 1.20, 0.8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = pe.process_frame(
            frame, calib, True, True, fc, fs, True)
        if result.ok:
            print("  %-22s OK  detector=%-12s z=%.4f m  reproj=%.3f px" % (
                name, result.detector, result.distance, result.reproj_rmse))
        else:
            print("  %-22s NG  detector=%-12s reason=%s" % (
                name, result.detector, result.reason))
    print("严苛档：倾斜 18° + JPEG70 + 噪声 sigma6 + 模糊 1.2（带真实径向畸变）")
    cases = (
        ("0.80 m 中心", 0.80, 0.0),
        ("1.20 m 中心", 1.20, 0.0),
        ("1.20 m 右边缘", 1.20, 0.45),
    )
    for label, z, off in cases:
        gray = render_board_realistic(calib, z, 18.0, 1.2, 6.0, 70, off)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cells = []
        for name, fc, fs in CONFIGS:
            result = pe.process_frame(frame, calib, True, True, fc, fs, True)
            if result.ok:
                cells.append("%s OK %.2fpx" % (result.detector, result.reproj_rmse))
            else:
                cells.append("%s NG(%s)" % (result.detector, result.reason[:12]))
        print("  %-16s %s" % (label, "  |  ".join(cells)))
    print()
    print("结论读法：若「现状」NG 而「全兜底」OK，说明只该加检测兜底，"
          "不该放宽 MAX_REPROJ_PX；合成图不代替现场实测。")

    print()
    print("拍摄台两套流程自检（0.50 m 合成帧，只换 dist）")
    import capture_studio as cs
    gray = render_board_realistic(calib, 0.50, 0.0, 0.8, 3.0, 90, 0.0)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    on = cs.solve_with_distortion(frame, calib, True, True)
    off = cs.solve_without_distortion(frame, calib, True, True)
    pair_on, pair_off = cs.solve_pair(frame, calib, True, True)
    spec = cs.GROUP_BY_KEY["distort_030_center"]
    row = cs.build_row(spec, pair_on, pair_off, "off")
    print("  畸变流程   ok=%s  detector=%-10s z=%.4f m" % (
        on.ok, on.detector, on.distance))
    print("  非畸变流程 ok=%s  detector=%-10s z=%.4f m" % (
        off.ok, off.detector, off.distance))
    print("  同帧两套   dt=%.3f mm  droll=%+.3f deg" % (
        float(row["dt_mm"] or 0), float(row["droll_deg"] or 0)))
    print("  CSV        flow=%s  detected=%s  列数=%d  含flow=%s" % (
        row.get("flow"), row.get("detected"), len(cs.CSV_HEADER),
        "flow" in cs.CSV_HEADER))
    print("  流程选项   %s" % ", ".join(
        "%s=%s" % item for item in cs.FLOW_LABELS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
