#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 推理：图片 / 视频 / 摄像头（题面 §7，§9 要求的 deploy_cpu.py）。

用法::

    python3 deploy_cpu.py --source data/raw/IMG_0001.jpg --out results/det.jpg
    python3 deploy_cpu.py --source data/raw                 # 整个目录，结果存 results/det/
    python3 deploy_cpu.py --source demo.mp4 --out results/demo_out.mp4
    python3 deploy_cpu.py --source 0                        # 摄像头 0，按 q 退出

输入类型自动判断：纯数字 = 摄像头序号；目录 = 批量图片；按扩展名分图片/视频。

默认用 models/best.onnx（不需要 torch，现场就用这个）；
没有 onnx 时自动退回 models/best.pt（需要 torch）。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train import Detector  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".flv")


def build_detector(args) -> Detector:
    """按参数建检测器。默认优先 ONNX：现场环境可以没有 torch。"""
    if args.backend == "auto":
        backend = "onnx" if os.path.exists(args.onnx) else "pt"
    else:
        backend = args.backend
    det = Detector(weights=args.weights if backend == "pt" else None,
                   onnx_path=args.onnx if backend == "onnx" else None,
                   input_size=args.size, conf_thres=args.conf,
                   nms_iou=args.iou, backend=backend, class_names=None)
    det.load()
    return det


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU 推理：图片 / 视频 / 摄像头")
    ap.add_argument("--source", required=True,
                    help="图片/视频路径、目录，或摄像头序号（0 表示默认摄像头）")
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "best.onnx"))
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best.pt"))
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "pt"])
    ap.add_argument("--conf", type=float, default=0.25, help="分数门槛，默认 0.25")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 阈值，默认 0.45")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", default=None,
                    help="输出路径：单图存图、视频存 mp4、目录模式下当输出目录")
    ap.add_argument("--show", action="store_true", help="弹窗显示（需要图形界面）")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    ap.add_argument("--limit", type=int, default=None,
                    help="目录模式下最多处理多少张图")
    ap.add_argument("--width", type=int, default=640, help="摄像头采集宽（默认 640）")
    ap.add_argument("--height", type=int, default=480, help="摄像头采集高（默认 480）")
    ap.add_argument("--fourcc", default="MJPG",
                    help="摄像头采集格式，默认 MJPG；绿屏时就是它没生效")
    ap.add_argument("--scale", type=float, default=1.6,
                    help="显示窗口放大倍数，默认 1.6（只放大显示，不改采集）")
    ap.add_argument("--exposure", type=int, default=-1,
                    help="摄像头曝光：-1 自动；>=0 锁定到该值（暗房间试 300，"
                         "明亮房间试 50~100）。自动曝光会让画面忽明忽暗")
    ap.add_argument("--smooth", type=int, default=3,
                    help="跨帧投票窗口，默认 3；设 1 关闭。用于压掉框的闪烁")
    args = ap.parse_args()

    det = build_detector(args)
    src = args.source

    # ---- 摄像头 ----
    if src.isdigit():
        det.run_camera(int(src), show=args.show or True,
                       max_frames=args.max_frames,
                       size=(args.width, args.height), fourcc=args.fourcc,
                       window_scale=args.scale, exposure=args.exposure,
                       smooth=args.smooth)
        return 0

    if not os.path.exists(src):
        print("找不到输入：%s" % src)
        return 1

    # ---- 目录：批量图片 ----
    if os.path.isdir(src):
        out_dir = args.out or os.path.join(HERE, "results", "det")
        os.makedirs(out_dir, exist_ok=True)
        names = [n for n in sorted(os.listdir(src))
                 if n.lower().endswith(IMG_EXT)]
        if args.limit:
            names = names[:args.limit]
        for i, n in enumerate(names, 1):
            det.run_image(os.path.join(src, n),
                          os.path.join(out_dir, n))
            print("  [%d/%d] %s" % (i, len(names), n))
        print("共处理 %d 张，结果在 %s" % (len(names), out_dir))
        return 0

    # ---- 单文件：按扩展名分流 ----
    ext = os.path.splitext(src)[1].lower()
    if ext in VID_EXT:
        out = args.out or os.path.join(HERE, "results", "video_out.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        det.run_video(src, out_path=out, show=args.show,
                      max_frames=args.max_frames)
        return 0

    out = args.out or os.path.join(HERE, "results", "det.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vis, dets = det.run_image(src, out_path=out, show=args.show)
    t = det.last_timing
    print("共 %d 个目标；预处理 %.1f ms / 推理 %.1f ms / 后处理 %.1f ms"
          % (len(dets), t["pre"] * 1000, t["infer"] * 1000, t["post"] * 1000))
    print("结果图：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
