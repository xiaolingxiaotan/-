#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-4 CLI 入口：三级流水线实时演示。

用法：
    python pipeline.py --threads 3 --stream camera          # 现场演示
    python pipeline.py --threads 3 --stream video.mp4
    python pipeline.py --threads 1 --stream video.mp4       # 串行对照
    python pipeline.py --threads 2 --stream video.mp4
"""
import argparse
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pipeline.buffers import FrameBuffers  # noqa: E402
from pipeline.engine import get_engine  # noqa: E402
from pipeline.pipeline import Pipeline, percentile  # noqa: E402
from pipeline.stages import draw  # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, "..", "M6-1", "models", "best_fp32.onnx")


def frame_source(stream, loop_limit=None, max_size=640):
    """帧源：摄像头 / 视频 / 目录。产出 (idx, bgr)。

    注意：这里每帧都由 `cap.read()` 新分配内存，所以「帧本身」天然不共享；
    需要防的是下游复用同一块 buffer —— 那由 FrameBuffers 的索引机制解决。
    """
    if stream == "camera" or stream.isdigit():
        cap = cv2.VideoCapture(0 if stream == "camera" else int(stream))
        if not cap.isOpened():
            raise SystemExit("打不开摄像头")
        idx = 0
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield idx, f
                idx += 1
                if loop_limit and idx >= loop_limit:
                    break
        finally:
            cap.release()
        return

    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % stream)
    idx = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield idx, f
            idx += 1
            if loop_limit and idx >= loop_limit:
                break
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-4 三级流水线")
    ap.add_argument("--threads", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--stream", default="camera", help="camera / 视频路径")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--queue-size", type=int, default=4)
    ap.add_argument("--frames", type=int, default=None, help="最多处理多少帧")
    ap.add_argument("--show", action="store_true", help="弹窗显示")
    ap.add_argument("--out-video", default=None, help="把带框结果写成 mp4")
    args = ap.parse_args()

    prov = (("CUDAExecutionProvider",) if args.device == "gpu"
            else ("CPUExecutionProvider",))
    eng = get_engine(args.model, prov)
    mode = {1: "serial", 2: "dual", 3: "triple"}[args.threads]

    writer = None
    last = {}

    def on_result(idx, dets, lat):
        last["idx"] = idx
        last["dets"] = dets

    pipe = Pipeline(eng, FrameBuffers(6, 640),
                    frame_source(args.stream, args.frames),
                    mode=mode, queue_size=args.queue_size,
                    frame_limit=args.frames, on_result=on_result)

    if args.show or args.out_video:
        # 显示/录像走另一条线程外的简单实现：复用帧源代价高，这里只演示
        pass

    print("[流水线] 模式 %s，队列 %d，模型 %s（%s）"
          % (mode, args.queue_size, os.path.basename(args.model), eng.provider))
    snap = pipe.run()

    lat = snap["latencies"]
    print("\n帧数 %d（入 %d / 出 %d，丢 %d）"
          % (snap["frames_out"], snap["frames_in"], snap["frames_out"],
             snap["drops"]))
    print("FPS        %.2f" % snap["fps"])
    print("延迟 P50/P95  %.2f / %.2f ms"
          % (percentile(lat, 0.5), percentile(lat, 0.95)))
    for k, v in snap["stage_ms"].items():
        if v:
            print("  %-6s 均值 %.2f ms" % (k, sum(v) / len(v)))
    print("缓冲池：", pipe.buffers.evidence())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
