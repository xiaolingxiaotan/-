#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-1 基准测试：连续跑满 60 秒，输出六项指标到 yaml（题面 §2.3）。

为什么必须跑满 60 秒（题面追问第 1 条）：
    * 冷启动：第一次推理要建内存池、JIT/图优化、加载权重到显存
    * 频率爬坡：CPU/GPU 从低频升到高频需要几十秒，前几秒的数偏低
    * 缓存命中：连续的相同尺寸输入会把数据留在 cache 里，短测会高估
  只跑几帧报出来的 FPS，和 60 秒稳态能差一倍。

用法：
    python tools/benchmark.py --model models/best_fp32.onnx --device cpu --seconds 60
    python tools/benchmark.py --model models/best_fp32.onnx --device gpu --seconds 60
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from infer import OnnxDetector, preprocess, postprocess  # noqa: E402


def mem_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        return None


def cpu_pct(sec=0.2):
    try:
        import psutil
        return round(psutil.Process().cpu_percent(interval=sec), 1)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-1 60 秒基准")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "best_fp32.onnx"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "npu"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--source", default=None,
                    help="一张图（循环喂）；不传就合成 640x480 的随机帧")
    ap.add_argument("--board", default=None, help="板子/机器名，写进 yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    if args.source:
        frame = cv2.imread(args.source)
        if frame is None:
            print("读不到图：%s" % args.source)
            return 1
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    if args.device == "npu":
        print("NPU：本机不具备该执行器，按题面 §2.3 记为空并说明原因")
        return 0

    det = OnnxDetector(args.model, imgsz=args.imgsz, device=args.device)

    # ---- 冷启动单独记一次，不混进统计 ----
    t_cold0 = time.perf_counter()
    det.detect(frame)
    cold_ms = (time.perf_counter() - t_cold0) * 1000
    print("[冷启动] 第 1 帧 %.1f ms（已排除在下面的统计之外）" % cold_ms)

    lat = []
    t_end = time.perf_counter() + args.seconds
    n = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        blob, meta = preprocess(frame, args.imgsz)
        out = det.sess.run(None, {det.in_name: blob})[0]
        postprocess(out, meta, det.conf, det.iou)
        lat.append((time.perf_counter() - t0) * 1000)
        n += 1

    lat_sorted = sorted(lat)
    fps = n / args.seconds
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]

    info = {
        "board": args.board or "本机（笔记本 RTX 5060 + CPU）",
        "model": os.path.basename(args.model),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "mode": args.device,
        "provider": det.provider,
        "seconds": args.seconds,
        "frames": n,
        "fps": round(fps, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_cold_ms": round(cold_ms, 2),
        "mem_mb": mem_mb(),
        "cpu_pct": cpu_pct(),
        "model_size_mb": round(os.path.getsize(args.model) / 1048576, 2),
    }
    out = args.out or os.path.join(
        HERE, "results", "benchmark_%s_%s.yaml" % (args.device, time.strftime("%m%d")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        import yaml
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(out.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n模式 %s（%s）跑 %.0f 秒，共 %d 帧"
          % (args.device, det.provider, args.seconds, n))
    print("  FPS        %.2f" % fps)
    print("  P50 / P95  %.2f / %.2f ms" % (p50, p95))
    print("  冷启动     %.1f ms（是稳态的 %.1f 倍）"
          % (cold_ms, cold_ms / max(p50, 1e-6)))
    print("  内存       %s MB" % info["mem_mb"])
    print("  CPU        %s %%" % info["cpu_pct"])
    print("  模型大小   %.2f MB" % info["model_size_mb"])
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
