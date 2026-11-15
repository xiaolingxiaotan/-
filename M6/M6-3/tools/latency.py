#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-3 延迟测量：1000 帧逐帧计时，输出直方图 + CDF + 统计表。

题面看的是**尾部**（P95 < 35ms、P99 < 50ms、最大 < 80ms），不是平均值。
平均值好看、偶尔飙到 80ms 的管线，在导航里就是「跟丢一次」。

用法：
    bash ../pycuda.sh tools/latency.py --device cuda --frames 1000
    python tools/latency.py --device cpu --frames 1000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
M61 = os.path.join(HERE, "..", "M6-1")
sys.path.insert(0, M61)

from infer import OnnxDetector  # noqa: E402


def pct(a, q):
    a = sorted(a)
    if not a:
        return 0.0
    return a[min(int(q * (len(a) - 1)), len(a) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-3 尾延迟测量")
    ap.add_argument("--model", default=os.path.join(M61, "models", "best_fp32.onnx"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "trt"])
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default=None, help="输出文件名后缀，如 cuda_graph")
    ap.add_argument("--graph", action="store_true", help="启用 CUDA Graph")
    args = ap.parse_args()

    import cv2
    if args.source:
        frame = cv2.imread(args.source)
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)

    dev = {"cpu": "cpu", "cuda": "gpu", "trt": "gpu"}[args.device]
    if args.device == "trt":
        # TensorRT EP：没装 libnvinfer 时会退化，这里显式构造以便暴露问题
        import onnxruntime as ort
        print("可用执行器:", ort.get_available_providers())
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            print("TensorRT EP 不可用（未安装 TensorRT 运行库）")
            return 3

    det = OnnxDetector(args.model, imgsz=args.imgsz, device=dev)
    if args.device == "trt":
        import onnxruntime as ort
        so = ort.SessionOptions()
        det.sess = ort.InferenceSession(
            args.model, sess_options=so,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider",
                       "CPUExecutionProvider"])
        det.provider = det.sess.get_providers()[0]
        print("[引擎] TensorRT ->", det.provider)

    # 预热（不算进统计）
    for _ in range(args.warmup):
        det.detect(frame)
    cold_note = "已排除 %d 帧预热" % args.warmup

    lat = []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        det.detect(frame)
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat = np.asarray(lat)
    stats = {
        "device": args.device,
        "provider": det.provider,
        "frames": args.frames,
        "warmup": args.warmup,
        "fps": round(args.frames / (lat.sum() / 1000.0), 2),
        "mean_ms": round(float(lat.mean()), 2),
        "p50_ms": round(pct(lat, 0.50), 2),
        "p95_ms": round(pct(lat, 0.95), 2),
        "p99_ms": round(pct(lat, 0.99), 2),
        "max_ms": round(float(lat.max()), 2),
        "min_ms": round(float(lat.min()), 2),
        "std_ms": round(float(lat.std()), 2),
        "pass_p95_35": bool(pct(lat, 0.95) < 35),
        "pass_p99_50": bool(pct(lat, 0.99) < 50),
        "pass_max_80": bool(lat.max() < 80),
        "pass_fps_30": bool(args.frames / (lat.sum() / 1000.0) >= 30),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.device
    with open(os.path.join(args.out_dir, "latency_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({**stats, "samples_ms": [round(float(v), 3) for v in lat]},
                  f, indent=1)

    print("\n[%s] %s，%d 帧（%s）" % (args.device, det.provider, args.frames,
                                    cold_note))
    print("  FPS      %.2f" % stats["fps"])
    print("  均值/P50 %.2f / %.2f ms" % (stats["mean_ms"], stats["p50_ms"]))
    print("  P95/P99  %.2f / %.2f ms   （阈值 35 / 50）" % (stats["p95_ms"],
                                                          stats["p99_ms"]))
    print("  最大      %.2f ms        （阈值 80）" % stats["max_ms"])

    # 直方图 + CDF
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(lat, bins=60, color="#378ADD", alpha=0.85)
        for q, c in ((0.95, "#A32D2D"), (0.99, "#BA7517")):
            axes[0].axvline(pct(lat, q), ls="--", c=c, lw=1.2,
                            label="P%d %.1f ms" % (q * 100, pct(lat, q)))
        axes[0].set_xlabel("latency (ms)")
        axes[0].set_ylabel("frames")
        axes[0].set_title("Latency histogram — %s" % args.device)
        axes[0].legend(fontsize=8)

        s = np.sort(lat)
        cdf = np.arange(1, len(s) + 1) / len(s)
        axes[1].plot(s, cdf, lw=1.4)
        for q, c in ((0.95, "#A32D2D"), (0.99, "#BA7517")):
            axes[1].axvline(pct(lat, q), ls="--", c=c, lw=1.2,
                            label="P%d" % (q * 100))
        axes[1].axhline(0.95, ls=":", c="gray", lw=0.8)
        axes[1].set_xlabel("latency (ms)")
        axes[1].set_ylabel("CDF")
        axes[1].set_title("CDF — tail is what matters")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "latency_%s.png" % tag)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("  图 -> %s" % out)
    except Exception as exc:
        print("  [提示] 画图失败：%s" % exc)

    ok = all([stats["pass_p95_35"], stats["pass_p99_50"],
              stats["pass_max_80"], stats["pass_fps_30"]])
    print("  达标：%s" % ("是" if ok else
                        "否（P95 %s / P99 %s / max %s / FPS %s）"
                        % (stats["pass_p95_35"], stats["pass_p99_50"],
                           stats["pass_max_80"], stats["pass_fps_30"])))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
