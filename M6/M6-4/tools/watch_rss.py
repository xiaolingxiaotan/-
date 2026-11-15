#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RSS 采样：验证 3 分钟稳定运行无内存泄漏（题面 §2.5）。

用法：
    python tools/watch_rss.py --seconds 180 --out results/rss_curve.png
    python tools/watch_rss.py --pid 12345 --seconds 180     # 监控已有进程

为什么看 RSS 而不是「感觉不卡」：
    队列无界、缓冲不归还、每帧新建数组，这三类问题都不会立刻崩，
    表现只是 RSS 缓慢爬升。3 分钟足够看出趋势：稳态流水线的 RSS 应当
    在 ±50MB 内波动，斜着往上走就是漏了。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_rss(pid=None):
    try:
        import psutil
    except ImportError:
        return None
    p = psutil.Process(pid) if pid else psutil.Process()
    return p.memory_info().rss / 1048576


def main() -> int:
    ap = argparse.ArgumentParser(description="RSS 采样")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=20.0,
                    help="前多少秒算预热，不计入波动统计（导入/ORT 内存池在这一段长大）")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "rss_curve.png"))
    ap.add_argument("--run-pipeline", action="store_true",
                    help="在本进程里跑流水线（不监控外部进程）")
    args = ap.parse_args()

    if args.run_pipeline:
        # 监控自己：另起线程跑流水线，主线程采样
        import threading
        sys.path.insert(0, HERE)
        from pipeline.buffers import FrameBuffers
        from pipeline.engine import get_engine
        from pipeline.pipeline import Pipeline
        import numpy as np

        eng = get_engine(os.path.join(HERE, "..", "M6-1", "models",
                                      "best_fp32.onnx"))

        def gen():
            # 预分配 4 帧循环用：帧源本身不能成为内存波动的来源，
            # 否则测出来的曲线是「帧源的分配行为」而不是「管线的泄漏」。
            rng = np.random.default_rng(0)
            pool = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
                    for _ in range(4)]
            i = 0
            while True:
                yield i, pool[i % 4]
                i += 1

        pipe = Pipeline(eng, FrameBuffers(6, 640), gen(), mode="triple",
                        queue_size=4)
        th = threading.Thread(target=pipe.run, daemon=True)
        th.start()
        t0 = time.perf_counter()
        xs, ys = [], []
        while time.perf_counter() - t0 < args.seconds:
            xs.append(time.perf_counter() - t0)
            ys.append(sample_rss())
            time.sleep(args.interval)
        pipe.stop.set()
        th.join(timeout=5)
    else:
        t0 = time.perf_counter()
        xs, ys = [], []
        while time.perf_counter() - t0 < args.seconds:
            v = sample_rss(args.pid)
            if v is None:
                print("需要 psutil")
                return 1
            xs.append(time.perf_counter() - t0)
            ys.append(v)
            time.sleep(args.interval)

    if not ys:
        print("没采到数据")
        return 1

    # 只统计预热之后的稳态段：前 20 秒包含 import、ORT 内存池扩张、Python
    # 分配器预热，这一段上涨是正常的，不能算泄漏。
    steady = [(x, y) for x, y in zip(xs, ys) if x >= args.warmup]
    if len(steady) < 5:
        steady = list(zip(xs, ys))
    sx = [p[0] for p in steady]
    sy = [p[1] for p in steady]
    lo, hi = min(sy), max(sy)
    # 线性趋势斜率（MB/s）：稳态段应当接近 0；持续为正才是泄漏
    n = len(sx)
    mx, my = sum(sx) / n, sum(sy) / n
    denom = sum((a - mx) ** 2 for a in sx) or 1e-9
    slope = sum((a - mx) * (b - my) for a, b in zip(sx, sy)) / denom

    print("稳态段（%.0fs 之后）RSS：最小 %.1f / 最大 %.1f / 波动 %.1f MB"
          % (args.warmup, lo, hi, hi - lo))
    print("线性趋势：%.2f MB/s（3 分钟累计 %.1f MB）"
          % (slope, slope * (args.seconds - args.warmup)))
    ok = (hi - lo) < 50.0
    print("判定：%s" % ("无泄漏（波动 < 50MB）" if ok
                       else "疑似泄漏，先查队列是否无界、buffer 是否没归还"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, lw=1.2)
        ax.axvline(args.warmup, ls=":", c="tab:orange", lw=1,
                   label="warmup ends")
        ax.axhline(lo, ls="--", c="gray", lw=0.8)
        ax.axhline(hi, ls="--", c="gray", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("RSS (MB)")
        ax.set_title("RSS over %.0fs — steady fluctuation %.1f MB, slope %.2f MB/s"
                     % (args.seconds, hi - lo, slope))
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print("曲线 -> %s" % args.out)
    except Exception as exc:
        print("[提示] 画图失败：%s" % exc)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
