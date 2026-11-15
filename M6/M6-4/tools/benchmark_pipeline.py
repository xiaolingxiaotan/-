#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-4 三方案对比：串行 / 双线程 / 三线程（题面 §2.4）。

同一模型、同一输入源、同一台机器，只改「谁在哪个线程里跑」。
输出 pipeline_benchmark.yaml：三组指标 + 加速比 + 瓶颈分析。

用法：
    python tools/benchmark_pipeline.py --source ../M2-3/data/raw --seconds 30 \
        --device cpu --modes serial,dual,triple
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.buffers import FrameBuffers  # noqa: E402
from pipeline.engine import get_engine  # noqa: E402
from pipeline.pipeline import Pipeline, percentile  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_frames(source):
    """把「目录里的图」或「视频」统一成帧列表生成函数。"""
    if os.path.isdir(source):
        names = [n for n in sorted(os.listdir(source))
                 if n.lower().endswith(IMG_EXT)]
        if not names:
            raise SystemExit("目录里没有图片：%s" % source)
        import cv2
        cached = []
        for n in names:
            f = cv2.imread(os.path.join(source, n))
            if f is not None:
                cached.append(f)
        print("输入源：%d 张图（循环喂）" % len(cached))

        def gen(limit):
            i = 0
            while limit is None or i < limit:
                yield i, cached[i % len(cached)]
                i += 1
        return gen

    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    print("输入源：视频 %s（%d 帧）" % (os.path.basename(source), n))

    def gen(limit):
        c = cv2.VideoCapture(source)
        i = 0
        while limit is None or i < limit:
            ok, f = c.read()
            if not ok:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = c.read()
                if not ok:
                    break
            yield i, f
            i += 1
        c.release()
    return gen


def rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        return None


def cpu_per_core():
    try:
        import psutil
        p = psutil.Process()
        p.cpu_percent(interval=None)
        time.sleep(0.3)
        return [round(x, 1) for x in p.cpu_percent(interval=0.3, percpu=True)]
    except Exception:
        return []


def run_mode(mode, eng, gen, seconds, qsize, frames):
    buffers = FrameBuffers(n=6, size=640)
    pipe = Pipeline(eng, buffers, gen(frames), mode=mode, queue_size=qsize,
                    render=True)          # 后处理含绘制，与架构图一致
    t0 = time.perf_counter()
    snap = pipe.run()
    wall = time.perf_counter() - t0
    lat = snap["latencies"]
    st = snap["stage_ms"]
    return {
        "fps": round(snap["frames_out"] / max(wall, 1e-9), 2),
        "frames": snap["frames_out"],
        "latency_p50_ms": round(percentile(lat, 0.5), 2),
        "latency_p95_ms": round(percentile(lat, 0.95), 2),
        "rss_mb": rss_mb(),
        "cpu_per_core_pct": cpu_per_core(),
        "stage_mean_ms": {k: round(sum(v) / len(v), 2) for k, v in st.items() if v},
        "wall_s": round(wall, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-4 三方案对比")
    ap.add_argument("--model", default=os.path.join(HERE, "..", "M6-1",
                                                    "models", "best_fp32.onnx"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--modes", default="serial,dual,triple")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="不指定 --frames 时，按串行单帧耗时折算帧数")
    ap.add_argument("--frames", type=int, default=None,
                    help="每个方案固定跑多少帧（建议 >=300，帧数太少噪声大）")
    ap.add_argument("--queue-size", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "pipeline_benchmark.yaml"))
    args = ap.parse_args()

    prov = (("CUDAExecutionProvider",) if args.device == "gpu"
            else ("CPUExecutionProvider",))
    eng = get_engine(args.model, prov)
    print("引擎：%s（%s）" % (eng.provider, os.path.basename(args.model)))

    gen_fn = list_frames(args.source)

    # 先用串行测一次单帧耗时，据此折算「跑够 seconds 秒需要多少帧」
    probe = Pipeline(eng, FrameBuffers(2, 640), gen_fn(3), mode="serial",
                     queue_size=2)
    probe.run()
    per_frame = sum(probe.stats.snapshot()["stage_ms"]["pre"]
                    + probe.stats.snapshot()["stage_ms"]["infer"]
                    + probe.stats.snapshot()["stage_ms"]["post"]) / 3 / 1000
    frames = args.frames or max(20, int(args.seconds / max(per_frame, 1e-3)))
    print("单帧约 %.1f ms -> 每个方案跑 %d 帧" % (per_frame * 1000, frames))

    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        print("\n[%s] 运行中 ..." % mode)
        results[mode] = run_mode(mode, eng, gen_fn, args.seconds,
                                 args.queue_size, frames)
        r = results[mode]
        print("  FPS %.2f  P50 %.2f / P95 %.2f ms  段均值 %s"
              % (r["fps"], r["latency_p50_ms"], r["latency_p95_ms"],
                 r["stage_mean_ms"]))

    # ---- 加速比与瓶颈 ----
    base = results.get("serial", {}).get("fps")
    for k in results:
        results[k]["speedup_vs_serial"] = (
            round(results[k]["fps"] / base, 2) if base else None)

    triple = results.get("triple") or {}
    st = triple.get("stage_mean_ms") or {}
    bottleneck = max(st, key=st.get) if st else None
    reason = ("三级的平均耗时：%s；最慢的是 %s（%.2f ms），"
              "流水线的吞吐上限就是它——%.2f ms/帧 ≈ %.1f FPS"
              % (", ".join("%s %.2f ms" % (k, v) for k, v in st.items()),
                 bottleneck, st.get(bottleneck, 0), st.get(bottleneck, 0),
                 1000.0 / max(st.get(bottleneck, 1), 1e-9))) if st else ""

    doc = {
        "model": os.path.basename(args.model),
        "source": os.path.basename(args.source.rstrip("/")),
        "device": args.device,
        "provider": eng.provider,
        "queue_size": args.queue_size,
        "frames_per_mode": frames,
        "results": results,
        "speedup_triple_vs_serial": triple.get("speedup_vs_serial"),
        "speedup_dual_vs_serial": results.get("dual", {}).get("speedup_vs_serial"),
        "bottleneck_stage": bottleneck,
        "bottleneck_reason": reason,
        "buffer_evidence": "见 docs/zero_copy.md（allocations_total 恒为 2N）",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        import yaml
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        import json
        with open(args.out.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    print("\n加速比：dual %.2f× / triple %.2f×（串行 %.2f FPS 为基准）"
          % (results.get("dual", {}).get("speedup_vs_serial") or 0,
             triple.get("speedup_vs_serial") or 0, base or 0))
    print("瓶颈：%s —— %s" % (bottleneck, reason))
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
