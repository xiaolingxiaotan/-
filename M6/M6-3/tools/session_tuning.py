#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-3 会话参数逐项实验：一次只改一个，量化每个参数的影响。

为什么要「一次只改一个」：SessionOptions 之间会互相影响（比如开了 CUDA Graph
之后 mem_pattern 必须关，否则捕获失败）。同时改三个，最后只能知道"组合有效"，
说不出哪个参数起了作用——题面要的正是「逐参数量化」。

输出 session_tuning.csv。
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M61 = os.path.join(HERE, "..", "M6-1")


def pct(a, q):
    a = sorted(a)
    return a[min(int(q * (len(a) - 1)), len(a) - 1)]


def measure(model, blob, providers, so_kw, frames):
    import onnxruntime as ort

    so = ort.SessionOptions()
    for k, v in so_kw.items():
        setattr(so, k, v)
    try:
        sess = ort.InferenceSession(model, sess_options=so, providers=providers)
    except Exception as exc:
        return None, "建会话失败: %s" % str(exc)[:80]
    name = sess.get_inputs()[0].name
    prov = sess.get_providers()[0]

    for _ in range(10):                     # 预热
        sess.run(None, {name: blob})
    lat = []
    try:
        for _ in range(frames):
            t0 = time.perf_counter()
            sess.run(None, {name: blob})
            lat.append((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return None, "推理失败: %s" % str(exc)[:80]
    return {"provider": prov, "mean": float(np.mean(lat)),
            "p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
            "p99": pct(lat, 0.99), "max": max(lat)}, None


def main() -> int:
    ap = argparse.ArgumentParser(description="SessionOptions 逐参数实验")
    ap.add_argument("--model", default=os.path.join(M61, "models", "best_fp32.onnx"))
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "session_tuning.csv"))
    args = ap.parse_args()

    blob = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if args.device == "cuda" else ["CPUExecutionProvider"])

    import onnxruntime as ort
    G = ort.GraphOptimizationLevel
    experiments = [
        ("baseline", {}),
        # ---- 线程数：CPU 上最有效的一项 ----
        ("intra_op=1", {"intra_op_num_threads": 1}),
        ("intra_op=2", {"intra_op_num_threads": 2}),
        ("intra_op=4", {"intra_op_num_threads": 4}),
        ("intra_op=8", {"intra_op_num_threads": 8}),
        # ---- 执行模式：并行 vs 顺序 ----
        ("exec=PARALLEL", {"execution_mode": ort.ExecutionMode.ORT_PARALLEL}),
        ("exec=SEQUENTIAL", {"execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL}),
        # ---- 图优化级别 ----
        ("graph=DISABLE", {"graph_optimization_level": G.ORT_DISABLE_ALL}),
        ("graph=BASIC", {"graph_optimization_level": G.ORT_ENABLE_BASIC}),
        ("graph=EXTENDED", {"graph_optimization_level": G.ORT_ENABLE_EXTENDED}),
        ("graph=ALL", {"graph_optimization_level": G.ORT_ENABLE_ALL}),
        # ---- 内存模式（CUDA Graph 必须关掉它）----
        ("mem_pattern=off", {"enable_mem_pattern": False}),
    ]
    if args.device == "cuda":
        experiments.append(("cuda_graph=on", {"enable_mem_pattern": False,
                                              "enable_cpu_mem_arena": False}))

    rows = []
    print("%-18s %-22s %8s %8s %8s %8s" %
          ("配置", "执行器", "P50", "P95", "P99", "max"))
    for tag, kw in experiments:
        if tag.startswith("cuda_graph"):
            # CUDA Graph 走 provider option，不是 SessionOptions
            import onnxruntime as ort2
            so = ort2.SessionOptions()
            so.enable_mem_pattern = False
            so.enable_cpu_mem_arena = False
            so.graph_optimization_level = ort2.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort2.InferenceSession(
                args.model, sess_options=so,
                providers=[("CUDAExecutionProvider",
                            {"enable_cuda_graph": "1"}),
                           "CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            for _ in range(10):
                sess.run(None, {name: blob})
            lat = []
            for _ in range(args.frames):
                t0 = time.perf_counter()
                sess.run(None, {name: blob})
                lat.append((time.perf_counter() - t0) * 1000)
            r = {"provider": sess.get_providers()[0], "mean": float(np.mean(lat)),
                 "p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
                 "p99": pct(lat, 0.99), "max": max(lat)}
            err = None
        else:
            r, err = measure(args.model, blob, provs, kw, args.frames)
        if r is None:
            print("%-18s %s" % (tag, err))
            rows.append({"config": tag, "error": err})
            continue
        print("%-18s %-22s %8.2f %8.2f %8.2f %8.2f"
              % (tag, r["provider"][:22], r["p50"], r["p95"], r["p99"], r["max"]))
        rows.append({"config": tag, "provider": r["provider"],
                     "p50_ms": round(r["p50"], 2), "p95_ms": round(r["p95"], 2),
                     "p99_ms": round(r["p99"], 2), "max_ms": round(r["max"], 2),
                     "mean_ms": round(r["mean"], 2)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    keys = ["config", "provider", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "max_ms", "error"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
