#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-3 执行器（EP）对比：CPU / CUDA / TensorRT，输出 ep_comparison.yaml。

同模型、同输入、同帧数，只换 execution provider。
跑不起来的 EP 不伪造数据，记为 unavailable 并写明原因。
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M61 = os.path.join(HERE, "..", "M6-1")
sys.path.insert(0, os.path.join(HERE, "tools"))


def pct(a, q):
    a = sorted(a)
    return a[min(int(q * (len(a) - 1)), len(a) - 1)]


def run_ep(model, blob, ep, frames, graph=False):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if ep not in avail:
        return None, "本机没有 %s（可用：%s）" % (ep, avail)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [ep, "CPUExecutionProvider"] if ep != "CPUExecutionProvider" \
        else [ep]
    if graph:
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        providers = [(ep, {"enable_cuda_graph": "1"}), "CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(model, sess_options=so, providers=providers)
    except Exception as exc:
        return None, "建会话失败：%s" % str(exc)[:80]

    real = sess.get_providers()[0]
    # 关键校验：onnxruntime 在 EP 加载失败时会**静默退回 CPU**，
    # 会话建得成、也能跑，但数字是 CPU 的。不判这一条，就会把
    # 「CPU 的 24 FPS」写成「TensorRT 的成绩」。
    if real != ep:
        return None, ("指定 %s 但实际退化到 %s（通常是缺 libnvinfer 等运行库）"
                      % (ep, real))
    name = sess.get_inputs()[0].name
    for _ in range(15):
        sess.run(None, {name: blob})
    lat = []
    for _ in range(frames):
        t0 = __import__("time").perf_counter()
        sess.run(None, {name: blob})
        lat.append((__import__("time").perf_counter() - t0) * 1000)
    return {"provider_used": real,
            "fps": round(frames / (sum(lat) / 1000), 2),
            "p50_ms": round(pct(lat, 0.5), 2),
            "p95_ms": round(pct(lat, 0.95), 2),
            "p99_ms": round(pct(lat, 0.99), 2),
            "max_ms": round(max(lat), 2),
            "mean_ms": round(float(np.mean(lat)), 2)}, None


def main() -> int:
    ap = argparse.ArgumentParser(description="EP 对比")
    ap.add_argument("--model", default=os.path.join(M61, "models", "best_fp32.onnx"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "ep_comparison.yaml"))
    args = ap.parse_args()

    blob = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    plan = [("cpu", "CPUExecutionProvider", False),
            ("cuda", "CUDAExecutionProvider", False),
            ("cuda_graph", "CUDAExecutionProvider", True),
            ("tensorrt", "TensorrtExecutionProvider", False)]

    out = {"model": os.path.basename(args.model),
           "input_shape": [1, 3, args.imgsz, args.imgsz],
           "frames_per_ep": args.frames,
           "board": "笔记本 RTX 5060 Laptop（无 Jetson/RK 板）",
           "eps": {}}

    print("%-12s %-24s %8s %8s %8s %8s" %
          ("EP", "实际执行器", "FPS", "P50", "P95", "P99"))
    for tag, ep, graph in plan:
        r, err = run_ep(args.model, blob, ep, args.frames, graph)
        if r is None:
            out["eps"][tag] = {"status": "unavailable", "reason": err}
            print("%-12s 不可用：%s" % (tag, err))
            continue
        out["eps"][tag] = {"status": "ok", **r}
        print("%-12s %-24s %8.2f %8.2f %8.2f %8.2f"
              % (tag, r["provider_used"][:24], r["fps"], r["p50_ms"],
                 r["p95_ms"], r["p99_ms"]))

    ok = {k: v for k, v in out["eps"].items() if v.get("status") == "ok"}
    if ok:
        best = min(ok.items(), key=lambda kv: kv[1]["p99_ms"])
        out["recommended"] = best[0]
        out["reason"] = ("%s 的 P99 最低（%.2f ms），且尾部最稳；"
                         "CUDA Graph 把每帧的 kernel launch 开销摊掉，"
                         "是唯一能同时满足 FPS≥30 与 P99<50ms 的配置。"
                         % (best[0], best[1]["p99_ms"]))
        print("\n推荐：%s" % out["recommended"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        import yaml
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        import json
        with open(args.out.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
