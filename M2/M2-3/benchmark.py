#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""耗时测量：三段计时 + CSV + 分布图（题面 §6.2，§9 要求的 benchmark.py）。

用法::

    python3 benchmark.py                                  # 用 data/raw 里第一张图
    python3 benchmark.py --source data/raw/IMG_0001.jpg --frames 300
    python3 benchmark.py --source demo.mp4 --frames 200

为什么分三段测：
    题面的考察重心是「部署后耗时花在哪，你知不知道」。只报一个总 FPS 回答
    不了这个问题——总慢可能是 resize 慢、可能是模型慢、也可能是 NMS 慢，
    三者的改法完全不同。

为什么报 P50 / P95：
    均值会把卡顿藏起来。P95 才是「演示时会不会突然卡一下」的答案。

为什么先预热再计时：
    前几帧包含内存分配、算子首次编译、缓存未命中，跑出来的数偏高。丢掉它们
    才是稳态性能。
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train import Detector  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".flv")


def pick_source(src: str) -> str:
    """把「目录 / 空」解析成一张具体的图。"""
    if src and os.path.isdir(src):
        for e in IMG_EXT:
            hits = sorted(glob.glob(os.path.join(src, "*" + e)))
            if hits:
                return hits[0]
        raise FileNotFoundError("%s 下没有图片" % src)
    if src and os.path.exists(src):
        return src
    for e in IMG_EXT:
        hits = sorted(glob.glob(os.path.join(HERE, "data", "raw", "*" + e)))
        if hits:
            return hits[0]
    raise FileNotFoundError("没有可用素材，请用 --source 指定一张图或一段视频")


def percentile(vals, q: float) -> float:
    """线性插值分位数（不依赖 numpy 的写法也能读）。"""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU 耗时测量")
    ap.add_argument("--source", default="", help="图片或视频；目录则取第一张图")
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "best.onnx"))
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best.pt"))
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "pt"])
    ap.add_argument("--frames", type=int, default=200, help="计时帧数，默认 200")
    ap.add_argument("--warmup", type=int, default=5, help="预热帧数，默认 5")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "onnx" if os.path.exists(args.onnx) else "pt"
    det = Detector(weights=args.weights if backend == "pt" else None,
                   onnx_path=args.onnx if backend == "onnx" else None,
                   input_size=args.size, conf_thres=args.conf, backend=backend)
    det.load()
    print("后端：%s" % backend)

    src = pick_source(args.source)
    is_video = os.path.splitext(src)[1].lower() in VID_EXT
    print("素材：%s（%s）" % (src, "视频" if is_video else "图片"))

    # ---- 准备帧来源 ----
    cap = None
    if is_video:
        import cv2
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("打不开视频：%s" % src)
            return 1

    from PIL import Image
    import numpy as np

    def next_frame():
        """取一帧 RGB；图片模式每次都重新读盘（端到端含读取）。"""
        if is_video:
            import cv2
            ok, bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = cap.read()
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with Image.open(src) as im:
            return np.array(im.convert("RGB"))

    # ---- 预热 ----
    for _ in range(max(args.warmup, 0)):
        det.detect(next_frame())

    # ---- 计时 ----
    t_read, t_pre, t_inf, t_post, t_all = [], [], [], [], []
    n_det = 0
    for _ in range(args.frames):
        a = time.perf_counter()
        frame = next_frame()
        b = time.perf_counter()
        blob, geom = det.preprocess(frame)
        c = time.perf_counter()
        raws = det.infer(blob)
        d = time.perf_counter()
        out = det.postprocess(raws, geom)
        e = time.perf_counter()
        t_read.append(b - a)
        t_pre.append(c - b)
        t_inf.append(d - c)
        t_post.append(e - d)
        t_all.append(e - a)
        n_det += len(out)

    if cap is not None:
        cap.release()

    mean_all = sum(t_all) / len(t_all)
    rows = [
        ("读帧/解码", t_read),
        ("预处理", t_pre),
        ("模型推理", t_inf),
        ("后处理", t_post),
    ]
    total = sum(sum(v) for _, v in rows)

    # ---- 打印 + CSV ----
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "benchmark.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("stage,mean_ms,p50_ms,p95_ms,share_percent\n")
        print("\n%-12s %10s %10s %10s %8s" % ("阶段", "均值ms", "P50", "P95", "占比"))
        for name, vals in rows:
            m = sum(vals) / len(vals) * 1000
            share = (sum(vals) / total * 100) if total else 0
            fh.write("%s,%.3f,%.3f,%.3f,%.2f\n"
                     % (name, m, percentile(vals, 0.5) * 1000,
                        percentile(vals, 0.95) * 1000, share))
            print("%-12s %10.2f %10.2f %10.2f %7.1f%%"
                  % (name, m, percentile(vals, 0.5) * 1000,
                     percentile(vals, 0.95) * 1000, share))
        fh.write("end_to_end,%.3f,%.3f,%.3f,100.00\n"
                 % (mean_all * 1000, percentile(t_all, 0.5) * 1000,
                    percentile(t_all, 0.95) * 1000))
        print("%-12s %10.2f %10.2f %10.2f %7.1f%%"
              % ("端到端", mean_all * 1000, percentile(t_all, 0.5) * 1000,
                 percentile(t_all, 0.95) * 1000, 100.0))
        print("\n帧数 %d（预热 %d），平均每帧检出 %.1f 个目标"
              % (args.frames, args.warmup, n_det / max(args.frames, 1)))
        print("端到端 FPS = %.2f（题面要求 CPU 上 >= 10）" % (1.0 / mean_all))
        print("CSV -> %s" % csv_path)

    # ---- 分布图：左柱状（各阶段均值）+ 右饼图（占比） ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r[0] for r in rows] + ["端到端"]
        means = [sum(v) / len(v) * 1000 for _, v in rows] + [mean_all * 1000]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].barh(names, means, color=["#888780", "#378ADD", "#1D9E75", "#BA7517", "#A32D2D"])
        axes[0].set_xlabel("ms")
        axes[0].set_title("Stage latency (mean)")
        for i, v in enumerate(means):
            axes[0].text(v, i, " %.1f" % v, va="center")
        shares = [sum(v) / total * 100 for _, v in rows]
        axes[1].pie(shares, labels=names[:-1], autopct="%1.1f%%",
                    colors=["#B4B2A9", "#85B7EB", "#5DCAA5", "#EF9F27"])
        axes[1].set_title("Latency share")
        fig.tight_layout()
        png = os.path.join(args.out_dir, "benchmark.png")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print("分布图 -> %s" % png)
    except Exception as exc:      # 画图失败不该让测量结果作废
        print("[提示] 画图失败（%s），CSV 仍然可用" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
