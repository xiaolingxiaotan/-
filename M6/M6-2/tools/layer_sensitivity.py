#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-2 逐层量化敏感度：把某一层权重压到 INT8 再还原，看输出掉多少。

为什么用这个方法（而不是直接看量化模型）：
    本机 onnxruntime 1.23.2 的 `quantize_static` 在 YOLOv8 图上硬崩溃，
    `quantize_dynamic` 虽然能产出模型，但它用的是 `ConvInteger`，
    CPU EP 没有实现，跑不起来。所以「拿量化模型逐层对比」这条路在本机走不通。

    改成：对**浮点模型**的每一层卷积，单独做一次「权重 INT8 量化 → 反量化」，
    其余层保持浮点，比较输出与原始输出的 CosSim。
    这样得到的敏感度只归因到这一层，比整体量化后的误差更能说明问题——
    整体量化时误差会层层累积，反而看不出是哪层贡献的。

输出：results/layer_sensitivity.csv + results/layer_sensitivity.png
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M61 = os.path.join(HERE, "..", "M6-1")


def quantize_per_channel(w, nbits=8):
    """对称 per-channel 量化再反量化。w: [OC, IC, KH, KW]。"""
    oc = w.shape[0]
    flat = w.reshape(oc, -1)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    amax[amax == 0] = 1e-8
    qmax = 2 ** (nbits - 1) - 1
    scale = amax / qmax
    q = np.clip(np.round(flat / scale), -qmax - 1, qmax)
    return (q * scale).reshape(w.shape)


def cos_sim(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description="逐层 INT8 敏感度")
    ap.add_argument("--model", default=os.path.join(M61, "models",
                                                    "best_fp32_static.onnx"))
    ap.add_argument("--calib-dir", default=os.path.join(HERE, "..", "..", "M2",
                                                        "M2-3", "data", "raw"))
    ap.add_argument("--n-img", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    import cv2
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    sys.path.insert(0, os.path.join(HERE, "tools"))
    from quantize import CalibReader

    # 取几张验证图（与训练集独立）
    paths = []
    for s in ("val", "test"):
        p = os.path.join(HERE, "..", "..", "M2", "M2-3", "data", "splits",
                         s + ".txt")
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                n = ln.strip()
                if n:
                    paths.append(os.path.join(args.calib_dir, n))
    paths = [p for p in paths if os.path.exists(p)][:args.n_img]
    reader = CalibReader([], args.imgsz)
    blobs = [reader._load(p) for p in paths]
    print("验证图 %d 张" % len(blobs))

    def run(model_bytes):
        sess = ort.InferenceSession(model_bytes,
                                    providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return [sess.run(None, {name: b})[0] for b in blobs]

    base_model = onnx.load(args.model)
    # 先做一次 shape 推导，让 value_info 里有形状（不改变计算）
    try:
        base_model = onnx.shape_inference.infer_shapes(base_model)
    except Exception:
        pass
    base_bytes = base_model.SerializeToString()
    base_out = run(base_bytes)
    print("基准输出就绪")

    convs = [n for n in base_model.graph.node if n.op_type == "Conv"]
    print("卷积层 %d 个，逐个量化权重…" % len(convs))

    rows = []
    for idx, node in enumerate(convs):
        wname = node.input[1]
        init = next((t for t in base_model.graph.initializer if t.name == wname),
                    None)
        if init is None:
            continue
        orig = numpy_helper.to_array(init)
        if orig.ndim != 4:
            continue
        q = quantize_per_channel(orig)
        m = onnx.ModelProto()
        m.CopyFrom(base_model)
        for t in m.graph.initializer:
            if t.name == wname:
                new_t = numpy_helper.from_array(q.astype(orig.dtype), wname)
                t.CopyFrom(new_t)
                break
        try:
            outs = run(m.SerializeToString())
            cs = float(np.mean([cos_sim(a, b) for a, b in zip(base_out, outs)]))
            mad = float(np.mean([np.abs(a - b).max() for a, b in zip(base_out, outs)]))
        except Exception as exc:
            cs, mad = float("nan"), float("nan")
            print("  [跳过] %s：%s" % (wname, str(exc)[:50]))
        rows.append({"layer": wname, "op": node.op_type,
                     "params": int(orig.size), "cossim": round(cs, 6),
                     "max_abs_diff": round(mad, 5)})
        if (idx + 1) % 10 == 0:
            print("  已处理 %d/%d" % (idx + 1, len(convs)))

    rows.sort(key=lambda r: r["cossim"])
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "layer_sensitivity.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "op", "params", "cossim",
                                          "max_abs_diff"])
        w.writeheader()
        w.writerows(rows)

    print("\n最敏感（CosSim 最低）的 8 层：")
    for r in rows[:8]:
        print("  %-44s CosSim %.6f  maxΔ %.4f  %d 参数"
              % (r["layer"][:44], r["cossim"], r["max_abs_diff"], r["params"]))
    print("\n最不敏感的 5 层：")
    for r in rows[-5:]:
        print("  %-44s CosSim %.6f" % (r["layer"][:44], r["cossim"]))
    print("\n-> %s" % csv_path)

    # 热力图（条形）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = rows[:25]
        names = [r["layer"].split("/")[1] if "/" in r["layer"] else r["layer"]
                 for r in top]
        vals = [1 - r["cossim"] for r in top]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(range(len(top)), vals, color="#A32D2D")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("1 - CosSim（越大越敏感）")
        ax.set_title("逐层 INT8 权重敏感度 Top 25")
        fig.tight_layout()
        out = os.path.join(args.out_dir, "layer_sensitivity.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("图 -> %s" % out)
    except Exception as exc:
        print("[提示] 画图失败：%s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
