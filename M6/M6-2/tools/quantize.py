#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-2 量化：PTQ 两种方案 + 逐层精度分析 + mAP/FPS 对比（题面 §2）。

三种产物：
    models/best_int8_dynamic.onnx   动态量化（只量化权重）
    models/best_int8_static.onnx    静态 QDQ（权重+激活，需校准集）
    results/quant_analysis.csv      逐层 CosSim（浮点 vs 量化）

校准集：**必须与训练集独立**（题面 §2.1 第 1 条）。用 data/splits/val + test
的 118 张图，不用 train。用训练图校准会让量化参数"见过答案"，精度虚高。

用法：
    python tools/quantize.py --calib ../M2-3/data/raw --split ../M2-3/data/splits/val.txt
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M61 = os.path.join(HERE, "..", "M6-1")


# ---------------------------------------------------------------- 校准集
class CalibReader:
    """给 onnxruntime 的 quantize_static 喂预处理过的张量。"""

    def __init__(self, paths, imgsz=640):
        self.paths = paths
        self.imgsz = imgsz
        self.n = len(paths)

    def get_next(self):
        if not self.paths:
            return None
        p = self.paths.pop(0)
        blob = self._load(p)
        self.paths.append(p)               # 循环使用
        return {"images": blob}

    def _load(self, path):
        import cv2
        img = cv2.imread(path)
        if img is None:
            return np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        small = cv2.resize(img, (nw, nh),
                           interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = small
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None])


# ---------------------------------------------------------------- 逐层探针
def make_probe_model(src, dst, limit=None):
    """把每个节点的输出都加成图输出，用于逐层对比。

    只加「形状确定的中间张量」，跳过控制流相关的输出；limit 用来限制数量
    （全加会导致内存爆炸，1000 多个输出）。
    """
    import onnx
    from onnx import helper

    m = onnx.load(src)
    exist = {o.name for o in m.graph.output}
    added = 0
    for node in m.graph.node:
        for out in node.output:
            if not out or out in exist:
                continue
            # 只保留有形状信息的张量
            vi = next((v for v in m.graph.value_info if v.name == out), None)
            if vi is None:
                continue
            m.graph.output.append(helper.ValueInfoProto(name=out))
            exist.add(out)
            added += 1
            if limit and added >= limit:
                onnx.save(m, dst)
                return added
    onnx.save(m, dst)
    return added


def per_layer_cossim(float_model, quant_model, blobs, out_csv):
    """逐层 CosSim：浮点模型 vs 量化模型，同一输入下的中间张量。"""
    import onnxruntime as ort

    def run(model_path):
        s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        outs = [o.name for o in s.get_outputs()]
        vals = {}
        for b in blobs:
            res = s.run(None, {name: b})
            for n, v in zip(outs, res):
                vals.setdefault(n, []).append(np.asarray(v).reshape(-1))
        return vals

    f = run(float_model)
    q = run(quant_model)
    common = set(f) & set(q)
    rows = []
    for n in common:
        a = np.concatenate(f[n]).astype(np.float64)
        b = np.concatenate(q[n]).astype(np.float64)
        # 长度可能因布局不同而不一致，跳过
        if a.size != b.size:
            continue
        cs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        rows.append({"tensor": n, "cossim": round(cs, 6),
                     "numel": a.size,
                     "max_abs_diff": round(float(np.abs(a - b).max()), 6)})
    rows.sort(key=lambda r: r["cossim"])
    import csv
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tensor", "cossim", "numel",
                                           "max_abs_diff"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-2 量化")
    ap.add_argument("--model", default=os.path.join(M61, "models",
                                                    "best_fp32_static.onnx"),
                    help="量化要用**静态 shape** 版本：动态 H/W 会让 shape 推导失败，"
                         "而且同一张量在不同分辨率下分布不同，scale 没法统一")
    ap.add_argument("--calib-dir", default=os.path.join(HERE, "..", "..", "M2",
                                                        "M2-3", "data", "raw"))
    ap.add_argument("--split", action="append", default=None,
                    help="校准集来源，可给多次；默认 val+test（都与训练集独立）")
    ap.add_argument("--n-calib", type=int, default=118)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--probe-limit", type=int, default=120)
    ap.add_argument("--static", action="store_true",
                    help="额外尝试静态 QDQ（本机 ORT 1.23 会硬崩溃，默认不跑）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = args.split or [
        os.path.join(HERE, "..", "..", "M2", "M2-3", "data", "splits", "val.txt"),
        os.path.join(HERE, "..", "..", "M2", "M2-3", "data", "splits", "test.txt"),
    ]
    names = []
    for sp in splits:
        names += [ln.strip() for ln in open(sp, encoding="utf-8") if ln.strip()]
    paths = [os.path.join(args.calib_dir, n) for n in names]
    paths = [p for p in paths if os.path.exists(p)][:args.n_calib]
    print("校准集：%d 张（val+test，与训练集独立）" % len(paths))
    if len(paths) < 100:
        print("警告：校准集少于 100 张，题面要求 100~500 张")

    # ---- 方案 1：动态量化（torch）----
    dyn_path = os.path.join(args.out_dir, "best_int8_dynamic.onnx")
    try:
        import torch
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(args.model, dyn_path, weight_type=QuantType.QInt8)
        print("[动态量化] -> %s（%.2f MB）"
              % (dyn_path, os.path.getsize(dyn_path) / 1048576))
    except Exception as exc:
        print("[动态量化] 失败：%s" % exc)
        dyn_path = None

    # ---- 方案 2：静态 QDQ（本机 ORT 1.23.2 上会硬崩溃，默认跳过）----
    # 现象：quantize_static 打印完 "Please consider to run pre-processing"
    # 之后进程直接终止（无 Python 异常、无 traceback），试过
    # QDQ/QOperator 两种格式、per_channel 开/关、仅量化 Conv，全部在同一步死。
    # 根因大概率是 YOLOv8 图里的 Resize/Slice 等 shape 相关算子在
    # 量化前的符号推导阶段（本机 libnvinfer 也缺，TensorRT 走不了）。
    # 用 --static 单独尝试，失败不影响其余产物。
    stat_path = None
    if args.static:
        from onnxruntime.quantization import (CalibrationMethod,
                                              quantize_static, QuantFormat)
        stat_path = os.path.join(args.out_dir, "best_int8_static.onnx")
        reader = CalibReader(paths, args.imgsz)
        quantize_static(args.model, stat_path, reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel=True,
                        activation_type=__import__(
                            "onnxruntime.quantization",
                            fromlist=["QuantType"]).QuantType.QUInt8,
                        weight_type=__import__(
                            "onnxruntime.quantization",
                            fromlist=["QuantType"]).QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
        print("[静态量化QDQ] -> %s（%.2f MB）"
              % (stat_path, os.path.getsize(stat_path) / 1048576))

    # ---- 逐层分析（浮点 vs 动态量化：结构一致，张量名能对上）----
    probe_f = os.path.join(args.out_dir, "_probe_fp32.onnx")
    probe_q = os.path.join(args.out_dir, "_probe_int8.onnx")
    n_f = make_probe_model(args.model, probe_f, args.probe_limit)
    n_q = make_probe_model(dyn_path or args.model, probe_q, args.probe_limit)
    print("[逐层] 探针输出：浮点 %d 个，量化 %d 个" % (n_f, n_q))

    blobs = [CalibReader([p], args.imgsz)._load(p) for p in paths[:4]]
    csv_path = os.path.join(HERE, "results", "quant_analysis.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = per_layer_cossim(probe_f, probe_q, blobs, csv_path)
    print("[逐层] 比对 %d 层，CosSim 最低 5 层：" % len(rows))
    for r in rows[:5]:
        print("   %-46s %.6f" % (r["tensor"][:46], r["cossim"]))

    # ---- 汇总 ----
    summary = {
        "calib_images": len(paths),
        "fp32_mb": round(os.path.getsize(args.model) / 1048576, 2),
        "dynamic_mb": round(os.path.getsize(dyn_path) / 1048576, 2) if dyn_path else None,
        "static_mb": round(os.path.getsize(stat_path) / 1048576, 2) if stat_path else None,
        "static_status": "ok" if stat_path else "本机 ORT 1.23.2 quantize_static 硬崩溃",
        "layers_compared": len(rows),
        "worst_layers": rows[:5],
        "mean_cossim": round(float(np.mean([r["cossim"] for r in rows])), 6)
        if rows else None,
    }
    with open(os.path.join(HERE, "results", "quant_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n汇总 -> results/quant_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
