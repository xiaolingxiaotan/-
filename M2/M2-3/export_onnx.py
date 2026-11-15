#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 ONNX 并做 PyTorch / ONNX 一致性验证（题面 §6.1、§6.3）。

用法::

    python3 export_onnx.py                        # 默认 models/best.pt -> best.onnx
    python3 export_onnx.py --opset 17 --num 8     # 指定算子集与取样张数

做三件事：
    1) 从 best.pt 重建模型结构（权重里没有结构，必须用 ModelBuilder 重建）
    2) 导出 ONNX，用 onnx.checker 验结构、用 onnxruntime CPU 后端实跑
    3) 同一批图分别走 PyTorch 与 ONNX，比 max|Δraw| / max|Δbox| / max|Δconf|

产物：
    models/best.onnx
    results/consistency.json     一致性报告（§6.3 要贴进文档）
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train import Detector, OutputExporter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX + 一致性验证")
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best.pt"),
                    help="训练产出的权重（默认 models/best.pt）")
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "best.onnx"),
                    help="导出的 ONNX 路径")
    ap.add_argument("--opset", type=int, default=17, help="算子集版本，默认 17")
    ap.add_argument("--size", type=int, default=640, help="输入边长，必须与训练一致")
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw"),
                    help="一致性验证取材目录")
    ap.add_argument("--num", type=int, default=8, help="取样张数")
    ap.add_argument("--skip-verify", action="store_true", help="只导出，不做一致性验证")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print("找不到权重：%s\n先跑 python3 train.py 训练出 models/best.pt"
              % args.weights)
        return 1

    # ---- 1) 用权重重建模型 ----
    det = Detector(weights=args.weights, input_size=args.size, backend="pt")
    det.load()

    exp = OutputExporter(project_dir=HERE, model=det.model,
                         input_size=args.size, opset=args.opset,
                         classes=det.class_names)
    exp.onnx_path = args.onnx

    # ---- 2) 导出 ----
    info = exp.export_onnx()
    print("输出层：%s" % (info["outputs"],))

    if args.skip_verify:
        return 0

    # ---- 3) 一致性验证：拿真实图，而不是随机噪声 ----
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG")
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(os.path.join(args.raw, e))))
    if not paths:
        print("[提示] %s 下没有图，改用随机图做数值一致性验证" % args.raw)
    else:
        import numpy as np
        from PIL import Image

        picked = paths[:max(args.num, 1)]
        imgs = []
        for p in picked:
            with Image.open(p) as im:
                imgs.append(np.array(im.convert("RGB")))
        exp.sample_images = imgs
        print("一致性取样 %d 张（%s ...）" % (len(imgs), os.path.basename(picked[0])))

    report = exp.verify_consistency()
    if not report["pass"]:
        print("\n未通过 1e-3：先查预处理（letterbox 方式、÷255、RGB、CHW），"
              "再查 NMS 细节。报告在 results/consistency.json")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
