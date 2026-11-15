#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-1 一致性验证：ONNX 输出 vs PyTorch 输出（题面 §2.1 硬性要求）。

给出三个数：
    CosSim          同一批输入下原始输出张量的余弦相似度   > 0.99
    最大绝对差      max|onnx_out - torch_out|
    检测框最大偏差  后处理后框的像素偏差

做法说明（这一点很关键）：**不能用 ultralytics 的 predict() 去比**——
它内部已经做完解码和 NMS，拿不到原始张量，比出来只是「后处理结果是否一样」。
这里直接调 DetectionModel.forward 取 raw 输出，与 ONNX 的 raw 输出逐元素对比，
才是题面要的那一层。

用法：
    python tools/check_consistency.py --onnx models/best_fp32.onnx \
        --pt models/best_fp32.pt --data ../M2-3/data/raw --num 8
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from infer import OnnxDetector, postprocess, preprocess  # noqa: E402


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX vs PyTorch 一致性")
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "best_fp32.onnx"))
    ap.add_argument("--pt", default=os.path.join(HERE, "models", "best_fp32.pt"))
    ap.add_argument("--data", required=True, help="图片目录")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "consistency.json"))
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    imgs = []
    for n in sorted(os.listdir(args.data)):
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            f = cv2.imread(os.path.join(args.data, n))
            if f is not None:
                imgs.append((n, f))
        if len(imgs) >= args.num:
            break
    if not imgs:
        print("没找到图片")
        return 1

    det = OnnxDetector(args.onnx, imgsz=args.imgsz, conf=0.25, iou=0.45)
    net = YOLO(args.pt).model.float().eval()

    rows = []
    for name, img in imgs:
        blob, meta = preprocess(img, args.imgsz)

        out_onnx = det.sess.run(None, {det.in_name: blob})[0]
        with torch.no_grad():
            y = net(torch.from_numpy(blob))
        out_torch = (y[0] if isinstance(y, (list, tuple)) else y).numpy()

        cs = cos_sim(out_onnx, out_torch)
        absd = float(np.abs(out_onnx - out_torch).max())

        d_onnx = postprocess(out_onnx, meta, 0.25, 0.45)
        d_torch = postprocess(out_torch, meta, 0.25, 0.45)
        if len(d_onnx) and len(d_torch):
            a = d_onnx[d_onnx[:, 4].argsort()[::-1]]
            b = d_torch[d_torch[:, 4].argsort()[::-1]]
            m = min(len(a), len(b))
            box_diff = float(np.abs(a[:m, :4] - b[:m, :4]).max())
            conf_diff = float(np.abs(a[:m, 4] - b[:m, 4]).max())
        else:
            box_diff = conf_diff = 0.0
        cnt_diff = abs(len(d_onnx) - len(d_torch))

        rows.append({"image": name, "cossim": round(cs, 8),
                     "max_abs_diff": absd, "boxes": len(d_onnx),
                     "count_diff": cnt_diff,
                     "max_box_diff_px": round(box_diff, 4),
                     "max_conf_diff": round(conf_diff, 6)})
        print("%-26s CosSim %.8f  maxΔ %.2e  框数 %d/%d  框偏差 %.4f px"
              % (name[:26], cs, absd, len(d_onnx), len(d_torch), box_diff))

    result = {
        "n_images": len(rows),
        "cossim_min": round(min(r["cossim"] for r in rows), 8),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "max_box_diff_px": round(max(r["max_box_diff_px"] for r in rows), 4),
        "max_conf_diff": round(max(r["max_conf_diff"] for r in rows), 6),
        "max_count_diff": int(max(r["count_diff"] for r in rows)),
        "threshold_cossim": 0.99,
        "detail": rows,
    }
    result["pass"] = bool(result["cossim_min"] > 0.99
                          and result["max_box_diff_px"] < 1.0
                          and result["max_count_diff"] == 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n最小 CosSim %.8f（阈值 0.99）| max|Δ| %.2e | 框偏差 %.4f px | 数量差 %d"
          % (result["cossim_min"], result["max_abs_diff"],
             result["max_box_diff_px"], result["max_count_diff"]))
    print("判定：%s" % ("通过" if result["pass"] else "未通过，需要解释"))
    print("结果 -> %s" % args.out)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
