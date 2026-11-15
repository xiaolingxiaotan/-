# -*- coding: utf-8 -*-
"""M6-1 模型导出：YOLOv8n -> ONNX（opset 17，batch 维可变）+ model_info.yaml。

用法：
    python export_onnx.py --weights models/best_fp32.pt --out models/best_fp32.onnx

做四件事：
  1) 导出 ONNX（opset 17 固定，题面指定；dynamic_axes 打开 batch 维）
  2) onnx.checker 结构校验
  3) 用 onnxruntime CPU 后端真跑一次，确认算子能落地
  4) 写 model_info.yaml：输入尺寸、输出布局、算子清单、文件大小
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-1 ONNX 导出")
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best_fp32.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "best_fp32.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17, help="题面指定 17")
    ap.add_argument("--dynamic", action="store_true", default=True,
                    help="batch 维可变（默认开）")
    ap.add_argument("--static", action="store_true",
                    help="导出静态 batch 版本（对比用）")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print("找不到权重：%s" % args.weights)
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from ultralytics import YOLO

    m = YOLO(args.weights)
    dyn = not args.static
    out = m.export(format="onnx", opset=args.opset, imgsz=args.imgsz,
                   dynamic=dyn, simplify=False, half=False,
                   device="cpu")
    # ultralytics 会导出到权重同目录，名字固定；挪到指定路径
    if os.path.abspath(str(out)) != os.path.abspath(args.out):
        os.replace(str(out), args.out)
    print("[导出] %s（%.2f MB）"
          % (args.out, os.path.getsize(args.out) / 1048576))

    # ---- 结构校验 ----
    import onnx
    model = onnx.load(args.out)
    onnx.checker.check_model(model)
    print("[校验] onnx.checker 通过")

    # ---- 算子清单 ----
    ops = Counter(n.op_type for n in model.graph.node)
    print("[算子] %d 种，%d 个节点"
          % (len(ops), sum(ops.values())))

    # ---- 输入输出规格 ----
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print("[输入] %s" % [(i.name, i.shape, i.type) for i in ins])
    print("[输出] %s" % [(o.name, o.shape, o.type) for o in outs])

    # 真跑一次
    x = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    y = sess.run(None, {ins[0].name: x})
    print("[实跑] CPUExecutionProvider 通过，输出 %s"
          % ([tuple(v.shape) for v in y],))

    # ---- model_info.yaml ----
    import yaml
    info = {
        "model": "YOLOv8n（ultralytics，AGPL-3.0）",
        "pretrained_on": "COCO 2017（80 类），本项目在其上微调 6 类",
        "task": "detect",
        "input": {
            "name": ins[0].name,
            "shape": [str(d) for d in ins[0].shape],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "normalize": "像素值 / 255，无 mean/std",
            "resize": "letterbox：等比缩放到 %d 内，短边对齐，剩余部分填 114 灰" % args.imgsz,
        },
        "output": {
            "name": outs[0].name,
            "shape": [str(d) for d in outs[0].shape],
            "layout": "N x (4 + 类别数) x 8400；8400 = 80²+40²+20²",
            "decode": "前 4 行是 cx,cy,w,h（已解码到输入像素尺度），"
                      "后面每行是一类的 sigmoid 分数，无 objectness",
        },
        "opset": args.opset,
        "dynamic_axes": {"batch": "可变"} if dyn else {"none": "全静态"},
        "size_mb": round(os.path.getsize(args.out) / 1048576, 2),
        "operators": dict(ops.most_common()),
        "runnable": ["CPUExecutionProvider"],
    }
    yaml_path = os.path.splitext(args.out)[0].replace("best_fp32", "model_info") + ".yaml"
    if not yaml_path.endswith("model_info.yaml"):
        yaml_path = os.path.join(os.path.dirname(args.out), "model_info.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)
    print("[信息] %s" % yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
