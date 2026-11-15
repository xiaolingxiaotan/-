#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6-1 统一推理脚本：ONNX + 预处理 + 后处理（NMS）一体化，输出检测框 JSON。

硬性要求：**不 import torch / tensorflow**。板端只装 onnxruntime + opencv + numpy
就能跑，现场会在干净环境里验证这一点。

用法：
    python infer.py --model models/best_fp32.onnx --source img.jpg --out dets.json
    python infer.py --model models/best_fp32.onnx --source data/raw --out dets.json
    python infer.py --model models/best_fp32.onnx --source clip.mp4 --out dets.json
    python infer.py --model models/best_fp32.onnx --source camera --out dets.json
    python infer.py --model models/best_fp32.onnx --device gpu --source camera

输出 JSON 有两种格式，用 --json-format 选：
    simple（默认）：{"frames":[{"frame":0,"dets":[{"cls":3,"conf":0.87,"bbox":[...]}]}]}
    m24          ：再补 video / fps / classes 三个字段，字段名与 M2-4 §6 的约定一致
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSES = ["书", "笔", "杯子", "保温杯", "吧唧", "草稿本"]


# ---------------------------------------------------------------- 预处理
def letterbox(img, new=640, color=114):
    """等比缩放 + 居中填充，与 ultralytics 训练时的 letterbox 完全一致。

    为什么不能直接 cv2.resize(img, (640, 640))：那会把 16:9 的画面压扁，
    物体的宽高比整体变形，框会系统性偏移——这是部署掉点的第一大原因。
    """
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        # 缩小用 INTER_AREA（面积平均，等价先低通再抽样），放大用 INTER_LINEAR
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, {"ratio": r, "pad": (left, top), "orig_hw": (h, w)}


def preprocess(img_bgr, imgsz=640):
    """BGR 图 -> (1,3,S,S) float32。通道顺序必须是 RGB，与训练一致。"""
    lb, meta = letterbox(img_bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    return np.ascontiguousarray(blob), meta


# ---------------------------------------------------------------- 后处理
def nms(boxes, scores, iou_thres, max_det=300):
    """单类非极大值抑制（numpy 实现，不依赖 torchvision）。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = ((boxes[rest, 2] - boxes[rest, 0]) *
              (boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / np.maximum(a1 + a2 - inter, 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)


def postprocess(out, meta, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """[1, 4+nc, 8400] -> [[x1,y1,x2,y2,conf,cls], ...]（原图坐标）。

    这个布局的解码要点（题面 §2.2 追问的第 3 点）：
      * 第 0 维是 batch，第 1 维才是「4 个框参数 + 类别分」，**要先转置**
        再按行取，否则会把通道当格点。
      * 前 4 个数已经是 cx,cy,w,h 的**像素值**（YOLOv8 导出时已解码），
        不要再乘步长、也不要再 sigmoid。
      * 后面是每类的分数，**已经过 sigmoid**，直接取 argmax；
        YOLOv8 没有独立 objectness，不要去找第 5 个数。
    """
    pred = out[0] if out.ndim == 3 else out
    pred = np.squeeze(pred)
    if pred.shape[0] < pred.shape[1]:      # [4+nc, 8400] -> [8400, 4+nc]
        pred = pred.T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    conf = cls_scores.max(axis=1)
    cls = cls_scores.argmax(axis=1)
    keep = conf > conf_thres
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]

    # cx,cy,w,h -> x1,y1,x2,y2
    cx, cy, w, h = boxes_xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 逐类 NMS：给不同类别加大偏移，跨类 IoU 恒为 0，一次抑制等价逐类抑制
    offset = cls[:, None].astype(np.float32) * 10000.0
    keep = nms(boxes + offset, conf, iou_thres, max_det)
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    # letterbox 反算回原图坐标
    r = meta["ratio"]
    left, top = meta["pad"]
    oh, ow = meta["orig_hw"]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / r, 0, ow)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / r, 0, oh)

    return np.concatenate([boxes, conf[:, None], cls[:, None]], axis=1)


# ---------------------------------------------------------------- 引擎
class OnnxDetector:
    """只依赖 onnxruntime / opencv / numpy 的检测器。"""

    def __init__(self, model_path, imgsz=640, conf=0.25, iou=0.45,
                 device="cpu", classes=None):
        import onnxruntime as ort

        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.classes = classes or DEFAULT_CLASSES
        want = {"cpu": ["CPUExecutionProvider"],
                "gpu": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "npu": None}[device]
        if want is None:
            raise SystemExit("NPU：本机没有可用的 NPU 执行器。"
                             "题面允许记录为「设备不具备」，见 docs/deploy.md")
        avail = ort.get_available_providers()
        use = [p for p in want if p in avail]
        if not use:
            raise SystemExit("需要的执行器 %s 都没装（可用：%s）"
                             % (want, avail))
        if device == "gpu" and "CUDAExecutionProvider" not in use:
            print("[警告] 没装 onnxruntime-gpu，退回 CPU 执行器")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=use)
        self.in_name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        self.model_path = model_path
        print("[引擎] %s  执行器 %s" % (os.path.basename(model_path),
                                      self.provider))

    def detect(self, img_bgr):
        blob, meta = preprocess(img_bgr, self.imgsz)
        out = self.sess.run(None, {self.in_name: blob})[0]
        return postprocess(out, meta, self.conf, self.iou)

    def draw(self, img, dets, fps=None):
        for x1, y1, x2, y2, s, c in dets:
            col = (60, 160, 60)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            name = self.classes[int(c)] if int(c) < len(self.classes) else str(int(c))
            cv2.putText(img, "%s %.2f" % (name, s), (int(x1), max(int(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if fps is not None:
            cv2.putText(img, "FPS %.1f" % fps, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        return img


# ---------------------------------------------------------------- 输入源
def iter_frames(src, imgsz=None, show=False, max_frames=None):
    """统一把 图片 / 目录 / 视频 / 摄像头 变成 (frame_idx, bgr) 的生成器。"""
    if src == "camera" or src.isdigit():
        cap = cv2.VideoCapture(0 if src == "camera" else int(src))
        if not cap.isOpened():
            raise SystemExit("打不开摄像头")
        idx = 0
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield idx, f
                idx += 1
                if max_frames and idx >= max_frames:
                    break
        finally:
            cap.release()
        return

    exts_img = (".jpg", ".jpeg", ".png", ".bmp")
    exts_vid = (".mp4", ".avi", ".mov", ".mkv")
    if os.path.isdir(src):
        names = [n for n in sorted(os.listdir(src)) if n.lower().endswith(exts_img)]
        for i, n in enumerate(names):
            f = cv2.imread(os.path.join(src, n))
            if f is not None:
                yield i, f
        return
    if src.lower().endswith(exts_vid):
        cap = cv2.VideoCapture(src)
        idx = 0
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield idx, f
                idx += 1
                if max_frames and idx >= max_frames:
                    break
        finally:
            cap.release()
        return
    f = cv2.imread(src)
    if f is None:
        raise SystemExit("读不到图：%s" % src)
    yield 0, f


def main() -> int:
    ap = argparse.ArgumentParser(description="M6-1 统一推理（零训练框架依赖）")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "best_fp32.onnx"))
    ap.add_argument("--source", required=True, help="图片 / 目录 / 视频 / camera")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "npu"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--json-format", default="simple", choices=["simple", "m24"])
    ap.add_argument("--max-frames", type=int, default=None, help="视频/摄像头最多处理几帧")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    det = OnnxDetector(args.model, imgsz=args.imgsz, conf=args.conf,
                       iou=args.iou, device=args.device)

    frames = []
    t0 = time.perf_counter()
    n = 0
    for idx, frame in iter_frames(args.source, max_frames=args.max_frames):
        dets = det.detect(frame)
        items = [{"cls": int(d[5]), "conf": round(float(d[4]), 4),
                  "bbox": [round(float(v), 2) for v in d[:4]]} for d in dets]
        frames.append({"frame": idx, "dets": items})
        n += 1
        if args.show:
            vis = det.draw(frame.copy(), dets)
            cv2.imshow("infer", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if args.show:
        cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    print("[推理] %d 帧，%.1f ms/帧，%.1f FPS" % (n, dt / max(n, 1) * 1000,
                                                n / max(dt, 1e-9)))

    doc = {"frames": frames}
    if args.json_format == "m24":
        doc.update({"video": os.path.basename(args.source),
                    "fps": round(n / max(dt, 1e-9), 2),
                    "classes": det.classes})
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        print("[输出] %s" % args.out)
    else:
        print(json.dumps(doc["frames"][:1], ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
