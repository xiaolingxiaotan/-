# -*- coding: utf-8 -*-
"""M2-3 检测输出的适配层（M2-4 §6 的接口）。

两种入口：
  1. 读 JSON（detections/<video>.json）——调试追踪时不必反复跑模型
  2. 接实时检测——直接调用 M2-3 的 ONNX 检测器（最终演示必须用这条）

JSON 结构（与题面 §6 一致）：
    {"video": "...", "fps": 30, "classes": [...],
     "frames": [{"frame": 0, "dets": [{"cls":3,"conf":0.87,"bbox":[...]}]}]}
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M23 = os.path.join(HERE, "..", "M2-3")


# ---------------------------------------------------------------- JSON 入口
def load_detections(json_path):
    """读检测存档，返回 (frames, meta)。

    frames: 列表，每项是 [n,6] 的数组 [x1,y1,x2,y2,conf,cls]
    meta:   {"video", "fps", "classes"}
    """
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = []
    for fr in doc["frames"]:
        dets = fr.get("dets") or []
        if dets:
            arr = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2],
                             d["bbox"][3], d["conf"], d["cls"]] for d in dets],
                           dtype=np.float32)
        else:
            arr = np.zeros((0, 6), dtype=np.float32)
        frames.append(arr)
    meta = {"video": doc.get("video"), "fps": doc.get("fps", 30),
            "classes": doc.get("classes")}
    return frames, meta


def save_detections(path, frames, video, fps, classes):
    """把检测结果写成 §6 约定的 JSON（给追踪用/给现场核对）。"""
    doc = {"video": os.path.basename(video), "fps": fps,
           "classes": list(classes), "frames": []}
    for i, arr in enumerate(frames):
        doc["frames"].append({
            "frame": i,
            "dets": [{"cls": int(c), "conf": round(float(s), 4),
                      "bbox": [round(float(v), 2) for v in b]}
                     for *b, s, c in arr],
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return path


# ---------------------------------------------------------------- 实时入口
class LiveDetector:
    """把 M2-3 的 ONNX 检测器包装成「喂一帧、返回 [n,6]」。

    依赖 M2-3/train.py 里的 Detector（它已经支持 YOLOv8 的单张量输出），
    这样 M2-4 与 M2-3 用的是**同一份推理代码**，预处理不会各写一套。
    """

    def __init__(self, onnx_path=None, conf=0.25, iou=0.45, imgsz=640):
        import sys
        sys.path.insert(0, M23)
        from train import Detector

        onnx_path = onnx_path or os.path.join(M23, "models", "best.onnx")
        self.det = Detector(onnx_path=onnx_path, backend="onnx",
                            input_size=imgsz, conf_thres=conf, nms_iou=iou)
        self.det.load()
        self.classes = [self.det.class_names.get(i, str(i))
                        for i in range(len(self.det.class_names) or 6)]

    def __call__(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.det.detect(rgb)          # [n,6]：x1,y1,x2,y2,conf,cls
        return out


def detect_video(video_path, onnx_path=None, conf=0.25, max_frames=None):
    """跑一段视频，返回逐帧检测（不落盘），供 save_detections 使用。"""
    import cv2

    det = LiveDetector(onnx_path, conf=conf)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("打不开视频：%s" % video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(det(f))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames, {"video": video_path, "fps": fps, "classes": det.classes}
