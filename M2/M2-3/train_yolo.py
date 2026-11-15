# -*- coding: utf-8 -*-
"""M2-3 最终版：YOLOv8n 微调（数据自采自标，权重来自 COCO 预训练）。

用法: python3 train_yolo.py [epochs]
"""
import sys

from ultralytics import YOLO

D = "/home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M2/M2-3"
W = "/home/xiaolingxiaotan/m23_autolabel/yolov8n.pt"      # COCO 预训练权重
DATA = D + "/yolo/data.yaml"

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 120

m = YOLO(W)
m.train(
    data=DATA,
    epochs=epochs,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=40,             # 连续 40 轮验证无提升则早停
    optimizer="auto",        # 官方推荐的 AdamW/SGD 自动选择
    cos_lr=True,             # 余弦退火
    close_mosaic=15,         # 最后 15 轮关掉 mosaic，让模型适应真实画面
    amp=True,
    project=D + "/runs",
    name="m23_yolov8n",
    exist_ok=True,
    seed=0,
    deterministic=True,
    verbose=True,
)
print("训练完成")
