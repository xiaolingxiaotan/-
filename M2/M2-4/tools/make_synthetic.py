# -*- coding: utf-8 -*-
"""生成一段合成的检测序列（没有真实视频时用来验证追踪逻辑）。

情节与你拍的视频要求一致：多目标并行 / 两目标交叉 / 短时遮挡 / 出画再入 / 静止目标。
**注意**：这只是给单测和联调用，正式报告里的数据必须用你自己拍的视频重跑
（`tools/experiments.py` 同一条命令，换 --source 即可）。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 640, 480
FPS = 30
N = 150


def box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def build():
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N):
        t = i / FPS
        dets = []

        # A：从左向右横穿（0~5s），与 B 交叉；第 60~70 帧被"遮挡"
        if 0 <= i < N:
            ax = 60 + (i / N) * 520
            ay = 240 + 10 * np.sin(i / 8.0)
            # A 第 60~69 帧被遮挡（正好 10 帧，等于题面的预测上限）
            if not (60 <= i <= 69):
                dets.append((2, box(ax, ay, 70, 40)))          # 杯子

            # B：从右向左，与 A 在画面中央交叉
            bx = 580 - (i / N) * 520
            by = 240 - 8 * np.sin(i / 8.0 + 1.0)
            dets.append((2, box(bx, by, 70, 40)))

            # C：静止目标（考滤波降噪）
            dets.append((0, box(120, 120, 90, 60)))            # 书

            # D：出画再入（0~40 帧在画面内，41~80 出去，81 之后回来）
            # 位置固定不动——原来写成 i%20 的锯齿会让框每 20 帧跳 38px，
            # 那不是"出画再入"，而是假的轨迹断裂，会把 ID 数搞乱。
            if i <= 40 or i >= 81:
                dets.append((3, box(320, 400, 50, 90)))        # 保温杯

        out = []
        for cls, b in dets:
            # 检测抖动 ~ N(0, 1.5px)
            jb = [v + rng.normal(0, 1.5) for v in b]
            conf = float(np.clip(rng.normal(0.85, 0.05), 0.3, 0.99))
            out.append({"cls": cls, "conf": round(conf, 3),
                        "bbox": [round(v, 2) for v in jb]})
        frames.append({"frame": i, "dets": out})

    doc = {"video": "synthetic.mp4", "fps": FPS,
           "classes": ["书", "笔", "杯子", "保温杯", "吧唧", "草稿本"],
           "frames": frames}
    path = os.path.join(HERE, "detections", "synthetic.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print("合成检测序列 -> %s（%d 帧）" % (path, len(frames)))
    print("情节：A/B 交叉、A 第 60~70 帧遮挡、C 静止、D 第 41~80 帧出画再入")
    return path


if __name__ == "__main__":
    build()
