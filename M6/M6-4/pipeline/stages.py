# -*- coding: utf-8 -*-
"""三级阶段的实现：预处理 / 推理 / 后处理。

每个阶段都是一个**纯函数**，输入输出明确，方便：
  * 单测里用假函数替换（题面禁止单测依赖真实模型）
  * 串行与流水线两种跑法复用同一份代码（保证结果可比）
"""
import cv2
import numpy as np

DEFAULT_CLASSES = ["书", "笔", "杯子", "保温杯", "吧唧", "草稿本"]


def letterbox_into(img, out, new=640, color=114):
    """把 img 等比缩放并居中填充到 out（**原地写入，不新建数组**）。

    这是零拷贝的关键一行：out 是缓冲池里的内存，直接写进去，
    不再 `np.full(...)` 新建。返回值只有几何参数。
    """
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    out[...] = color
    top = (new - nh) // 2
    left = (new - nw) // 2
    out[top:top + nh, left:left + nw] = small
    return {"ratio": r, "pad": (left, top), "orig_hw": (h, w)}


def preprocess(img_bgr, canvas, blob, imgsz=640):
    """BGR 帧 -> 直接写进 canvas 与 blob（**两块都是预分配内存，运行期零分配**）。

    通道变换 + 归一化 + HWC→CHW 三步合成三次 `np.divide(..., out=blob[k])`：
        blob[0,0] ← img 蓝通道 / 255   （OpenCV 读进来是 BGR）
        blob[0,1] ← img 绿通道 / 255
        blob[0,2] ← img 红通道 / 255
    这样就省掉了 `cvtColor` + `astype` + `transpose` 三次大数组分配——
    它们是隐藏的拷贝开销，也是「明明用了引用却还是慢」的常见原因。
    """
    meta = letterbox_into(img_bgr, canvas, imgsz)
    np.divide(canvas[:, :, 2], 255.0, out=blob[0, 0])
    np.divide(canvas[:, :, 1], 255.0, out=blob[0, 1])
    np.divide(canvas[:, :, 0], 255.0, out=blob[0, 2])
    return blob, meta


def infer(session, in_name, blob):
    """跑一次推理，返回原始输出（不解码）。"""
    return session.run(None, {in_name: blob})[0]


def nms(boxes, scores, iou_thres, max_det=300):
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


def postprocess(raw, meta, conf=0.25, iou=0.45):
    """[1, 4+nc, 8400] -> [n,6]（原图坐标）。与 M6-1/infer.py 同一套数学。"""
    pred = raw[0] if raw.ndim == 3 else raw
    pred = np.squeeze(pred)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    xywh, cls_s = pred[:, :4], pred[:, 4:]
    conf_v = cls_s.max(axis=1)
    cls = cls_s.argmax(axis=1)
    keep = conf_v > conf
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)
    xywh, conf_v, cls = xywh[keep], conf_v[keep], cls[keep]
    cx, cy, w, h = xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    offset = cls[:, None].astype(np.float32) * 10000.0
    k = nms(boxes + offset, conf_v, iou)
    boxes, conf_v, cls = boxes[k], conf_v[k], cls[k]

    r = meta["ratio"]
    left, top = meta["pad"]
    oh, ow = meta["orig_hw"]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / r, 0, ow)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / r, 0, oh)
    return np.concatenate([boxes, conf_v[:, None], cls[:, None]], axis=1)


def draw(img, dets, classes=None, fps=None):
    """画框 + 类别 + 置信度 + FPS。"""
    classes = classes or DEFAULT_CLASSES
    for x1, y1, x2, y2, s, c in dets:
        col = (60, 170, 60)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        name = classes[int(c)] if int(c) < len(classes) else str(int(c))
        cv2.putText(img, "%s %.2f" % (name, s), (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(img, "FPS %.1f" % fps, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return img
