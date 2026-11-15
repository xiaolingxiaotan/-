# -*- coding: utf-8 -*-
"""M2-4 追踪器：匈牙利关联 + 卡尔曼滤波 + 短时遮挡预测 + 轨迹与累计位移。

设计要点（对应题面 §2.1~2.4）：

    关联：IoU 代价矩阵 + 匈牙利（scipy.optimize.linear_sum_assignment）。
          不用贪心——目标交叉时贪心会先到先得，锁错 ID。
          代价用的是**预测框**与检测框的 IoU，不是上一帧检测框：
          运动快或帧率低时，上一帧位置已经偏出去了，用预测框才配得上。

    两阶段匹配：先高置信度（> conf_hi），再低置信度（> conf_lo）补匹配。
          低分检测只允许匹配已存在的轨迹，不允许新建——否则背景噪声会刷出假目标。

    新生：连续 min_hits 帧命中才分配 ID（默认 2）。单帧偶发检测不给 ID，
          这就是「抖动检测会刷出假目标」的解法。

    死亡：连续 max_age 帧无匹配才删（默认 10，与题面的 10 帧预测一致）。
          期间输出**预测框**并标记 PREDICTED，ID 保持不变；重新匹配上时
          用新检测更新并恢复实线。

    累计位移：从轨迹诞生到当前的**实际移动距离**（逐帧位移累加），
          不是首末直线距离——绕圈回来时两者差很多。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kalman import KalmanBox, iou_matrix, measure_jitter  # noqa: E402

CLASSES = ["书", "笔", "杯子", "保温杯", "吧唧", "草稿本"]

# 默认 R（观测噪声），会被 measure_jitter 的实测值覆盖
DEFAULT_R = np.array([4.0, 4.0, 9.0, 9.0])


def box_iou(a, b):
    return float(iou_matrix(np.asarray([a]), np.asarray([b]))[0, 0])


class Track:
    """一条轨迹。"""

    def __init__(self, det, r_var=DEFAULT_R, q_accel=2.0, use_kalman=True):
        # ID 先不发：单帧偶发检测不能占用 ID，否则噪声会刷出一堆假 ID。
        # 只有连续命中 min_hits 帧、被 Tracker 确认之后才 assign_id()。
        self.id = None
        self.cls = int(det[5])
        self.conf = float(det[4])
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        self.predicted = False
        self.use_kalman = use_kalman
        self.kf = KalmanBox(det[:4], r_var, q_accel=q_accel) if use_kalman else None
        self.box_now = np.asarray(det[:4], dtype=np.float64)
        self.history = [tuple(self.center())]
        self.displacement = 0.0

    # ---- 推进 ----
    def assign_id(self, tid):
        """确认后才发 ID（ID 一旦发出永不复用）。

        ID 由 Tracker 统一分配，不用类变量：同一进程里跑多个追踪器（对比实验、
        参数扫描）时，类变量会一直累加，第二个追踪器的 ID 从 7 开始，
        跟第一份结果对不上——调试时会以为是追踪逻辑错了。
        """
        self.id = tid
        return self.id

    def predict(self):
        self.age += 1
        if self.kf is not None:
            self.box_now = self.kf.predict()
        self.time_since_update += 1
        return self.box_now

    def update(self, det):
        prev = self.center()
        self.cls = int(det[5])
        self.conf = float(det[4])
        if self.kf is not None:
            self.box_now = self.kf.update(det[:4])
        else:
            self.box_now = np.asarray(det[:4], dtype=np.float64)
        self.hits += 1
        self.time_since_update = 0
        self.predicted = False
        c = self.center()
        self.displacement += math.hypot(c[0] - prev[0], c[1] - prev[1])
        self.history.append(c)
        if len(self.history) > 300:
            self.history.pop(0)

    # ---- 读取 ----
    def center(self):
        if self.kf is not None:
            return self.kf.center()
        return (float((self.box_now[0] + self.box_now[2]) / 2),
                float((self.box_now[1] + self.box_now[3]) / 2))

    def next_box(self):
        """预测下一帧位置（运动估计）。"""
        if self.kf is not None:
            return self.kf.next_box(1)
        cx, cy = self.center()
        w = self.box_now[2] - self.box_now[0]
        h = self.box_now[3] - self.box_now[1]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def pos_std(self):
        if self.kf is not None:
            return self.kf.pos_std()
        return (0.0, 0.0)


class Tracker:
    """多目标追踪器。"""

    def __init__(self, r_var=DEFAULT_R, q_accel=2.0, iou_thr=0.3,
                 max_age=10, min_hits=2, conf_hi=0.5, conf_lo=0.15,
                 use_kalman=True):
        self.r_var = np.asarray(r_var, dtype=np.float64)
        self.q_accel = q_accel
        self.iou_thr = iou_thr
        if not use_kalman and max_age == 10:
            # 纯 IoU 基线：没有预测能力，遮挡时无框可匹配 —— 只能直接删。
            # 这正是题面 §0 说的「遮挡时没有检测可用，目标就被删了」。
            max_age = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.conf_hi = conf_hi
        self.conf_lo = conf_lo
        self.use_kalman = use_kalman
        self.tracks = []
        self.confirmed = []          # 已分配 ID 的
        self.frame = 0
        self._next_id = 1            # 本追踪器自己的 ID 计数器（不复用）
        self.n_id_created = 0
        self.n_id_switches = 0
        self._prev_rows = []         # 上一帧的 (id, box)，用于统计 ID 跳变

    # ---- 一帧 ----
    def step(self, dets):
        self.frame += 1
        # 1) 全部轨迹先预测
        for t in self.tracks:
            t.predict()

        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6) \
            if len(dets) else np.zeros((0, 6))
        matched, unmatched_det, unmatched_trk = self._associate(dets)

        # 2) 更新匹配上的
        for ti, di in matched:
            self.tracks[ti].update(dets[di])

        # 3) 未匹配的检测 -> 新建候选轨迹（此时还没有 ID）
        for di in unmatched_det:
            t = Track(dets[di], self.r_var, self.q_accel, self.use_kalman)
            self.tracks.append(t)

        # 4) 清理超龄
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # 5) 够 hit 次数的发 ID，并决定哪些对外输出
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits and t.id is None:
                t.assign_id(self._next_id)
                self._next_id += 1
                self.n_id_created += 1
            if t.id is None:                # 还是候选，不对外
                continue
            if t.time_since_update == 0:
                out.append(t)
            elif t.time_since_update <= self.max_age:
                t.predicted = True          # 遮挡中：用预测值顶着
                out.append(t)

        self._count_switches(out)
        return out

    # ---- 关联 ----
    def _associate(self, dets):
        if len(self.tracks) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(self.tracks)))

        trk_boxes = np.array([t.box_now for t in self.tracks], dtype=np.float64)
        det_boxes = dets[:, :4]
        iou = iou_matrix(trk_boxes, det_boxes)

        # 类别不一致不匹配（检测类别抖动时，靠低分阶段兜底）
        same_cls = np.array([[t.cls == int(d[5]) for d in dets]
                             for t in self.tracks])
        cost = 1.0 - iou
        cost[~same_cls] = 1e6
        cost[iou < self.iou_thr] = 1e6

        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        matched, used_d, used_t = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e5:
                continue
            matched.append((int(r), int(c)))
            used_d.add(int(c))
            used_t.add(int(r))

        # 第二阶段：低分检测补匹配（只给已有轨迹，不新建）
        rest_d = [i for i in range(len(dets)) if i not in used_d]
        rest_t = [i for i in range(len(self.tracks)) if i not in used_t]
        if rest_d and rest_t:
            sub_iou = iou[rest_t][:, rest_d]
            sub_cost = 1.0 - sub_iou
            for a, b in zip(*linear_sum_assignment(sub_cost)):
                if sub_iou[a, b] >= self.iou_thr:
                    matched.append((rest_t[a], rest_d[b]))
                    used_d.add(rest_d[b])
                    used_t.add(rest_t[a])

        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        return matched, unmatched_d, unmatched_t

    # ---- ID 跳变统计 ----
    def _count_switches(self, out):
        """统计「同一位置在两帧之间换了 ID」的次数。

        做法：对当前帧每个框，在上一帧里找 IoU 最大的框；若 IoU > 0.5 但 ID
        不同，记一次跳变。没有真值 ID 的视频也能算，口径统一，两种实现可比。
        """
        cur = [(t.id, t.box_now) for t in out if not t.predicted]
        if self._prev_rows and cur:
            prev_boxes = np.array([b for _, b in self._prev_rows])
            prev_ids = [i for i, _ in self._prev_rows]
            for tid, b in cur:
                iou = iou_matrix(np.asarray([b]), prev_boxes)[0]
                j = int(np.argmax(iou))
                if iou[j] > 0.5 and prev_ids[j] != tid:
                    self.n_id_switches += 1
        self._prev_rows = cur


# ---------------------------------------------------------------- 绘制
def _dashed_rect(img, x1, y1, x2, y2, color, dash=8, gap=6, thick=2):
    """画虚线框（表示这是预测，不是真检测）。"""
    import cv2
    for x in range(int(x1), int(x2), dash + gap):
        cv2.line(img, (x, int(y1)), (min(x + dash, int(x2)), int(y1)), color, thick)
        cv2.line(img, (x, int(y2)), (min(x + dash, int(x2)), int(y2)), color, thick)
    for y in range(int(y1), int(y2), dash + gap):
        cv2.line(img, (int(x1), y), (int(x1), min(y + dash, int(y2))), color, thick)
        cv2.line(img, (int(x2), y), (int(x2), min(y + dash, int(y2))), color, thick)


def draw_tracks(img, tracks, classes=None, tail=40, show_next=True):
    """实线=真检测，虚线=预测，青色细框=下一帧预测，折线=轨迹。"""
    import cv2

    classes = classes or CLASSES
    for t in tracks:
        x1, y1, x2, y2 = t.box_now
        name = classes[t.cls] if t.cls < len(classes) else str(t.cls)
        label = "ID%d %s %.2f" % (t.id, name, t.conf)

        if t.predicted:
            _dashed_rect(img, x1, y1, x2, y2, (0, 165, 255))
            cv2.putText(img, "PREDICTED", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (60, 200, 60), 2)

        # 轨迹尾迹
        pts = t.history[-tail:]
        for i in range(1, len(pts)):
            cv2.line(img, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])), (60, 200, 60), 2)

        # 下一帧预测（青色细框）
        if show_next:
            nx1, ny1, nx2, ny2 = t.next_box()
            cv2.rectangle(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)),
                          (255, 200, 0), 1)

        # 累计位移
        cv2.putText(img, "%s  d=%.0fpx" % (label, t.displacement),
                    (int(x1), int(y2) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------- CLI
def main() -> int:
    import argparse
    import json

    import cv2

    from detector_adapter import detect_video, load_detections, save_detections

    ap = argparse.ArgumentParser(description="M2-4 追踪")
    ap.add_argument("--source", required=True,
                    help="检测 JSON，或视频路径（实时跑 M2-3 检测器）")
    ap.add_argument("--out-json", default=None, help="视频模式下把检测存成 JSON")
    ap.add_argument("--out-video", default=None, help="输出带标注的 mp4")
    ap.add_argument("--max-age", type=int, default=10)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--no-kalman", action="store_true", help="纯 IoU 基线")
    ap.add_argument("--r-var", default=None,
                    help="观测噪声方差 cx,cy,w,h；不给就用实测默认值")
    ap.add_argument("--q-accel", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    if args.source.lower().endswith(".json"):
        frames, meta = load_detections(args.source)
    else:
        frames, meta = detect_video(args.source, max_frames=args.max_frames)
        if args.out_json:
            save_detections(args.out_json, frames, meta["video"],
                            meta["fps"], meta["classes"])

    r_var = (np.array([float(x) for x in args.r_var.split(",")])
             if args.r_var else DEFAULT_R)
    trk = Tracker(r_var=r_var, q_accel=args.q_accel, iou_thr=args.iou_thr,
                  max_age=args.max_age, min_hits=args.min_hits,
                  use_kalman=not args.no_kalman)

    # 逐帧输出
    per_frame = []
    for i, dets in enumerate(frames):
        out = trk.step(dets)
        per_frame.append([{"id": t.id, "cls": t.cls, "conf": round(t.conf, 3),
                           "bbox": [round(float(v), 1) for v in t.box_now],
                           "predicted": bool(t.predicted),
                           "disp_px": round(t.displacement, 1),
                           "std_px": round(float(np.hypot(*t.pos_std())), 2)}
                          for t in out])

    stats = {"frames": len(frames), "ids": trk.n_id_created,
             "id_switches": trk.n_id_switches,
             "kalman": not args.no_kalman,
             "max_age": trk.max_age, "iou_thr": args.iou_thr,
             "r_var": list(map(float, r_var))}
    print(json.dumps(stats, ensure_ascii=False))

    if args.stats_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)), exist_ok=True)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "per_frame": per_frame}, f,
                      ensure_ascii=False, indent=1)
        print("逐帧结果 -> %s" % args.stats_out)

    # 可选：把标注画回视频
    if args.out_video and not args.source.lower().endswith(".json"):
        cap = cv2.VideoCapture(args.source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        writer = cv2.VideoWriter(args.out_video,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        trk2 = Tracker(r_var=r_var, q_accel=args.q_accel, iou_thr=args.iou_thr,
                       max_age=args.max_age, min_hits=args.min_hits,
                       use_kalman=not args.no_kalman)
        cap = cv2.VideoCapture(args.source)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok or (args.max_frames and i >= args.max_frames):
                break
            out = trk2.step(frames[i] if i < len(frames) else np.zeros((0, 6)))
            draw_tracks(f, out, meta["classes"])
            writer.write(f)
            i += 1
        cap.release()
        writer.release()
        print("标注视频 -> %s" % args.out_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
