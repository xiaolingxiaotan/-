#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 IoU 基线 vs 卡尔曼：生成题面 §2.5 的对比表。

两者用**同一份检测序列、同一套关联算法（匈牙利）、同一个匹配阈值**，
唯一差别是：
    纯 IoU ：用上一帧的检测框做 IoU 匹配，没匹配上立刻删（无预测）
    +卡尔曼：用卡尔曼预测框做匹配，没匹配上继续预测 10 帧

指标口径（与题面表一致）：
    IDs              全程创建的 ID 总数（越少越好，说明没被切碎）
    ID 切换           同位置跨帧换了 ID 的次数（交接错误）
    遮挡后 ID 保持率   抹掉 k 帧检测后，原 ID 还在的比例
    静止目标抖动      静止目标相邻帧中心位移的标准差（px，越小越稳）

用法：
    python compare_iou_only.py --source detections/my_scene.json
    python compare_iou_only.py --source video.mp4      # 先跑 M2-3 检测器
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detector_adapter import detect_video, load_detections  # noqa: E402
from kalman import iou_matrix  # noqa: E402
from tracker import DEFAULT_R, Tracker  # noqa: E402


def run(frames, **kw):
    trk = Tracker(**kw)
    out = []
    for dets in frames:
        rows = trk.step(dets)
        out.append([(t.id, t.box_now.copy(), t.predicted) for t in rows])
    return out, trk


def jitter_of_static(per_frame, min_frames=20):
    """挑一条位置最不动的轨迹，算它相邻帧中心位移的 std。

    没有真值也能做：全程总位移最小的那条轨迹就是静止目标。
    """
    seq = {}
    for rows in per_frame:
        for tid, b, pred in rows:
            if pred:
                continue
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            seq.setdefault(tid, []).append(c)
    best = None
    for tid, pts in seq.items():
        if len(pts) < min_frames:
            continue
        arr = np.asarray(pts)
        total = float(np.sum(np.hypot(*np.diff(arr, axis=0).T)))
        if best is None or total < best[1]:
            best = (tid, total, arr)
    if best is None:
        return None
    arr = best[2]
    step = np.hypot(*np.diff(arr, axis=0).T)
    return {"id": best[0], "frames": len(arr),
            "total_move_px": round(best[1], 1),
            "jitter_std_px": round(float(np.std(step)), 3)}


def _blank(frames, per_frame, tid, start, k):
    """把 tid 在 [start, start+k) 的检测抹掉（用于造遮挡）。"""
    out = []
    for i, dets in enumerate(frames):
        if start <= i < start + k:
            ref = next((b for t, b, _ in per_frame[i] if t == tid), None)
            if ref is not None and len(dets):
                iou = iou_matrix(dets[:, :4], np.asarray([ref]))[:, 0]
                dets = dets[iou < 0.3]
        out.append(dets)
    return out


def occlusion_retention(frames, per_frame, kw, ks=(5, 10, 15, 20), start=60):
    """逐档遮挡后「原 ID 还在不在」的比例（多目标平均）。

    注意 kw 必须传进来：之前这里忘了传，两版都用默认的卡尔曼参数跑，
    结果两张表的保持率一模一样——看起来像"卡尔曼没用"，其实是测错了对象。
    """
    # 只对「遮挡开始那一帧确实在画面里」的目标做实验：
    # 拿一个本来就已经出画的目标去测遮挡保持率没有意义（分子分母都不对）。
    present = {t for t, _, _ in per_frame[start]}
    tids = [t for t in sorted({t for rows in per_frame for t, _, _ in rows})
            if t in present][:4]
    res = {}
    for k in ks:
        keep = 0
        for tid in tids:
            pf, _ = run(_blank(frames, per_frame, tid, start, k), **kw)
            idx = min(start + k, len(pf) - 1)
            keep += int(any(i == tid for i, _, _ in pf[idx]))
        res[k] = round(keep / max(len(tids), 1), 3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="纯 IoU vs 卡尔曼")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if args.source.lower().endswith(".json"):
        frames, meta = load_detections(args.source)
    else:
        frames, meta = detect_video(args.source, max_frames=args.max_frames)

    print("序列：%d 帧" % len(frames))

    res = {}
    for tag, kw in (("纯 IoU", {"use_kalman": False}),
                    ("+卡尔曼", {"use_kalman": True, "max_age": 10})):
        pf, trk = run(frames, r_var=DEFAULT_R, **kw)
        st = jitter_of_static(pf)
        occ = occlusion_retention(frames, pf, kw)
        res[tag] = {"ids": trk.n_id_created, "id_switches": trk.n_id_switches,
                    "static": st, "occlusion_retention": occ}
        print("%-8s IDs %2d  切换 %2d  静止抖动 %s px  遮挡保持 %s"
              % (tag, trk.n_id_created, trk.n_id_switches,
                 st["jitter_std_px"] if st else "-",
                 {k: "%.0f%%" % (v * 100) for k, v in occ.items()}))

    # ---- 写对比表 ----
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "compare.csv")
    a = res["纯 IoU"]
    b = res["+卡尔曼"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["指标", "纯 IoU", "+卡尔曼"])
        w.writerow(["ID 总数", a["ids"], b["ids"]])
        w.writerow(["ID 切换次数", a["id_switches"], b["id_switches"]])
        for k in sorted(a["occlusion_retention"]):
            w.writerow(["遮挡 %d 帧后 ID 保持率" % k,
                        "%.0f%%" % (a["occlusion_retention"][k] * 100),
                        "%.0f%%" % (b["occlusion_retention"][k] * 100)])
        w.writerow(["静止目标抖动 std (px)",
                    a["static"]["jitter_std_px"] if a["static"] else "-",
                    b["static"]["jitter_std_px"] if b["static"] else "-"])
    print("\n对比表 -> %s" % path)
    with open(os.path.join(args.out_dir, "compare.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
