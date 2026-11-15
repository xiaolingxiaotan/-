# -*- coding: utf-8 -*-
"""遮挡与预测实验（题面 §2.3 的三张表 / 一条曲线）。

三件实验，全部用同一份检测序列，**只改参数不改数据**：

  1. 遮挡恢复率：把某条轨迹的检测抹掉 k 帧（5/10/15/20），
     看轨迹的 ID 是否还在、是否被复用。
  2. 预测误差曲线：抹掉的这 k 帧里，预测框中心 vs 参考运行（未抹）的真实中心，
     误差随帧数怎么涨 —— 这条曲线就是「10 帧阈值」的依据。
  3. 10 帧阈值的权衡：换 5 / 10 / 20 帧，看「保持率」和「错跟率」怎么变。

参考运行 = 用完整检测跑一遍，拿它的输出当真值（不需要人工标注）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from detector_adapter import load_detections  # noqa: E402
from kalman import iou_matrix  # noqa: E402
from tracker import DEFAULT_R, Tracker  # noqa: E402


def run(frames, **kw):
    trk = Tracker(**kw)
    per_frame = []
    for dets in frames:
        out = trk.step(dets)
        per_frame.append([(t.id, t.box_now.copy(), t.predicted) for t in out])
    return per_frame, trk


def pick_target(per_frame, frame_idx=50):
    """挑一条「在遮挡前后都在、且运动明显」的轨迹当实验对象。"""
    ids_here = {i for i, _, _ in per_frame[frame_idx]}
    best = None
    for tid in ids_here:
        # 数一数这条 ID 在全程出现的帧数
        cnt = sum(1 for row in per_frame if any(i == tid for i, _, _ in row))
        if best is None or cnt > best[1]:
            best = (tid, cnt)
    return best[0] if best else None


def blank_track(frames, tid, ref, start, k):
    """把 tid 在 [start, start+k) 里的检测删掉，返回新序列。"""
    out = []
    for i, dets in enumerate(frames):
        if start <= i < start + k:
            # 参考运行里该帧该 ID 的框，用来找出对应检测并抹掉
            ref_box = None
            for j, b, _ in ref[i]:
                if j == tid:
                    ref_box = b
                    break
            if ref_box is not None and len(dets):
                iou = iou_matrix(dets[:, :4], np.asarray([ref_box]))[:, 0]
                keep = iou < 0.3
                dets = dets[keep]
        out.append(dets)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="遮挡/预测实验")
    ap.add_argument("--source", default=os.path.join(HERE, "detections",
                                                     "synthetic.json"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--ks", default="5,10,15,20")
    args = ap.parse_args()

    frames, meta = load_detections(args.source)
    ref, _ = run(frames)
    tid = pick_target(ref)
    if tid is None:
        print("没找到可用的目标轨迹")
        return 1
    print("实验对象：ID %d（%d 帧序列）" % (tid, len(frames)))

    ks = [int(x) for x in args.ks.split(",")]
    rows = []
    curve = {}                      # k -> 每帧误差列表

    for k in ks:
        start = 60
        dets_mod = blank_track(frames, tid, ref, start, k)
        per_frame, trk = run(dets_mod)

        # 遮挡结束后那一帧，这个 ID 还在不在
        after = start + k
        alive = any(i == tid for i, _, _ in per_frame[min(after, len(per_frame) - 1)])
        # 遮挡期间是否用预测值顶着
        pred_frames = sum(1 for f in range(start, min(after, len(per_frame)))
                          if any(i == tid and p for i, _, p in per_frame[f]))
        # 预测误差：预测框中心 vs 参考中心
        errs = []
        for f in range(start, min(after, len(per_frame))):
            gt = next((b for i, b, _ in ref[f] if i == tid), None)
            pr = next((b for i, b, _ in per_frame[f] if i == tid), None)
            if gt is None or pr is None:
                errs.append(float("nan"))
            else:
                gc = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
                pc = ((pr[0] + pr[2]) / 2, (pr[1] + pr[3]) / 2)
                errs.append(float(np.hypot(gc[0] - pc[0], gc[1] - pc[1])))
        curve[k] = errs
        rows.append({"k": k, "id_kept": bool(alive),
                     "predicted_frames": pred_frames,
                     "mean_err_px": round(float(np.nanmean(errs)), 2)
                     if errs else None,
                     "max_err_px": round(float(np.nanmax(errs)), 2)
                     if errs else None})

    # ---- 纯 IoU 基线在同一实验下的表现 ----
    base_rows = []
    for k in ks:
        dets_mod = blank_track(frames, tid, ref, 60, k)
        per_frame, _ = run(dets_mod, use_kalman=False)
        after = 60 + k
        alive = any(i == tid for i, _, _ in per_frame[min(after, len(per_frame) - 1)])
        base_rows.append({"k": k, "id_kept": bool(alive)})

    print("\n遮挡帧数 | 卡尔曼: ID保持 / 预测帧数 / 平均误差 | 纯IoU: ID保持")
    for r, b in zip(rows, base_rows):
        print("   %2d 帧  | %-6s %2d 帧  %6s px            | %-6s"
              % (r["k"], "保持" if r["id_kept"] else "丢失",
                 r["predicted_frames"], r["mean_err_px"], 
                 "保持" if b["id_kept"] else "丢失"))

    # ---- 追加：放开 max_age，看误差一直涨到多少帧才不可信 ----
    # 上表里 k>10 时轨迹已被删，误差只统计到 10 帧；要回答「为什么是 10 帧」，
    # 必须把预测一直跑到 25 帧，看误差曲线在哪一帧穿过可接受线。
    long_run = {}
    dets_mod = blank_track(frames, tid, ref, 60, 25)
    per_frame_long, _ = run(dets_mod, max_age=30)
    for f in range(60, 60 + 25):
        gt = next((b for i, b, _ in ref[f] if i == tid), None)
        pr = next((b for i, b, _ in per_frame_long[f] if i == tid), None)
        if gt is None or pr is None:
            long_run[f - 60 + 1] = float("nan")
        else:
            gc = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
            pc = ((pr[0] + pr[2]) / 2, (pr[1] + pr[3]) / 2)
            long_run[f - 60 + 1] = float(np.hypot(gc[0] - pc[0], gc[1] - pc[1]))

    print("\n放开 max_age=30、连续预测 25 帧的误差（找 10 帧的依据）：")
    for n in sorted(long_run):
        if n % 5 == 0 or n <= 3:
            print("   第 %2d 帧  %.1f px" % (n, long_run[n]))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "occlusion.json"), "w",
              encoding="utf-8") as f:
        json.dump({"target_id": tid, "kalman": rows, "iou_only": base_rows,
                   "long_run_px": long_run},
                  f, ensure_ascii=False, indent=2)

    # ---- 曲线：误差随遮挡帧数增长 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from matplotlib import font_manager
        for cand in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                     "/mnt/c/Windows/Fonts/msyh.ttc"):
            if os.path.exists(cand):
                font_manager.fontManager.addfont(cand)
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=cand).get_name()
                break

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k, errs in curve.items():
            xs = list(range(1, len(errs) + 1))
            ax.plot(xs, errs, marker="o", ms=3, label="遮挡 %d 帧" % k)
        lx = [n for n in sorted(long_run) if not np.isnan(long_run[n])]
        ly = [long_run[n] for n in lx]
        if lx:
            ax.plot(lx, ly, "k--", lw=1.2, label="放开上限，连续预测")
        ax.axhline(20, ls="--", c="crimson", lw=1,
                   label="20px 参考线（约目标框短边 1/2）")
        ax.axvline(10, ls=":", c="gray", lw=1, label="max_age = 10 帧")
        ax.set_xlabel("遮挡持续帧数")
        ax.set_ylabel("预测中心误差 (px)")
        ax.set_title("预测误差随遮挡帧数的增长")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "occlusion_curve.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("\n曲线 -> %s" % out)
    except Exception as exc:
        print("[提示] 画图失败：%s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
