#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test 集评估：mAP + 逐类 P/R/F1 + 混淆矩阵 + 失败样本（题面 §5）。

用法::

    python3 eval.py                      # 默认评 data/splits/test.txt
    python3 eval.py --part val           # 评验证集
    python3 eval.py --conf 0.25          # 换置信度门槛看 P/R 的变化

输出（全部写进 results/）：
    eval_metrics.json      mAP@0.5 / @0.5:0.95 / 逐类 AP / 逐类 P R F1
    eval_report.txt        人能直接读的表
    confusion_matrix.csv   行=真值类别，列=预测类别，最后一列/行是背景
    confusion.png          混淆矩阵热图
    failures.csv           每张图的 TP/FP/FN，按错得最多排序
    failures/              错得最多的前若干张，画了红框（预测）和绿框（真值）

注意 §10：test 集只在最终报告时用一次。调参请看 val，不要反复跑这个脚本。
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train import Detector, TrainingLoop  # noqa: E402


def load_gt(pictures: str):
    """读 annotations.json，返回 {文件名: [[x1,y1,x2,y2,类别], ...]} 与类别名。

    只读框，不读图——一次把 378 张原图读进内存会吃掉 2 GB 以上。
    """
    with open(pictures, encoding="utf-8") as fh:
        blob = json.load(fh)
    names = {i: n for i, n in enumerate(blob.get("class_names") or [])}
    gt = {}
    for item in blob["items"]:
        f = item.get("file")
        boxes = [[float(v) for v in b] for b in item.get("boxes", [])]
        gt[f] = boxes
    return gt, names


def iou_one_to_many(a, many):
    """一个框与一组框的 IoU。"""
    import numpy as np
    if len(many) == 0:
        return np.zeros(0)
    x1 = np.maximum(a[0], many[:, 0])
    y1 = np.maximum(a[1], many[:, 1])
    x2 = np.minimum(a[2], many[:, 2])
    y2 = np.minimum(a[3], many[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    a2 = np.clip(many[:, 2] - many[:, 0], 0, None) * np.clip(
        many[:, 3] - many[:, 1], 0, None)
    return inter / np.maximum(a1 + a2 - inter, 1e-12)


def match(preds, gts, thr=0.5, class_aware=True):
    """按分数从高到低贪心配对，返回 (tp_flags, 配对结果, 未匹配的真值下标)。

    class_aware=True  类别必须一致才算命中（算 P/R/F1 用）
    class_aware=False 只看框重叠（做混淆矩阵用：先判「框对不对」，再看
                      「类别判对没有」，两件事分开，报告里才说得清）
    """
    import numpy as np
    used = np.zeros(len(gts), dtype=bool)
    hit = []          # [(pred_idx, gt_idx)]
    for i in range(len(preds)):
        if len(gts) == 0:
            break
        ious = iou_one_to_many(preds[i, :4], gts)
        ok = (~used) & (ious >= thr)
        if class_aware:
            ok &= (gts[:, 4].astype(int) == int(preds[i, 5]))
        if not ok.any():
            continue
        j = int(np.argmax(np.where(ok, ious, -1)))
        used[j] = True
        hit.append((i, j))
    left = [j for j in range(len(gts)) if not used[j]]
    return hit, left


def draw_compare(frame, preds, gts, names, path):
    """画绿框=真值、红框=预测，存一张对比图（失败样本展示用）。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(np.ascontiguousarray(frame))
    d = ImageDraw.Draw(img)
    font = None
    for p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/mnt/c/Windows/Fonts/msyh.ttc"):
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, 18)
                break
            except OSError:
                pass
    for g in gts:
        d.rectangle([g[0], g[1], g[2], g[3]], outline=(0, 210, 90), width=3)
        txt = "GT %s" % names.get(int(g[4]), int(g[4]))
        d.text((g[0] + 2, max(g[1] - 20, 0)), txt, fill=(0, 210, 90), font=font)
    for p in preds:
        d.rectangle([p[0], p[1], p[2], p[3]], outline=(230, 50, 50), width=2)
        txt = "%s %.2f" % (names.get(int(p[5]), int(p[5])), p[4])
        d.text((p[0] + 2, p[3] + 2), txt, fill=(230, 50, 50), font=font)
    img.save(path)


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description="test/val 集评估")
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--splits", default=os.path.join(HERE, "data", "splits"))
    ap.add_argument("--pictures", default=os.path.join(HERE, "annotations.json"))
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw"))
    ap.add_argument("--onnx", default=os.path.join(HERE, "models", "best.onnx"))
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "best.pt"))
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "pt"])
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="算 P/R/F1 时的分数门槛；mAP 用 0.001 全量算")
    ap.add_argument("--iou", type=float, default=0.5, help="判定命中的 IoU 阈值")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--save-failures", type=int, default=6,
                    help="保存前 N 张失败样本的对比图（题面要求至少 6 张）")
    args = ap.parse_args()

    lst = os.path.join(args.splits, args.part + ".txt")
    if not os.path.exists(lst):
        print("找不到划分清单：%s\n先跑 python3 train.py（它会写 data/splits/）"
              % lst)
        return 1
    with open(lst, encoding="utf-8") as fh:
        files = [ln.strip() for ln in fh if ln.strip()]
    gt_all, names = load_gt(args.pictures)
    nc = max(len(names), max((int(b[4]) for v in gt_all.values() for b in v),
                             default=0) + 1)
    print("评估 %s 集：%d 张图，%d 类" % (args.part, len(files), nc))

    backend = args.backend
    if backend == "auto":
        backend = "onnx" if os.path.exists(args.onnx) else "pt"
    det = Detector(weights=args.weights if backend == "pt" else None,
                   onnx_path=args.onnx if backend == "onnx" else None,
                   input_size=args.size, conf_thres=0.001,   # 先全留，后面再筛
                   nms_iou=0.45, backend=backend, class_names=names)
    det.load()

    from PIL import Image
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "failures"), exist_ok=True)

    records, gts_center = [], []
    rows, frames_cache = [], {}
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(nc)}
    conf_mat = np.zeros((nc + 1, nc + 1), dtype=np.int64)   # 最后一列/行=背景

    for idx, name in enumerate(files):
        path = os.path.join(args.raw, name)
        if not os.path.exists(path):
            print("  [跳过] 找不到 %s" % path)
            continue
        with Image.open(path) as im:
            frame = np.array(im.convert("RGB"))
        dets = det.detect(frame)
        keep = dets[dets[:, 4] >= args.conf] if len(dets) else dets

        gts = np.array(gt_all.get(name, []), dtype=np.float64).reshape(-1, 5)
        records.append((idx, (keep[:, :4], keep[:, 4], keep[:, 5].astype(int))
                        if len(keep) else (np.zeros((0, 4)), np.zeros(0),
                                           np.zeros(0, dtype=int))))
        if len(gts):
            w = gts[:, 2] - gts[:, 0]
            h = gts[:, 3] - gts[:, 1]
            gts_center.append(np.stack([gts[:, 4], gts[:, 0] + w / 2,
                                        gts[:, 1] + h / 2, w, h], axis=1))
        else:
            gts_center.append(np.zeros((0, 5)))

        # ---- 逐类 P/R/F1 ----
        hit, left = match(keep, gts, args.iou, class_aware=True)
        hit_p = {i for i, _ in hit}
        for i, _ in hit:
            per_class[int(keep[i, 5])]["tp"] += 1
        for i in range(len(keep)):
            if i not in hit_p:
                per_class[int(keep[i, 5])]["fp"] += 1
        for j in left:
            per_class[int(gts[j, 4])]["fn"] += 1

        # ---- 混淆矩阵：先配框，再看类别 ----
        hit2, left2 = match(keep, gts, args.iou, class_aware=False)
        hit2_p = {i: j for i, j in hit2}
        for i in range(len(keep)):
            if i in hit2_p:
                conf_mat[int(gts[hit2_p[i], 4]), int(keep[i, 5])] += 1
            else:
                conf_mat[nc, int(keep[i, 5])] += 1        # 背景 -> 预测
        for j in left2:
            conf_mat[int(gts[j, 4]), nc] += 1             # 真值 -> 背景（漏检）

        rows.append({"file": name, "gt": len(gts), "pred": len(keep),
                     "tp": len(hit), "fp": len(keep) - len(hit),
                     "fn": len(left)})
        frames_cache[name] = (frame, keep, gts)

    # ---- mAP（用 0.001 门槛的全量预测）----
    loop = TrainingLoop()

    class _Head:
        num_classes = nc
    loop._head = _Head()
    maps = loop._mean_ap(records, gts_center, len(records))

    # ---- 汇总 ----
    lines = ["M2-3 评估报告（%s 集，%d 张图）" % (args.part, len(records)), ""]
    lines.append("mAP@0.5        %.4f" % maps["mAP50"])
    lines.append("mAP@0.5:0.95   %.4f" % maps["mAP50-95"])
    lines.append("")
    lines.append("%-8s %8s %8s %8s %8s %8s" %
                 ("类别", "TP", "FP", "FN", "P", "R"))
    f1_table = {}
    for c in range(nc):
        s = per_class[c]
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f1_table[c] = f1
        lines.append("%-8s %8d %8d %8d %8.4f %8.4f  F1 %.4f  AP50 %s"
                     % (names.get(c, str(c)), s["tp"], s["fp"], s["fn"], p, r,
                        f1, ("%.4f" % maps["ap50"][c])
                        if c in maps["ap50"] else "-"))
    lines.append("")
    lines.append("混淆矩阵（行=真值，列=预测，最后一列/行=背景）：")
    head = "        " + "".join("%8s" % names.get(c, str(c))[:6] for c in range(nc))
    lines.append(head + "   背景")
    for i in range(nc + 1):
        row = "".join("%8d" % conf_mat[i, j] for j in range(nc))
        lines.append("%-8s" % ("背景" if i == nc else names.get(i, i)[:6])
                     + row + "%7d" % conf_mat[i, nc])
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(args.out_dir, "eval_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"part": args.part, "images": len(records),
                   "conf": args.conf, "iou": args.iou,
                   "mAP50": maps["mAP50"], "mAP50-95": maps["mAP50-95"],
                   "ap50": {int(k): v for k, v in maps["ap50"].items()},
                   "per_class": {int(c): per_class[c] for c in per_class},
                   "f1": {int(c): f1_table[c] for c in f1_table}},
                  fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "confusion_matrix.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["真值\\预测"] + [names.get(c, str(c)) for c in range(nc)]
                   + ["背景"])
        for i in range(nc + 1):
            w.writerow(["背景" if i == nc else names.get(i, str(i))]
                       + list(conf_mat[i]))

    # ---- 失败样本清单 ----
    rows.sort(key=lambda r: -(r["fp"] + r["fn"]))
    with open(os.path.join(args.out_dir, "failures.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "gt", "pred", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(rows)

    saved = 0
    for r in rows:
        if saved >= args.save_failures:
            break
        if r["fp"] + r["fn"] == 0 and r["tp"] == r["gt"]:
            continue
        frame, keep, gts = frames_cache[r["file"]]
        out = os.path.join(args.out_dir, "failures",
                           os.path.splitext(r["file"])[0] + "_fail.jpg")
        draw_compare(frame, keep, gts, names, out)
        saved += 1
    print("\n失败样本图：%d 张 -> %s" % (saved,
                                      os.path.join(args.out_dir, "failures")))

    # ---- 混淆矩阵热图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 中文字体：不设的话类别名全是方块（DejaVu 没有汉字）
        for cand in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                     "/mnt/c/Windows/Fonts/msyh.ttc"):
            if os.path.exists(cand):
                font_manager.fontManager.addfont(cand)
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=cand).get_name()
                plt.rcParams["axes.unicode_minus"] = False
                break

        labels = [names.get(c, str(c)) for c in range(nc)] + ["背景"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(conf_mat, cmap="Blues")
        ax.set_xticks(range(nc + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(nc + 1))
        ax.set_yticklabels(labels)
        ax.set_xlabel("预测")
        ax.set_ylabel("真值")
        ax.set_title("Confusion matrix")
        for i in range(nc + 1):
            for j in range(nc + 1):
                if conf_mat[i, j]:
                    ax.text(j, i, int(conf_mat[i, j]), ha="center",
                            va="center", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion.png"), dpi=130)
        plt.close(fig)
        print("混淆矩阵图 -> results/confusion.png")
    except Exception as exc:
        print("[提示] 混淆矩阵画图失败（%s），CSV 仍可用" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
