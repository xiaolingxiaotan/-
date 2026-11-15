#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用摄像头批量采集训练数据（题面 §3 要求的「自采」，顺带消掉域差距）。

为什么要有这个脚本：
    训练集是手机俯拍的桌面照，现场是笔记本前置摄像头——视角、距离、光照、
    清晰度全都不一样。要让摄像头认得出，训练集里就必须有「这个摄像头拍的」
    样本。手工一张张存太慢，这里做成按键连拍。

用法::

    cd ~/智能车一轮考核/智能车一轮考核/M2/M2-3
    source ../../.venv/bin/activate
    python3 capture_cam.py

按键：
    SPACE    拍一张
    c        连拍 5 张（摆一个姿态，稍微动一下再按）
    1~6      切换当前类别标签（只记进文件名，方便事后按类整理）
    0        切到「负样本」模式：画面里什么都不摆，专门拍会引起误检的
             背景（蚊帐、被子、白墙、空桌子）。文件名带 bg_ 前缀，
             标框时这些图**一张框都不要画**，直接保存
    + / -    调曝光
    q        退出

产物：data/raw_cam/<类别>_<序号>.jpg，拍完用 label_tool.py 标框即可
     python3 label_tool.py --folder data/raw_cam
"""
import argparse
import os
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CJK_FONT = None
for _p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/mnt/c/Windows/Fonts/msyh.ttc"):
    if os.path.exists(_p):
        CJK_FONT = _p
        break


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头采集训练数据")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "raw_cam"))
    ap.add_argument("--index", type=int, default=0, help="摄像头序号")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--exposure", type=int, default=-1,
                    help="-1 自动；>=0 锁定（暗房间 300，明亮 50~100）")
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--classes", nargs="*",
                    default=["书", "笔", "杯子", "保温杯", "吧唧", "草稿本"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print("打不开摄像头 %d" % args.index)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    expo = args.exposure
    # 曝光设置必须传 V4L2 原始枚举：1=手动，3=自动。传 0.75 会被截成 1（手动），
    # 这就是「画面一直很暗」的原因。另外不手动设置时也要显式设回自动，
    # 因为它是设备级状态，上一次进程锁过之后会一直留着。
    if expo >= 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(expo))
        cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    for _ in range(12):
        cap.read()

    win = "capture (SPACE=1  c=5  1-6=class  +/-=exposure  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(args.width * args.scale),
                     int(args.height * args.scale))

    cls_idx = 0
    shot = 0
    t_prev = time.perf_counter()
    fps = 0.0
    print("已存目录：%s" % args.out)
    print("当前类别：%s   （按 1~6 切换，按 0 切负样本）" % args.classes[cls_idx])

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            mean = float(frame.mean())
            # 画提示条（用 PIL，中文才显示得出来）
            from PIL import Image, ImageDraw, ImageFont

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(img)
            font = None
            if CJK_FONT:
                try:
                    font = ImageFont.truetype(CJK_FONT, 22)
                except OSError:
                    font = None

            head = ("FPS %.0f  已拍 %d 张  当前:%s"
                    % (fps, shot,
                       "负样本(不画框)" if cls_idx < 0 else args.classes[cls_idx]))
            d.rectangle([6, 6, 6 + len(head) * 12, 36], fill=(0, 0, 0))
            d.text((10, 10), head, fill=(0, 255, 120), font=font)
            if mean < 25:
                warn = "画面过暗 %.0f/255，请开灯" % mean
                d.rectangle([6, 42, 6 + len(warn) * 14, 72], fill=(160, 20, 20))
                d.text((10, 46), warn, fill=(255, 255, 255), font=font)
            vis = cv2.cvtColor(__import__("numpy").array(img), cv2.COLOR_RGB2BGR)

            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF

            burst = 0
            if k == ord(" "):
                burst = 1
            elif k == ord("c"):
                burst = 5
            elif k == ord("+"):
                expo = max(1, (expo if expo > 0 else 100) + 50)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(expo))
                print("曝光 -> %d" % expo)
            elif k == ord("-"):
                expo = max(1, (expo if expo > 0 else 200) - 50)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(expo))
                print("曝光 -> %d" % expo)
            elif ord("1") <= k <= ord("6"):
                cls_idx = k - ord("1")
                print("当前类别 -> %s" % args.classes[cls_idx])
            elif k == ord("0"):
                cls_idx = -1
                print("当前 -> 负样本（拍完标框时留空，一张框都不画）")

            for _ in range(burst):
                ok, f = cap.read()
                if not ok:
                    continue
                shot += 1
                tag = "bg" if cls_idx < 0 else args.classes[cls_idx]
                name = "%s_%04d.jpg" % (tag, shot)
                cv2.imwrite(os.path.join(args.out, name), f)
                print("  存 %s" % name)
            if k == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("共拍 %d 张，存在 %s" % (shot, args.out))
    print("下一步标框：python3 label_tool.py --folder %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
