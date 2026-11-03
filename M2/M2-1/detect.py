import argparse
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "normal")
PHOTO_DIR = os.path.join(HERE, "photos")  # p 键拍的原图，用于离线标定

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
PREVIEW_MAX_WIDTH = 1280

def preview(frame):
    """预览缩放到 PREVIEW_MAX_WIDTH。"""
    scale = min(1.0, PREVIEW_MAX_WIDTH / frame.shape[1])
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def wait_for_start(cap, window="camera"):
    """开场窗口：预览画面，空格开始检测，q 退出。

    开始时锁一次白平衡，避免检测过程中色温漂移。
    """
    print("按空格开始检测，q 退出；开始后按 p 拍照")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败")
            return False

        shot = preview(frame)
        cv2.putText(shot, "space: start   q: quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow(window, shot)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            return False
        if key == ord(" "):
            lock_white_balance(cap)
            return True


def lock_white_balance(cap):
    """锁定白平衡，避免检测过程中色温漂移。"""
    wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if 0 < wb < 10000:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        print(f"白平衡已锁定: {wb:.0f}K")


class Preprocessor:
    """预处理：高斯滤波去噪，输出 uint8 BGR。

    放在检测链最前面：raw -> Preprocessor.apply_gaussian() -> LightDetector。
    """

    GAUSSIAN_KSIZE = 5      # 高斯核边长，必须是奇数
    GAUSSIAN_SIGMA = 0.0    # 0 表示由核大小自动推算

    def apply_gaussian(self, frame, ksize=GAUSSIAN_KSIZE,
                       sigma=GAUSSIAN_SIGMA):
        """对 BGR 帧做高斯滤波，抑制随机噪声和 MJPG 压缩振铃。"""
        k = int(ksize)
        if k < 1:
            raise ValueError("ksize 必须 >= 1")
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), sigma)


class LightDetector:
    """灯箱检测：灯罩几何定位 + 中心/外圈 S 比值与红灯 H 判亮灭。

    每帧独立，灯箱在画面里移动也能锁住：灯罩本身始终是饱和彩色圆，
    用黄、绿两盏定位，红灯位置外推；亮灭看灯罩——
    黄/绿看中心与外圈的 S 比值，红看色相 H。
    """

    LAMP_ORDER = ("yellow", "green", "red")   # 物理顺序：从上到下

    # 灯罩几何定位（每帧独立，不依赖上一帧位置）
    # 黄、绿各自用色相掩码找圆，绿灯亮时的光晕再大也不会把黄灯吞掉；
    # 红不靠自己的色块，直接由黄、绿圆心外推。
    LENS_S_MIN = 80              # 定位用的饱和度下限
    LENS_V_MIN = 50              # 太暗的像素不参与
    LENS_MIN_AREA = 2500         # 候选圆最小面积（px）
    LENS_MIN_RADIUS = 25
    LENS_MAX_CAND_RADIUS = 360   # 候选圆半径上限（发光晕可远超真实灯罩）
    LENS_MIN_CIRCULARITY = 0.4   # 面积 / 外接圆面积
    LENS_SURROUND_V = 120        # 外圈 1.15~1.6R 中位 V 下限（白板背景）
    # 昏暗自适应：两个 V 阈值只会在暗场景往下放，正常光时仍是 50/120
    LENS_DARK_SAMPLE_STEP = 8    # 全图隔点取 V，估计环境亮度
    LENS_V_MIN_FLOOR = 15
    LENS_V_MIN_SCALE = 0.40
    LENS_SURROUND_V_FLOOR = 30
    LENS_SURROUND_V_SCALE = 1.0
    LENS_HUE_YELLOW = (15, 45)
    LENS_HUE_GREEN = (45, 110)
    LENS_MIN_SPAN = 80           # 黄绿相邻间距范围（px）
    LENS_MAX_SPAN = 2000
    # 候选圆尺寸只做宽松校验，真实灯罩半径统一由 span 推：
    # R = LENS_RADIUS_RATIO * span。发光晕只会让候选圆偏大，所以
    # 绿色候选不设上限，黄色上限也放宽。
    LENS_CAND_MIN_RATIO = 0.5    # 候选半径 >= 0.5R
    LENS_CAND_MAX_RATIO = 2.5    # 黄色候选半径 <= 2.5R
    LENS_TILT_MAX = 1.0          # |dx|/dy 上限（灯柱近似竖直，约 45°）
    LENS_RADIUS_RATIO = 0.36     # 三灯统一半径 = 0.36 * 相邻间距
    LENS_SAMPLE_RATIO = 0.8      # 红灯取样盘 = 0.8R（黄绿用 RATIO_CENTER）
    LENS_HOLD_FRAMES = 3         # 偶尔锁不到时沿用上帧位置的最大帧数

    # 黄/绿亮灭（中心 S / 外圈 S 的比值）：灯亮时中心被冲白、外圈发光环
    # 仍饱和，比值塌下来；灯灭时整个灯罩同色，比值≈1。强光/昏暗都成立。
    RATIO_CENTER = 0.4           # 中心取样盘 = 0.4R
    RATIO_RIM = (0.85, 1.15)     # 外圈取样环（亮灯白心会扩到 0.8R 外）
    RATIO_RIM_PERCENTILE = 60    # 外圈取 60 分位，避开白心/白板
    RATIO_RIM_S_FLOOR = 15.0     # 外圈 S 太低说明整盏冲白，这帧不投票
    RATIO_ON_MID = 0.95          # 未亮时 < 0.95 判亮
    RATIO_OFF_MID = 1.15         # 已亮时 > 1.15 才回落为灭
    S_CONFIRM_FRAMES = 3         # 连续多少帧才翻转状态

    # 红灯改用色相判亮灭：亮灯发橙 H≈26，灭灯是品红镜片 H≈172。
    # H 跨 0 回绕，先折成 d = H if H<=90 else H-180，亮 ≈ +26，灭 ≈ -6。
    # S 在红灯上不可靠：灭 S≈100，亮时过曝白心 S 可低到 43。
    RED_H_ON = 10.0              # d > 10 判亮
    RED_H_OFF = 5.0              # 已亮时 d < 5 才回落为灭
    RED_H_S_FLOOR = 40           # 算 H 只用 S 够高的像素，避开过曝白心
    RED_H_MIN_FRAC = 0.35        # 有效像素占比低于此值时不投票，保持原状态
    RED_H_KEEP = (45, 140)       # 只保留红色相：H<=45 或 H>=140（排除旁边绿光）

    def __init__(self):
        # 灯罩定位保持 + 亮灭迟滞状态
        self.last_lenses = None
        self.lens_miss = 0
        self.s_states = {c: "unknown" for c in self.LAMP_ORDER}
        self.s_votes = {c: 0 for c in self.LAMP_ORDER}

    def _lens_candidates(self, hsv, hue_range, v_min):
        """在单一色相范围内找饱和圆形候选，返回 [{center, radius}]。

        黄、绿分开做掩码，绿灯亮时的绿色光晕再大也不会把黄色灯罩并进
        同一个连通域。候选圆的半径只用于后续宽松校验，不作为灯罩尺寸。
        """
        mask = cv2.inRange(
            hsv, (hue_range[0], self.LENS_S_MIN, int(round(v_min))),
            (hue_range[1], 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.LENS_MIN_AREA:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            if r < self.LENS_MIN_RADIUS or r > self.LENS_MAX_CAND_RADIUS:
                continue
            if area / (np.pi * r * r + 1e-6) < self.LENS_MIN_CIRCULARITY:
                continue
            candidates.append({
                "center": np.array([cx, cy], np.float64),
                "radius": float(r),
            })
        return candidates

    def detect_lenses(self, hsv=None, frame=None):
        """逐帧定位三个灯罩，返回 {颜色: {center, radius}}。

        每帧独立，不依赖上一帧位置，灯箱在画面里移动也能锁住：
          1. 黄、绿各用自己的色相掩码找饱和圆形连通域。绿灯亮时绿色光晕
             再大也不会和黄灯连成一块；暗场景按全图 V 中位数下放亮度阈值；
          2. 黄在上、绿在下配成一对，三灯半径统一由物理间距推：
             R = 0.36 * span。候选圆自己的半径只做宽松校验，因为发光晕
             只会把它撑大；
          3. 黄、绿、外推红三个灯心在半径 R 处的外圈都必须是白板，
             否则这组黄绿是误配；红灯位置由 绿 + (绿 - 黄) 外推。

        锁不到返回 {}；短时间锁不到时沿用上一帧位置（最多 LENS_HOLD_FRAMES 帧）。
        """
        if hsv is None:
            if frame is None:
                raise ValueError("hsv 和 frame 至少要给一个")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        # 用全图 V 中位数估计环境亮度：正常光时两个阈值仍是 50/120，
        # 只有变暗时才往下放，保证正常光行为不变。
        v_ref = float(np.median(v[::self.LENS_DARK_SAMPLE_STEP,
                                   ::self.LENS_DARK_SAMPLE_STEP]))
        v_min = min(self.LENS_V_MIN,
                    max(self.LENS_V_MIN_FLOOR,
                        self.LENS_V_MIN_SCALE * v_ref))
        surround_v = min(self.LENS_SURROUND_V,
                         max(self.LENS_SURROUND_V_FLOOR,
                             self.LENS_SURROUND_V_SCALE * v_ref))

        yellows = self._lens_candidates(hsv, self.LENS_HUE_YELLOW, v_min)
        greens = self._lens_candidates(hsv, self.LENS_HUE_GREEN, v_min)

        H, W = v.shape
        best = None
        for y in yellows:
            for g in greens:
                span = float(np.linalg.norm(g["center"] - y["center"]))
                if span < self.LENS_MIN_SPAN or span > self.LENS_MAX_SPAN:
                    continue
                dy = float(g["center"][1] - y["center"][1])
                if dy <= 0:
                    continue        # 物理顺序：黄在上、绿在下
                dx = float(g["center"][0] - y["center"][0])
                if abs(dx) > self.LENS_TILT_MAX * dy:
                    continue        # 灯柱近似竖直，排除斜穿画面的误配

                radius = self.LENS_RADIUS_RATIO * span
                if not (self.LENS_CAND_MIN_RATIO * radius
                        <= y["radius"]
                        <= self.LENS_CAND_MAX_RATIO * radius):
                    continue
                # 绿色是发光灯，候选圆可能被光晕撑得很大，只校验下限
                if g["radius"] < self.LENS_CAND_MIN_RATIO * radius:
                    continue

                center_r = g["center"] + (g["center"] - y["center"])
                if not (0 <= center_r[0] < W and 0 <= center_r[1] < H):
                    continue

                # 三个灯心在物理半径 R 处的外圈都应该是白板
                if (self._annulus_median(v, y["center"], radius)
                        < surround_v):
                    continue
                if (self._annulus_median(v, g["center"], radius)
                        < surround_v):
                    continue
                if self._annulus_median(v, center_r, radius) < surround_v:
                    continue

                # 光晕只会让候选半径变大，取较小的那个更接近真实灯罩
                small_r = min(y["radius"], g["radius"])
                score = (abs(small_r / radius - 1.0)
                         + 0.3 * abs(y["radius"] - g["radius"]) / span)
                if best is None or score < best[0]:
                    best = (score, y, g, span)

        if best is None:
            return self._lens_fallback()

        _, y, g, span = best
        radius = self.LENS_RADIUS_RATIO * span
        center_r = g["center"] + (g["center"] - y["center"])
        lenses = {
            "yellow": {"center": y["center"], "radius": radius},
            "green": {"center": g["center"], "radius": radius},
            "red": {"center": center_r, "radius": radius},
        }
        self.lens_miss = 0
        self.last_lenses = lenses
        return lenses

    def reset_states(self):
        """清空灯罩定位保持和亮灭迟滞（换场景/重新锁 L 时调用）。"""
        self.last_lenses = None
        self.lens_miss = 0
        for color in self.LAMP_ORDER:
            self.s_states[color] = "unknown"
            self.s_votes[color] = 0

    def _lens_fallback(self):
        """本帧没锁到：短时间沿用上一帧位置，超时则清空。"""
        self.lens_miss += 1
        if (self.last_lenses is not None
                and self.lens_miss <= self.LENS_HOLD_FRAMES):
            return self.last_lenses
        self.last_lenses = None
        return {}

    @staticmethod
    def _annulus_median(plane, center, radius, step=2):
        """圆心外 1.15~1.6R 环内的中位亮度，用来判断灯罩是否装在白板上。

        step>1 时在环上隔点取样，精度足够且快很多。
        """
        H, W = plane.shape
        cx, cy = int(round(center[0])), int(round(center[1]))
        r0 = max(1, int(round(radius * 1.15)))
        r1 = max(r0 + 1, int(round(radius * 1.6)))
        x0, y0 = max(0, cx - r1), max(0, cy - r1)
        x1, y1 = min(W, cx + r1 + 1), min(H, cy + r1 + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = (dist >= r0) & (dist <= r1)
        if not np.any(ring):
            return 0.0
        return float(np.median(plane[y0:y1:step, x0:x1:step][ring]))

    def judge_lens_state(self, hsv, lenses=None):
        """在三个灯罩上判亮灭。

        黄/绿：中心 0.4R 的 S 中位数 / 外圈 0.85~1.15R 的 S 60 分位。
            灯亮时中心冲白、发光环仍饱和，比值塌下来；灯灭时整盏同色，
            比值≈1。强光把灯罩整体照亮时这个比值仍然分得开。
        红：S 不可靠（灭≈100，亮时白心可低到 43），改用色相——
            亮灯发橙 H≈26，灭灯品红 H≈172，折成 d 后分成正负两侧。
            只统计红色相像素，避免旁边绿灯亮时把 H 拉到绿色区。
        状态翻转有迟滞 + 连续帧确认，短暂抖动不会来回跳。
        """
        if lenses is None:
            lenses = self.last_lenses or {}

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        H, W = s.shape
        states = {}
        detail = {}

        for color in self.LAMP_ORDER:
            lens = lenses.get(color)
            if lens:
                cx, cy = lens["center"]
                radius = float(lens["radius"])
                r = max(1, int(round(radius * self.LENS_SAMPLE_RATIO)))
                icx, icy = int(round(cx)), int(round(cy))
                x0, y0 = max(0, icx - r), max(0, icy - r)
                x1, y1 = min(W, icx + r + 1), min(H, icy + r + 1)
                # 外推出来的灯心可能在画面外，裁完没有有效区域就当没锁到
                if x1 <= x0 or y1 <= y0:
                    lens = None

            if not lens:
                states[color] = "unknown"
                detail[color] = None
                self.s_states[color] = "unknown"
                self.s_votes[color] = 0
                continue

            if color == "red":
                disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.circle(disk, (icx - x0, icy - y0), r, 255, -1)
                hit = disk > 0
                sc = s[y0:y1, x0:x1]
                hc = h[y0:y1, x0:x1]
                value_s = float(np.median(sc[hit]))
                # 只用 S 够高、且色相在红色区的像素算 H：
                # 过曝白心 H 无意义，旁边绿灯的绿光也不能算进来。
                valid = hit & (sc >= self.RED_H_S_FLOOR)
                valid &= ((hc <= self.RED_H_KEEP[0])
                          | (hc >= self.RED_H_KEEP[1]))
                frac = float(np.count_nonzero(valid)) / max(
                    int(np.count_nonzero(hit)), 1)
                if frac >= self.RED_H_MIN_FRAC:
                    med = float(np.median(hc[valid]))
                    d_val = med if med <= 90 else med - 180.0
                    state = self._lens_state_vote(color, d_val, is_hue=True)
                else:
                    # 整个灯罩都冲白了：这帧不投票，保持原状态
                    d_val = None
                    state = self.s_states.get(color, "unknown")
                d = {
                    "center": (float(cx), float(cy)),
                    "radius": radius,
                    "value": d_val,
                    "threshold": float(self.RED_H_ON),
                    "pixels": int(np.count_nonzero(hit)),
                    "mode": "H",
                    "S": value_s,
                    "frac": frac,
                }
            else:
                # 黄/绿：中心 S / 外圈 S，外圈要放宽到 1.15R 才包得住发光环
                rr = max(1, int(round(radius * self.RATIO_RIM[1])))
                rx0, ry0 = max(0, icx - rr), max(0, icy - rr)
                rx1, ry1 = min(W, icx + rr + 1), min(H, icy + rr + 1)
                yy, xx = np.ogrid[ry0:ry1, rx0:rx1]
                dist = np.sqrt((xx - icx) ** 2 + (yy - icy) ** 2)
                sc = s[ry0:ry1, rx0:rx1]
                center_mask = dist <= radius * self.RATIO_CENTER
                rim_mask = ((dist >= radius * self.RATIO_RIM[0])
                            & (dist <= radius * self.RATIO_RIM[1]))
                center_s = (float(np.median(sc[center_mask]))
                            if np.any(center_mask) else 0.0)
                rim_s = (float(np.percentile(sc[rim_mask],
                                             self.RATIO_RIM_PERCENTILE))
                         if np.any(rim_mask) else 0.0)
                if rim_s < self.RATIO_RIM_S_FLOOR:
                    # 外圈都冲白了，比值没有意义：这帧不投票
                    d_val = None
                    state = self.s_states.get(color, "unknown")
                else:
                    d_val = center_s / max(rim_s, 1.0)
                    state = self._lens_state_vote(color, d_val)
                d = {
                    "center": (float(cx), float(cy)),
                    "radius": radius,
                    "value": d_val,
                    "threshold": float(self.RATIO_ON_MID),
                    "pixels": int(np.count_nonzero(center_mask)),
                    "mode": "ratio",
                    "center_S": center_s,
                    "rim_S": rim_s,
                }

            states[color] = state
            detail[color] = d
        return states, detail

    def _lens_state_vote(self, color, value, is_hue=False):
        """按判据值 + 迟滞 + 连续帧确认更新单盏灯的亮灭。"""
        cur = self.s_states.get(color, "unknown")
        if is_hue:
            # 红：d 为正（橙）是亮，为负（品红）是灭
            if cur == "on":
                cand = "on" if value >= self.RED_H_OFF else "off"
            else:
                cand = "on" if value > self.RED_H_ON else "off"
        else:
            if cur == "on":
                cand = "on" if value <= self.RATIO_OFF_MID else "off"
            else:
                cand = "on" if value < self.RATIO_ON_MID else "off"

        if cur == "unknown" or cand == cur:
            self.s_states[color] = cand
            self.s_votes[color] = 0
            return cand

        self.s_votes[color] += 1
        if self.s_votes[color] >= self.S_CONFIRM_FRAMES:
            self.s_states[color] = cand
            self.s_votes[color] = 0
            return cand
        return cur


class Output:
    """输出类：图上标注 + 控制台数据 + 结果保存。

    source:
      "camera"  按空格把当前标注帧存为 save_dir/annotated_XX.png
      "image"   直接写 save_dir/<stem>_annotated.png
      "video"   每帧写入 save_dir/<stem>_annotated.mp4，结束时调 close()
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICK = 2
    LINE_THICK = 2

    ORDER = LightDetector.LAMP_ORDER   # 标注/打印顺序与物理顺序一致
    LAMP_COLORS = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
    }
    ON_COLOR = (0, 255, 0)
    OFF_COLOR = (180, 180, 180)
    UNKNOWN_COLOR = (0, 165, 255)
    TEXT_BG = (0, 0, 0)
    FPS_PRINT_INTERVAL = 1.0  # 状态不变时，至少每秒打一行帧率

    def __init__(self, source="camera", input_path=None,
                 save_dir=RESULTS_DIR, fps=CAPTURE_FPS):
        if source not in ("camera", "image", "video"):
            raise ValueError('source 必须是 "camera"/"image"/"video"')
        self.source = source
        self.input_path = input_path
        self.save_dir = save_dir
        self.video_fps = fps

        self.writer = None
        self.video_path = None
        self.saved_image = False
        self.last_states = None

        self.out_fps = None
        self._last_time = time.monotonic()
        self._last_print = 0.0

        os.makedirs(self.save_dir, exist_ok=True)

    # 板块 1：图上标注
    def draw(self, frame, states, detail):
        """在图上画灯的位置/大小/范围/亮灭，返回标注后的新图。

        十字架交点=圆心，臂长=半径，所以位置和大小一个标记就够；
        外面再画一个圆把灯的范围圈出来。
        """
        img = frame.copy()

        for color in self.ORDER:
            d = None if detail is None else detail.get(color)
            if not d:
                continue

            state = "unknown" if states is None else states.get(color, "unknown")
            cx, cy = (int(round(v)) for v in d["center"])
            r = max(1, int(round(d["radius"])))
            lamp_bgr = self.LAMP_COLORS[color]

            # 十字架：臂长 = 半径
            self._line(img, (cx - r, cy), (cx + r, cy), lamp_bgr)
            self._line(img, (cx, cy - r), (cx, cy + r), lamp_bgr)
            # 圈出灯的实际范围
            self._circle(img, (cx, cy), r, lamp_bgr)

            state_text = "?" if state == "unknown" else state.upper()
            label = f"{color.upper()} {state_text} r={r}"
            self._label(img, label, (cx - r, cy - r - 8),
                        self._state_color(state))

        return img

    # 板块 2：控制台数据
    def report(self, states, detail, force=False):
        """输出帧率、每个灯的状态、圆心、半径、判据值和阈值。

        每帧都会更新输出帧率；默认在状态变化或距上次打印超过
        FPS_PRINT_INTERVAL 秒时打印，force=True 每帧都打印。
        """
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            inst = 1.0 / dt
            self.out_fps = (inst if self.out_fps is None
                            else 0.9 * self.out_fps + 0.1 * inst)

        if states is None:
            states = {}
        key = tuple(states.get(c, "unknown") for c in self.ORDER)
        changed = key != self.last_states
        self.last_states = key
        if (not force and not changed
                and now - self._last_print < self.FPS_PRINT_INTERVAL):
            return
        self._last_print = now

        modes = []
        parts = []
        for color in self.ORDER:
            d = None if detail is None else detail.get(color)
            state = states.get(color, "unknown")
            if not d:
                parts.append(f"{color}:{state}")
                continue
            mode = d.get("mode")
            if mode and mode not in modes:
                modes.append(mode)
            cx, cy = d["center"]
            if mode == "H":
                hv = "--" if d["value"] is None else f"{d['value']:+.0f}"
                parts.append(
                    f"{color}:{state} c=({cx:.0f},{cy:.0f}) "
                    f"r={d['radius']:.1f} "
                    f"H={hv}/{d['threshold']:.0f} "
                    f"S={d['S']:.0f} frac={d['frac']:.2f} "
                    f"px={d['pixels']}"
                )
            elif mode == "ratio":
                rv = "--" if d["value"] is None else f"{d['value']:.2f}"
                parts.append(
                    f"{color}:{state} c=({cx:.0f},{cy:.0f}) "
                    f"r={d['radius']:.1f} "
                    f"ratio={rv}/{d['threshold']:.2f} "
                    f"Sc={d['center_S']:.0f} Sr={d['rim_S']:.0f} "
                    f"px={d['pixels']}"
                )
            else:
                parts.append(
                    f"{color}:{state} c=({cx:.0f},{cy:.0f}) "
                    f"r={d['radius']:.1f} "
                    f"S={d['value']:.0f}/{d['threshold']:.0f} "
                    f"px={d['pixels']}"
                )

        fps_text = "--" if self.out_fps is None else f"{self.out_fps:.1f}"
        mode_text = "+".join(modes) if modes else "-"
        line = f"[{mode_text}] fps={fps_text} | " + " | ".join(parts)
        print(line, flush=True)

    # 板块 3：保存
    def save(self, frame):
        """保存当前标注帧。摄像头由空格触发，图片/视频自动触发。"""
        if self.source == "camera":
            return self._save_numbered(frame)
        if self.source == "image":
            return self._save_image(frame)
        return self._save_video(frame)

    def snapshot(self, frame):
        """按 p 拍照：把原始帧存成 photos/photo_XX.png，用于离线标定。"""
        os.makedirs(PHOTO_DIR, exist_ok=True)
        index = next_index(PHOTO_DIR, prefix="photo_")
        img_path = os.path.join(PHOTO_DIR, f"photo_{index:02d}.png")
        if not cv2.imwrite(img_path, frame):
            print(f"拍照保存失败: {img_path}")
            return None
        print(f"已拍照: {img_path}")
        return img_path

    def handle_key(self, key, frame):
        """摄像头模式按键：空格保存当前帧。返回 True 表示 q/Esc 退出。"""
        if self.source == "camera" and key == ord(" "):
            self.save(frame)
        return key in (ord("q"), 27)

    def close(self):
        """视频模式收尾；摄像头/图片模式无操作。"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"已保存视频: {self.video_path}")

    def _save_numbered(self, frame):
        index = next_index(self.save_dir)
        path = os.path.join(self.save_dir, f"annotated_{index:02d}.png")
        if not cv2.imwrite(path, frame):
            print(f"保存失败: {path}")
            return None
        print(f"已保存: {path}")
        return path

    def _save_image(self, frame):
        if self.saved_image:
            return None
        stem = os.path.splitext(
            os.path.basename(self.input_path or "image"))[0]
        path = os.path.join(self.save_dir, f"{stem}_annotated.png")
        if not cv2.imwrite(path, frame):
            print(f"保存失败: {path}")
            return None
        self.saved_image = True
        print(f"已保存: {path}")
        return path

    def _save_video(self, frame):
        if self.writer is None:
            stem = os.path.splitext(
                os.path.basename(self.input_path or "video"))[0]
            path = os.path.join(self.save_dir, f"{stem}_annotated.mp4")
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h))
            if not self.writer.isOpened():
                print(f"无法创建视频: {path}")
                self.writer = None
                return None
            self.video_path = path
            print(f"正在写入: {path}")
        self.writer.write(frame)
        return self.video_path

    @classmethod
    def _line(cls, img, p1, p2, color):
        # 先黑后彩，白面板上也能看清
        cv2.line(img, p1, p2, cls.TEXT_BG, cls.LINE_THICK + 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, cls.LINE_THICK, cv2.LINE_AA)

    @classmethod
    def _circle(cls, img, center, r, color):
        cv2.circle(img, center, r, cls.TEXT_BG, cls.LINE_THICK + 2, cv2.LINE_AA)
        cv2.circle(img, center, r, color, cls.LINE_THICK, cv2.LINE_AA)

    @classmethod
    def _label(cls, img, text, org, color):
        (tw, th), base = cv2.getTextSize(
            text, cls.FONT, cls.FONT_SCALE, cls.FONT_THICK)
        x = min(max(0, org[0]), max(0, img.shape[1] - tw - 6))
        y = min(max(th + 4, org[1]), img.shape[0] - base - 2)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base),
                      cls.TEXT_BG, -1)
        cv2.putText(img, text, (x + 3, y - 2), cls.FONT, cls.FONT_SCALE,
                    color, cls.FONT_THICK, cv2.LINE_AA)

    @classmethod
    def _state_color(cls, state):
        if state == "on":
            return cls.ON_COLOR
        if state == "off":
            return cls.OFF_COLOR
        return cls.UNKNOWN_COLOR


# 命令行读取函数
def parse_args():
    parser = argparse.ArgumentParser(description="红绿灯检测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="输入图片路径")
    source.add_argument("--video", help="输入视频路径")
    source.add_argument("--camera", type=int, help="摄像头索引，如 0")
    parser.add_argument("--show", action="store_true", help="显示结果窗口")
    return parser.parse_args()


def next_index(save_dir, prefix="annotated_"):
    """从已有文件名接着编号，避免重跑时覆盖之前的图片。"""
    used = []
    for name in os.listdir(save_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".png" and stem.startswith(prefix):
            try:
                used.append(int(stem[len(prefix):]))
            except ValueError:
                pass
    return max(used, default=0) + 1


def main():
    args = parse_args()

    detector = LightDetector()
    pre = Preprocessor()

    def process(frame, out):
        """单帧完整流水线：预处理 -> 灯罩几何定位 -> 中心/外圈 S 比值与 H 判亮灭 -> 显示。"""
        frame = pre.apply_gaussian(frame)

        # 每帧只转一次 BGR->HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 每帧独立定位三个灯罩，再用黄绿比值 / 红灯 H 判亮灭
        lenses = detector.detect_lenses(hsv=hsv)
        states, detail = detector.judge_lens_state(hsv, lenses)

        annotated = out.draw(frame, states, detail)
        out.report(states, detail)
        return annotated

    # ---------- 摄像头 ----------
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("无法打开摄像头（WSL 下需要先用 usbipd 把摄像头挂进来）")
            return

        # USB/IP 下必须走 MJPG，1080p 只能跑 15 fps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              f"@{int(cap.get(cv2.CAP_PROP_FPS))}fps")

        # 空格开始，q 退出（开始时锁白平衡）
        if not wait_for_start(cap):
            cap.release()
            cv2.destroyAllWindows()
            return
        cv2.destroyAllWindows()
        detector.reset_states()

        out = Output(source="camera")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("读取画面失败")
                    break

                annotated = process(frame, out)
                cv2.imshow("detect", preview(annotated))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    out.snapshot(frame)
                if out.handle_key(key, annotated):
                    break
        finally:
            cap.release()
            out.close()
            cv2.destroyAllWindows()
        return

    # ---------- 图片 ----------
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"无法读取图片: {args.image}")
            return
        print(f"图片: {args.image} ({frame.shape[1]}x{frame.shape[0]})")

        out = Output(source="image", input_path=args.image)
        annotated = process(frame, out)
        out.save(annotated)

        if args.show:
            cv2.imshow("detect", preview(annotated))
            print("按任意键退出")
            cv2.waitKey(0)
        out.close()
        cv2.destroyAllWindows()
        return

    # ---------- 视频 ----------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法读取视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = CAPTURE_FPS

    ok, frame = cap.read()
    if not ok:
        print(f"视频没有可读帧: {args.video}")
        cap.release()
        return
    print(f"视频: {args.video} ({frame.shape[1]}x{frame.shape[0]}"
          f"@{fps:.1f}fps)")

    out = Output(source="video", input_path=args.video, fps=fps)
    try:
        while ok:
            annotated = process(frame, out)
            out.save(annotated)

            if args.show:
                cv2.imshow("detect", preview(annotated))
                if out.handle_key(cv2.waitKey(1) & 0xFF, annotated):
                    break
            ok, frame = cap.read()
    finally:
        cap.release()
        out.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
