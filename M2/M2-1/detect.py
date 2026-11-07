import argparse
import os
import time

import cv2
import numpy as np

# 路径定义
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "normal")
PHOTO_DIR = os.path.join(HERE, "photos")  # p 键拍的原图，用于离线标定

# 摄像头采集参数
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
# PREVIEW_MAX_WIDTH：预览窗口最大宽度，超过就等比缩小。
PREVIEW_MAX_WIDTH = 1280


# 预处理类：相机接入 + 高斯滤波
class Preprocessor:
    # 预处理：相机开场/白平衡/预览缩放 + 高斯滤波，输出 uint8 BGR。
    # 放在检测链最前面：raw -> Preprocessor.apply_gaussian() -> LightDetector。

    GAUSSIAN_KSIZE = 5      
    GAUSSIAN_SIGMA = 0.0    # 0 表示由核大小自动推算

    # 把预览画面缩放到 PREVIEW_MAX_WIDTH 以内，保持宽高比。
    @staticmethod
    def preview(frame):
        scale = min(1.0, PREVIEW_MAX_WIDTH / frame.shape[1])
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    # 开场循环：显示预览，按空格锁白平衡并开始，按 q/Esc 退出。
    def wait_for_start(self, cap, window="camera"):
        print("按空格开始检测，q 退出；开始后按 p 拍照")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取画面失败")
                return False

            # shot：缩放后的预览图，只用于显示提示文字。
            shot = self.preview(frame)
            cv2.putText(shot, "按空格开始，按q退出程序", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow(window, shot)

            # key：当前按键；q退出，空格锁白平衡并开始。
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                return False
            if key == ord(" "):
                self.lock_white_balance(cap)
                return True

    # 关闭自动白平衡并把当前色温写回，避免检测过程中色温漂移。
    @staticmethod
    def lock_white_balance(cap):
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if 0 < wb < 10000:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
            print(f"白平衡已锁定: {wb:.0f}K")

    # 对 BGR 帧做高斯滤波，抑制随机噪声和 MJPG 压缩振铃。
    def apply_gaussian(self, frame, ksize=GAUSSIAN_KSIZE,
                       sigma=GAUSSIAN_SIGMA):
        k = int(ksize)
        if k < 1:
            raise ValueError("ksize 必须 >= 1")
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), sigma)


# 灯箱检测类：用黄/绿灯罩定位三个灯，再用中心/外圈 S 比值或红灯 H 判亮灭。
class LightDetector:
     # 物理顺序：黄→绿→红；图像方向随灯箱旋转
    LAMP_ORDER = ("yellow", "green", "red")  

    # 排除不符合的小噪声点
    LENS_S_MIN = 80              
    LENS_V_MIN = 50              
    LENS_MIN_AREA = 2500         
    LENS_MIN_RADIUS = 25         
    LENS_MAX_CAND_RADIUS = 360   
    LENS_MIN_CIRCULARITY = 0.4   

    # 霍夫圆检测参数
    LENS_HOUGH_DP = 1.2          # Hough 累加器分辨率
    LENS_HOUGH_SCALE = 0.5       # 掩码先缩小再 Hough，速度快 4 倍
    LENS_HOUGH_MIN_DIST = 80     # 两个圆心的最小距离（px，原图尺度）
    LENS_HOUGH_PARAM1 = 80       # Canny 高阈值
    LENS_HOUGH_PARAM2 = 30       # 圆心累加器阈值，越大越严格
    LENS_SURROUND_V = 120        # 外圈 1.15~1.6R 中位 V 下限（白板背景）

    # 昏暗自适应：两个 V 阈值只会在暗场景往下放，正常光时仍是 50/120
    LENS_DARK_SAMPLE_STEP = 8    # 全图隔点取 V，估计环境亮度
    LENS_V_MIN_FLOOR = 15        # v_min 的下限
    LENS_V_MIN_SCALE = 0.40      # v_min = 0.40 * 环境 V 中位数
    LENS_SURROUND_V_FLOOR = 30   # surround_v 的下限
    LENS_SURROUND_V_SCALE = 1.0  # surround_v = 1.0 * 环境 V 中位数
    LENS_HUE_YELLOW = (15, 45)   # 黄色灯罩的 H 范围
    LENS_HUE_GREEN = (45, 110)   # 绿色灯罩的 H 范围
    LENS_MIN_SPAN = 80           # 黄绿相邻间距范围（px）
    LENS_MAX_SPAN = 2000         # 黄绿相邻间距上限（px）

    # 候选圆尺寸只做宽松校验，真实灯罩半径统一由 span 推：
    # R = LENS_RADIUS_RATIO * span。发光晕只会让候选圆偏大，所以
    # 绿色候选不设上限，黄色上限也放宽。
    LENS_CAND_MIN_RATIO = 0.5    # 候选半径 >= 0.5R
    LENS_CAND_MAX_RATIO = 2.5    # 黄色候选半径 <= 2.5R
    LENS_RADIUS_RATIO = 0.36     # 三灯统一半径 = 0.36 * 相邻间距
    # 红灯亮时灯芯呈橙黄色，也会落进黄色掩码；靠红灯镜片外圈的洋红环，排除红灯被识别为黄灯
    LENS_RED_BEZEL_RING = (1.10, 1.35)  # 洋红镜片环的半径范围（×R）
    LENS_RED_BEZEL_S = 60        # 洋红环的 S 下限
    LENS_RED_BEZEL_V = 50        # 洋红环的 V 下限
    LENS_RED_BEZEL_H = 140       # H大于阈值视为洋红/红紫
    LENS_RED_MATCH = 0.15        # 红端候选距外推点 <= 0.15*span 才当作红灯
    LENS_RED_PRESENCE = 0.20     # 红端 0.8R 圆盘或 1.10~1.35R 环带里
                                 # 洋红占比低于此值，说明红端可能不是真红灯
    LENS_RED_PRESENCE_W = 0.5    # 红端存在性惩罚项的权重
    LENS_SAMPLE_RATIO = 0.8      # 红灯取样盘 = 0.8R（黄绿用 RATIO_CENTER）
    LENS_HOLD_FRAMES = 3         # 偶尔锁不到时沿用上帧位置的最大帧数

    # 绿灯亮时绿色光晕会把绿心沿灯轴推走，span 变大，外推红心跟着偏。
    # 若黄灯灭，黄色镜片整片可见，半径最可信：用黄色半径作主半径，
    # 再在红端 ROI 上做 V 通道 Hough 找红灯镜片本体，重新定三个圆心。
    LENS_ANCHOR_PAD = 2.2        # 红端 ROI = 外推红心 ±2.2R
    LENS_ANCHOR_RADIUS = (0.65, 1.35)  # 红端 Hough 半径范围（×R）
    LENS_ANCHOR_AXIS = (1.4, 2.6)      # 红心投影 / span 的允许范围
    LENS_ANCHOR_PERP = 0.5       # 红心横向偏移 <= 0.5R
    LENS_ANCHOR_SPAN_SHIFT = (0.8, 1.2)  # 新 span / 旧 span 的允许范围
    LENS_ANCHOR_HOUGH_DP = 1.2   # 红端 ROI Hough 累加器分辨率
    LENS_ANCHOR_HOUGH_PARAM1 = 100  # 红端 ROI Canny 高阈值
    LENS_ANCHOR_HOUGH_PARAM2 = 28   # 红端 ROI 圆心累加阈值
    LENS_ANCHOR_MAGENTA_RADIUS = (0.5, 1.8)  # 洋红镜片半径范围（×R）
    LENS_ANCHOR_MAGENTA_AREA = 300   # 洋红连通域最小面积
    LENS_ANCHOR_MAGENTA_S = 40   # 洋红连通域 S 下限
    LENS_ANCHOR_MAGENTA_V = 25   # 洋红连通域 V 下限

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
        # 上一帧锁到的三个灯罩位置，短暂丢帧时沿用。
        self.last_lenses = None
        # 连续没锁到灯罩的帧数，用来限制沿用时长。
        self.lens_miss = 0
        # 每盏灯的当前稳定状态，初值 unknown。
        self.s_states = {c: "unknown" for c in self.LAMP_ORDER}
        # s_votes：每盏灯连续投票与当前状态不同的次数。
        self.s_votes = {c: 0 for c in self.LAMP_ORDER}

    # 在单一色相范围内找饱和圆形候选，返回 center/radius 列表。
    def _lens_candidates(self, hsv, hue_range, v_min):
        # 在单一色相范围内找饱和圆形候选，返回 [{center, radius}]。
        #
        # 黄、绿分开做掩码，绿灯亮时的绿色光晕再大也不会把黄色灯罩并进
        # 同一个连通域。候选圆的半径只用于后续宽松校验，不作为灯罩尺寸。
        # mask：当前色相范围的二值掩码，只保留饱和度高、亮度够的像素。
        mask = cv2.inRange(
            hsv, (hue_range[0], self.LENS_S_MIN, int(round(v_min))),
            (hue_range[1], 255, 255))
        # 先开运算去掉小噪点，再闭运算把灯罩内部的小孔补上。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
        # contours：所有候选连通域的外轮廓。
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # candidates：通过面积/半径/圆度筛选的候选圆列表。
        candidates = []
        for contour in contours:
            # area：连通域面积；太小直接跳过。
            area = float(cv2.contourArea(contour))
            if area < self.LENS_MIN_AREA:
                continue
            # (cx,cy)：最小外接圆圆心；r：半径。
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            if r < self.LENS_MIN_RADIUS or r > self.LENS_MAX_CAND_RADIUS:
                continue
            # 圆度 = 面积 / 外接圆面积，太扁的形状不是灯罩。
            if area / (np.pi * r * r + 1e-6) < self.LENS_MIN_CIRCULARITY:
                continue
            candidates.append({
                "center": np.array([cx, cy], np.float64),
                "radius": float(r),
            })
        # 连通域候选全被背景并掉时（典型：绿灯大开 + 绿色背景），
        # 在同一张色相掩码上跑 Hough 圆，直接找灯罩的圆形边缘。
        if not candidates:
            candidates = self._hough_candidates(mask)
        return candidates

    # 在色相掩码上用 Hough 圆找灯罩，作为连通域失败时的兜底。
    @classmethod
    def _hough_candidates(cls, mask):
        # 先缩小掩码：Hough 对像素数很敏感，0.5 倍速度约快 4 倍，
        # 圆心的量化误差只有 1~2 px，足够后续配对和半径校验。
        scale = cls.LENS_HOUGH_SCALE
        small = cv2.resize(mask, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        # blurred：缩小后再模糊，压掉掩码边缘锯齿。
        blurred = cv2.GaussianBlur(small, (9, 9), 2)
        # circles：Hough 返回的 (x, y, r) 数组，可能为 None。
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=cls.LENS_HOUGH_DP,
            minDist=max(8, int(round(cls.LENS_HOUGH_MIN_DIST * scale))),
            param1=cls.LENS_HOUGH_PARAM1,
            param2=cls.LENS_HOUGH_PARAM2,
            minRadius=max(4, int(round(cls.LENS_MIN_RADIUS * scale))),
            maxRadius=max(5, int(round(cls.LENS_MAX_CAND_RADIUS * scale))))
        if circles is None:
            return []
        # candidates：把 Hough 结果换算回原图尺度，格式与连通域候选一致。
        candidates = []
        for cx, cy, r in np.round(circles[0]).astype(float):
            candidates.append({
                "center": np.array([cx / scale, cy / scale], np.float64),
                "radius": float(r / scale),
            })
        return candidates

    # 计算灯罩外圈洋红像素占比，用来识别“亮着的红灯”。
    @classmethod
    def _magenta_ring(cls, hsv, center, radius):
        # 在 1.10~1.35R 的环带里，统计 H>=140 的洋红像素占比。
        #
        # 亮红灯的橙黄色灯芯会被黄色掩码当成候选，但它镜片外圈仍是
        # 洋红色；真黄灯的外圈没有这个环。返回值只用于配对打分，
        # 与灯箱在画面里的旋转方向无关。
        H, W = hsv.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        # r0/r1：环带内外半径；x0..y1：环带外接矩形，裁到图像内。
        r0 = max(1, int(round(radius * cls.LENS_RED_BEZEL_RING[0])))
        r1 = max(r0 + 1, int(round(radius * cls.LENS_RED_BEZEL_RING[1])))
        x0, y0 = max(0, cx - r1), max(0, cy - r1)
        x1, y1 = min(W, cx + r1 + 1), min(H, cy + r1 + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        # yy/xx：矩形内每个像素的坐标；ring：落在环带内的掩码。
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = (dist2 >= r0 * r0) & (dist2 <= r1 * r1)
        if not ring.any():
            return 0.0
        # sub：环带外接矩形对应的 HSV 子图。
        sub = hsv[y0:y1, x0:x1]
        valid = (ring
                 & (sub[:, :, 1] >= cls.LENS_RED_BEZEL_S)
                 & (sub[:, :, 2] >= cls.LENS_RED_BEZEL_V))
        if valid.sum() < 10:
            return 0.0
        return float(np.mean(sub[:, :, 0][valid] >= cls.LENS_RED_BEZEL_H))

    # 计算灯罩中心圆盘里的洋红像素占比，识别“灭着的红灯”。
    @classmethod
    def _magenta_disk(cls, hsv, center, radius):
        # 灭红灯的镜片是洋红色，中心 0.8R 圆盘内会有大量 H>=140 像素；
        # 真黄灯的圆盘里没有。红灯亮时灯芯偏橙，靠 _magenta_ring 兜住。
        H, W = hsv.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        rr = max(1, int(round(radius * cls.LENS_SAMPLE_RATIO)))
        x0, y0 = max(0, cx - rr), max(0, cy - rr)
        x1, y1 = min(W, cx + rr + 1), min(H, cy + rr + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        # yy/xx：外接矩形坐标；disk：落在圆盘内的掩码。
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
        # sub：圆盘外接矩形对应的 HSV 子图。
        sub = hsv[y0:y1, x0:x1]
        valid = (disk
                 & (sub[:, :, 1] >= cls.LENS_RED_BEZEL_S)
                 & (sub[:, :, 2] >= cls.LENS_RED_BEZEL_V))
        if valid.sum() < 10:
            return 0.0
        return float(np.mean(sub[:, :, 0][valid] >= cls.LENS_RED_BEZEL_H))

    def _ratio_value(self, hsv, center, radius):
        # 单盏黄/绿的本帧中心/外圈 S 比值；外圈冲白时返回 None。
        # 公式与 judge_lens_state 相同，但不带迟滞和连续帧确认，
        # 供几何修正在同一帧内判断黄、绿是否亮着。
        H, W = hsv.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        # rr：外圈取样半径；取样区域裁到图像内。
        rr = max(1, int(round(radius * self.RATIO_RIM[1])))
        x0, y0 = max(0, cx - rr), max(0, cy - rr)
        x1, y1 = min(W, cx + rr + 1), min(H, cy + rr + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        # yy/xx：取样区域坐标；dist：到灯心的距离。
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # sc：取样区域的 S 通道。
        sc = hsv[y0:y1, x0:x1, 1]
        # center_mask：中心 0.4R 圆盘；rim_mask：0.85~1.15R 外圈。
        center_mask = dist <= radius * self.RATIO_CENTER
        rim_mask = ((dist >= radius * self.RATIO_RIM[0])
                    & (dist <= radius * self.RATIO_RIM[1]))
        if not np.any(center_mask) or not np.any(rim_mask):
            return None
        # center_s：中心 S 中位数；rim_s：外圈 S 的 60 分位。
        center_s = float(np.median(sc[center_mask]))
        rim_s = float(np.percentile(sc[rim_mask],
                                    self.RATIO_RIM_PERCENTILE))
        if rim_s < self.RATIO_RIM_S_FLOOR:
            return None
        return center_s / max(rim_s, 1.0)

    def _red_hue_value(self, hsv, center, radius):
        # 单盏红灯的本帧色相判据 d：亮 ≈ +26，灭 ≈ -6。
        # 公式与 judge_lens_state 的红灯部分相同；有效像素太少时返回 None。
        H, W = hsv.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        # r：取样盘半径；disk：圆盘掩码。
        r = max(1, int(round(radius * self.LENS_SAMPLE_RATIO)))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(W, cx + r + 1), min(H, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(disk, (cx - x0, cy - y0), r, 255, -1)
        hit = disk > 0
        # sc/hc：取样区域的 S、H 通道。
        sc = hsv[y0:y1, x0:x1, 1]
        hc = hsv[y0:y1, x0:x1, 0]
        # valid：S 够高、色相在红色区的像素，避开过曝白心和旁边绿光。
        valid = hit & (sc >= self.RED_H_S_FLOOR)
        valid &= ((hc <= self.RED_H_KEEP[0]) | (hc >= self.RED_H_KEEP[1]))
        if np.count_nonzero(valid) < (self.RED_H_MIN_FRAC
                                      * np.count_nonzero(hit)):
            return None
        # med：有效像素的 H 中位数；折回成 d 后判断正负。
        med = float(np.median(hc[valid]))
        return med if med <= 90 else med - 180.0

    def _magenta_anchor(self, hsv, center, radius):
        # 在中心附近找洋红镜片连通域，返回 (外接圆圆心, 外接圆半径)。
        # 灭红灯的镜片是实心洋红盘，圆心/半径都可用；高光会把连通域
        # 咬成月牙，圆心有偏差，所以再和红端 Hough 的圆心取平均。
        H, W = hsv.shape[:2]
        cx0, cy0 = int(round(center[0])), int(round(center[1]))
        # 只在灯心 ±2R 的 ROI 里做掩码，避免整幅 1080p 的 inRange/形态学。
        pad = max(1, int(round(2.0 * radius)))
        x0, y0 = max(0, cx0 - pad), max(0, cy0 - pad)
        x1, y1 = min(W, cx0 + pad + 1), min(H, cy0 + pad + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        # sub_hsv：灯心附近 ROI；mask：洋红像素掩码。
        sub_hsv = hsv[y0:y1, x0:x1]
        mask = cv2.inRange(
            sub_hsv,
            (self.LENS_RED_BEZEL_H, self.LENS_ANCHOR_MAGENTA_S,
             self.LENS_ANCHOR_MAGENTA_V),
            (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        rmin, rmax = self.LENS_ANCHOR_MAGENTA_RADIUS
        best = None
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < self.LENS_ANCHOR_MAGENTA_AREA:
                continue
            # x/y/w/h：当前连通域的外接矩形；sub：该连通域的二值子图。
            x, y, w, h = stats[j, :4]
            sub = (lab[y:y + h, x:x + w] == j).astype(np.uint8)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            # (cx,cy)：外接圆圆心；r：外接圆半径。
            (cx, cy), r = cv2.minEnclosingCircle(
                max(cnts, key=cv2.contourArea))
            ctr = np.array([cx + x + x0, cy + y + y0], np.float64)
            if not (rmin * radius < r < rmax * radius):
                continue
            d = float(np.linalg.norm(ctr - center))
            if d <= 0.6 * radius and (best is None or d < best[0]):
                best = (d, ctr, float(r))
        if best is None:
            return None
        return best[1], best[2]

    def _red_hough_anchor(self, hsv, y, g, radius):
        # 在外推红心附近对 V 通道做 Hough，找红灯镜片本体的圆心/半径。
        # 绿色光晕只影响绿心，红灯本体仍是完整的圆，可以把红心拉回来。
        # 返回 (center, radius) 或 None。
        yc = np.asarray(y["center"], np.float64)
        gc = np.asarray(g["center"], np.float64)
        # span：当前黄绿间距；u：黄指向绿的单位向量。
        span = float(np.linalg.norm(gc - yc))
        if span <= 1e-6:
            return None
        u = (gc - yc) / span
        # pred：当前外推红心；ROI 取 pred ± LENS_ANCHOR_PAD*R。
        pred = gc + (gc - yc)
        H, W = hsv.shape[:2]
        pad = int(round(self.LENS_ANCHOR_PAD * radius))
        x0 = max(0, int(round(pred[0])) - pad)
        y0 = max(0, int(round(pred[1])) - pad)
        x1 = min(W, int(round(pred[0])) + pad + 1)
        y1 = min(H, int(round(pred[1])) + pad + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        # roi：红端 ROI 的 V 通道；先缩小 0.5 倍再 Hough，速度快约 4 倍。
        roi = cv2.GaussianBlur(hsv[y0:y1, x0:x1, 2], (7, 7), 1.5)
        scale = self.LENS_HOUGH_SCALE
        small = cv2.resize(roi, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        # 半径范围放宽：Hough 锁到镜片内圈或外圈都可以，两个圆同心，
        # 圆心才是这里要的量；最终灯罩半径由黄灯半径/洋红镜片单独定。
        rmin, rmax = self.LENS_ANCHOR_RADIUS
        circles = cv2.HoughCircles(
            small, cv2.HOUGH_GRADIENT,
            dp=self.LENS_ANCHOR_HOUGH_DP,
            minDist=max(8, int(round(0.8 * radius * scale))),
            param1=self.LENS_ANCHOR_HOUGH_PARAM1,
            param2=self.LENS_ANCHOR_HOUGH_PARAM2,
            minRadius=max(4, int(round(rmin * radius * scale))),
            maxRadius=max(5, int(round(rmax * radius * scale))))
        if circles is None:
            return None
        amin, amax = self.LENS_ANCHOR_AXIS
        best = None
        for cx, cy, r in circles[0]:
            # 换算回原图坐标：ROI 偏移 + 0.5 倍缩放。
            c = np.array([cx / scale + x0, cy / scale + y0], np.float64)
            # rel：红心相对黄心；proj：沿灯轴投影；perp：横向偏差。
            rel = c - yc
            proj = float(np.dot(rel, u))
            perp = float(np.linalg.norm(rel - proj * u))
            if not (amin * span <= proj <= amax * span):
                continue
            if perp > self.LENS_ANCHOR_PERP * radius:
                continue
            # score：越接近 2*span、越贴近灯轴越好。
            score = abs(proj / span - 2.0) + perp / span
            if best is None or score < best[0]:
                best = (score, c, float(r / scale))
        if best is None:
            return None
        return best[1], best[2]

    def _refine_green_on(self, hsv, y, g, span, radius, yellows):
        # 绿灯亮时修正三灯几何；不需要修正时返回 None。
        # 只处理“绿灯亮”的帧，绿灯灭时一个像素都不动，保证正常光精度。
        green_ratio = self._ratio_value(hsv, g["center"], radius)
        if green_ratio is None or green_ratio >= self.RATIO_ON_MID:
            return None
        yellow_ratio = self._ratio_value(hsv, y["center"], radius)
        if yellow_ratio is None:
            return None
        # yellow_on：本帧黄灯是否亮着；决定黄色半径能不能信。
        yellow_on = yellow_ratio < self.RATIO_ON_MID

        # 红灯亮时洋红环是空心的，Hough/外接圆圆心容易被环的缺口带偏；
        # 这类帧保持原几何，只修“红灯灭”的帧，避免把正常结果改坏。
        red_pred = g["center"] + (g["center"] - y["center"])
        red_d = self._red_hue_value(hsv, red_pred, radius)
        if red_d is None or red_d > self.RED_H_ON:
            return None

        # 黄灯灭：黄色镜片整片可见，取离原黄心最近的候选作黄锚点。
        # 直接复用本帧定位时已经算好的 yellows，不再重复做掩码和 Hough。
        y_anchor = None
        if not yellow_on:
            near = [c for c in yellows
                    if float(np.linalg.norm(c["center"] - y["center"]))
                    <= 0.5 * radius]
            if near:
                y_anchor = min(near, key=lambda c: float(
                    np.linalg.norm(c["center"] - y["center"])))

        # y_new：新黄心；黄灯灭用候选圆心，黄灯亮保持原黄心。
        y_new = (y_anchor["center"] if y_anchor is not None
                 else y["center"])

        # 红端两个独立锚点：V 通道 Hough 找圆周，洋红连通域找镜片中心。
        # 高光/金属环会让两者各偏一边，取平均后圆心更稳，转 90° 也不跳。
        red_hough = self._red_hough_anchor(hsv, y, g, radius)
        mag_anchor = self._magenta_anchor(hsv, red_pred, radius)
        if red_hough is not None and mag_anchor is not None:
            red_new = 0.5 * (red_hough[0] + mag_anchor[0])
        elif red_hough is not None:
            red_new = red_hough[0]
        elif mag_anchor is not None:
            red_new = mag_anchor[0]
        else:
            # 两个锚点都没有：黄灯灭时按黄色半径推间距，沿灯轴外推。
            if y_anchor is None:
                return None
            r_y = float(y_anchor["radius"])
            u = g["center"] - y["center"]
            u = u / max(float(np.linalg.norm(u)), 1e-6)
            red_new = y_new + 2.0 * (r_y / self.LENS_RADIUS_RATIO) * u

        # 红心必须在黄->绿方向的延长线上，横向偏差太大说明锚点错了。
        u = g["center"] - y_new
        u = u / max(float(np.linalg.norm(u)), 1e-6)
        rel = red_new - y_new
        proj = float(np.dot(rel, u))
        perp = float(np.linalg.norm(rel - proj * u))
        if perp > 0.6 * radius:
            return None

        # span_new：新黄红间距的一半；偏离太大说明锚点不可信。
        span_new = 0.5 * float(np.linalg.norm(red_new - y_new))
        smin, smax = self.LENS_ANCHOR_SPAN_SHIFT
        if not (smin * span <= span_new <= smax * span):
            return None

        # green_new：物理上黄、绿、红等间距，绿心取黄红中点。
        green_new = 0.5 * (y_new + red_new)

        # 半径：黄灯灭用黄色候选半径（黄色镜片整片可见，最可信）；
        # 黄灯亮时黄色半径被光晕撑大，改用灭红灯的洋红镜片外接圆。
        if y_anchor is not None:
            new_radius = float(y_anchor["radius"])
        else:
            new_radius = (min(radius, mag_anchor[1])
                          if mag_anchor is not None else radius)

        # 半径必须和间距同一量级（正常约 0.34~0.37 span）。
        if not (0.25 * span_new < new_radius < 0.5 * span_new):
            return None

        return {
            "yellow": {"center": y_new, "radius": new_radius},
            "green": {"center": green_new, "radius": new_radius},
            "red": {"center": red_new, "radius": new_radius},
        }

    # 逐帧定位黄、绿、红三个灯罩，失败时短时间沿用上一帧位置。
    def detect_lenses(self, hsv=None, frame=None):
        # 逐帧定位三个灯罩，返回 {颜色: {center, radius}}。
        #
        # 每帧独立，不依赖上一帧位置，灯箱在画面里移动也能锁住：
        # 1. 黄、绿各用自己的色相掩码找饱和圆形连通域。绿灯亮时绿色光晕
        # 再大也不会和黄灯连成一块；暗场景按全图 V 中位数下放亮度阈值；
        # 2. 黄、绿按颜色配成一对，不限制黄在绿的上方：灯箱可以 0~360°
        # 任意旋转。三灯半径统一由物理间距推：R = 0.36 * span。候选圆
        # 自己的半径只做宽松校验，因为发光晕只会把它撑大；
        # 3. 亮红灯的灯芯也会落进黄色掩码，所以还要看两端外圈有没有洋红
        # 镜片环，把真黄灯和亮红灯分开；这个判据同样与朝向无关；
        # 4. 黄、绿、外推红三个灯心在半径 R 处的外圈都必须是白板，
        # 否则这组黄绿是误配；红灯位置由 绿 + (绿 - 黄) 外推；
        # 5. 绿灯亮时绿色光晕会把绿心沿灯轴推走，再用黄半径/红灯本体
        # Hough 重新定一次圆心和半径；只修“绿灯亮 + 红灯灭”的帧，
        # 其他帧直接返回，保证原结果一个像素都不动。
        #
        # 锁不到返回 {}；短时间锁不到时沿用上一帧位置（最多 LENS_HOLD_FRAMES 帧）。
        if hsv is None:
            if frame is None:
                raise ValueError("hsv 和 frame 至少要给一个")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # v：HSV 的亮度通道，后面的白板外圈校验和暗光自适应都用它。
        v = hsv[:, :, 2]

        # 用全图 V 中位数估计环境亮度：正常光时两个阈值仍是 50/120，
        # 只有变暗时才往下放，保证正常光行为不变。
        # v_ref：环境亮度参考值；v_min：定位用的亮度下限。
        v_ref = float(np.median(v[::self.LENS_DARK_SAMPLE_STEP,
                                   ::self.LENS_DARK_SAMPLE_STEP]))
        v_min = min(self.LENS_V_MIN,
                    max(self.LENS_V_MIN_FLOOR,
                        self.LENS_V_MIN_SCALE * v_ref))
        # surround_v：白板外圈亮度的下限，低于它说明灯没装在白板上。
        surround_v = min(self.LENS_SURROUND_V,
                         max(self.LENS_SURROUND_V_FLOOR,
                             self.LENS_SURROUND_V_SCALE * v_ref))

        # yellows/greens：黄、绿两种颜色的候选圆列表。
        yellows = self._lens_candidates(hsv, self.LENS_HUE_YELLOW, v_min)
        greens = self._lens_candidates(hsv, self.LENS_HUE_GREEN, v_min)

        # H/W：图像高宽，用于外推红灯时判断是否越界。
        H, W = v.shape
        # best：当前最优配对，格式 (score, 黄候选, 绿候选, span)。
        best = None
        for y in yellows:
            for g in greens:
                # span：黄绿圆心距，也是推三灯半径的物理间距。
                span = float(np.linalg.norm(g["center"] - y["center"]))
                if span < self.LENS_MIN_SPAN or span > self.LENS_MAX_SPAN:
                    continue
                # span 已经是黄绿圆心距；不检查 dy/dx 的方向，
                # 灯箱转到任意角度时黄绿连线都可能朝向任何方向。

                # radius：由物理间距推出的统一灯罩半径。
                radius = self.LENS_RADIUS_RATIO * span
                if not (self.LENS_CAND_MIN_RATIO * radius
                        <= y["radius"]
                        <= self.LENS_CAND_MAX_RATIO * radius):
                    continue
                # 绿色是发光灯，候选圆可能被光晕撑得很大，只校验下限
                if g["radius"] < self.LENS_CAND_MIN_RATIO * radius:
                    continue

                # center_r：由“绿 + (绿 - 黄)”外推出的红灯圆心。
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

                # 红端是不是真的红灯：灭红灯看 0.8R 圆盘里的洋红，亮红灯
                # 看 1.10~1.35R 环带里的洋红；两者都低说明红端是误配。
                mag_r_ring = self._magenta_ring(hsv, center_r, radius)
                mag_r_disk = self._magenta_disk(hsv, center_r, radius)
                red_presence = max(mag_r_ring, mag_r_disk)
                red_penalty = max(
                    0.0, self.LENS_RED_PRESENCE - red_presence
                ) * self.LENS_RED_PRESENCE_W

                # 红灯亮时灯芯也会落进黄色掩码，单看颜色分不清哪端是黄、
                # 哪端是红。若外推的红端附近确实有一个黄色候选，就再比较
                # 两端的洋红镜片环：真黄灯没有环，亮红灯有环。这个惩罚项
                # 只影响黄绿配对的选择，不改变三个灯心的几何关系。
                near_red = min(
                    float(np.linalg.norm(c["center"] - center_r))
                    for c in yellows)
                bezel_penalty = 0.0
                if near_red <= self.LENS_RED_MATCH * span:
                    mag_y = self._magenta_ring(hsv, y["center"], radius)
                    bezel_penalty = max(0.0, mag_y - mag_r_ring)

                # 光晕只会让候选半径变大，取较小的那个更接近真实灯罩
                # small_r：黄绿候选半径中较小的一个；score：越小越可信。
                small_r = min(y["radius"], g["radius"])
                score = (abs(small_r / radius - 1.0)
                         + 0.3 * abs(y["radius"] - g["radius"]) / span
                         + bezel_penalty
                         + red_penalty)
                if best is None or score < best[0]:
                    best = (score, y, g, span)

        if best is None:
            return self._lens_fallback()

        # 解包最优配对，并按统一半径重新构造三个灯的字典。
        _, y, g, span = best
        radius = self.LENS_RADIUS_RATIO * span
        center_r = g["center"] + (g["center"] - y["center"])
        # lenses：颜色 -> {center, radius}，后续判亮灭直接用。
        lenses = {
            "yellow": {"center": y["center"], "radius": radius},
            "green": {"center": g["center"], "radius": radius},
            "red": {"center": center_r, "radius": radius},
        }
        # 绿灯亮时用黄半径/红端 Hough 修正光晕造成的圆心偏移；
        # 绿灯灭时 _refine_green_on 返回 None，lenses 保持上面的原值。
        refined = self._refine_green_on(hsv, y, g, span, radius, yellows)
        if refined is not None:
            lenses = refined
        self.lens_miss = 0
        self.last_lenses = lenses
        return lenses

    # 清空灯罩位置保持和亮灭迟滞状态，换场景或重新开始时调用。
    def reset_states(self):
        # 清空灯罩定位保持和亮灭迟滞（换场景/重新锁 L 时调用）。
        self.last_lenses = None
        self.lens_miss = 0
        # color：逐盏灯把稳定状态和投票计数恢复为初值。
        for color in self.LAMP_ORDER:
            self.s_states[color] = "unknown"
            self.s_votes[color] = 0

    # 本帧没锁到灯时沿用上一帧位置，超过保持帧数就清空。
    def _lens_fallback(self):
        # 本帧没锁到：短时间沿用上一帧位置，超时则清空。
        self.lens_miss += 1
        if (self.last_lenses is not None
                and self.lens_miss <= self.LENS_HOLD_FRAMES):
            return self.last_lenses
        self.last_lenses = None
        return {}

    # 计算圆心外 1.15~1.6R 环带的中位亮度，用来判断灯是否装在白板上。
    @staticmethod
    def _annulus_median(plane, center, radius, step=2):
        # 圆心外 1.15~1.6R 环内的中位亮度，用来判断灯罩是否装在白板上。
        #
        # step>1 时在环上隔点取样，精度足够且快很多。
        # H/W：图像高宽；cx/cy：圆心整数坐标。
        H, W = plane.shape
        cx, cy = int(round(center[0])), int(round(center[1]))
        # r0/r1：环带的内外半径，分别取 1.15R 和 1.6R。
        r0 = max(1, int(round(radius * 1.15)))
        r1 = max(r0 + 1, int(round(radius * 1.6)))
        # x0/y0/x1/y1：环带的外接矩形，裁到图像范围内。
        x0, y0 = max(0, cx - r1), max(0, cy - r1)
        x1, y1 = min(W, cx + r1 + 1), min(H, cy + r1 + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        # yy/xx：矩形内按 step 隔点采样的坐标网格。
        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        # dist：每个采样点到圆心的距离；ring：落在环带内的布尔掩码。
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = (dist >= r0) & (dist <= r1)
        if not np.any(ring):
            return 0.0
        return float(np.median(plane[y0:y1:step, x0:x1:step][ring]))

    # 在三个灯罩上判亮灭：黄/绿用 S 比值，红灯用色相 H，并做迟滞和连续帧确认。
    def judge_lens_state(self, hsv, lenses=None):
        # 在三个灯罩上判亮灭。
        #
        # 黄/绿：中心 0.4R 的 S 中位数 / 外圈 0.85~1.15R 的 S 60 分位。
        # 灯亮时中心冲白、发光环仍饱和，比值塌下来；灯灭时整盏同色，
        # 比值≈1。强光把灯罩整体照亮时这个比值仍然分得开。
        # 红：S 不可靠（灭≈100，亮时白心可低到 43），改用色相——
        # 亮灯发橙 H≈26，灭灯品红 H≈172，折成 d 后分成正负两侧。
        # 只统计红色相像素，避免旁边绿灯亮时把 H 拉到绿色区。
        # 状态翻转有迟滞 + 连续帧确认，短暂抖动不会来回跳。
        if lenses is None:
            lenses = self.last_lenses or {}

        # h/s：HSV 的色相、饱和度通道；H/W：图像高宽。
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        H, W = s.shape
        # states：每盏灯的最终状态；detail：判据值、阈值、像素数等调试信息。
        states = {}
        detail = {}

        # color：当前处理的灯（yellow/green/red）；lens：该灯的圆心和半径。
        for color in self.LAMP_ORDER:
            lens = lenses.get(color)
            if lens:
                # cx/cy：灯心浮点坐标；radius：统一灯罩半径。
                cx, cy = lens["center"]
                radius = float(lens["radius"])
                # r：红灯取样盘半径；icx/icy：灯心整数坐标。
                r = max(1, int(round(radius * self.LENS_SAMPLE_RATIO)))
                icx, icy = int(round(cx)), int(round(cy))
                # x0/y0/x1/y1：取样区域，裁到图像范围内。
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
                # disk：取样圆盘掩码；hit：圆盘内的像素布尔掩码。
                disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.circle(disk, (icx - x0, icy - y0), r, 255, -1)
                hit = disk > 0
                # sc/hc：取样区域内的 S 和 H 通道。
                sc = s[y0:y1, x0:x1]
                hc = h[y0:y1, x0:x1]
                # value_s：圆盘内 S 的中位数，只用于打印诊断。
                value_s = float(np.median(sc[hit]))
                # 只用 S 够高、且色相在红色区的像素算 H：
                # 过曝白心 H 无意义，旁边绿灯的绿光也不能算进来。
                valid = hit & (sc >= self.RED_H_S_FLOOR)
                valid &= ((hc <= self.RED_H_KEEP[0])
                          | (hc >= self.RED_H_KEEP[1]))
                # frac：有效像素占圆盘的比例；太低说明整盏灯过曝。
                frac = float(np.count_nonzero(valid)) / max(
                    int(np.count_nonzero(hit)), 1)
                if frac >= self.RED_H_MIN_FRAC:
                    # med：有效像素的 H 中位数；d_val：折回后的色相判据。
                    med = float(np.median(hc[valid]))
                    d_val = med if med <= 90 else med - 180.0
                    # state：经过迟滞和连续帧确认后的状态。
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
                # rr：外圈取样半径；rx0/ry0/rx1/ry1：外圈取样区域。
                rr = max(1, int(round(radius * self.RATIO_RIM[1])))
                rx0, ry0 = max(0, icx - rr), max(0, icy - rr)
                rx1, ry1 = min(W, icx + rr + 1), min(H, icy + rr + 1)
                # yy/xx：取样区域的坐标网格；dist：到灯心的距离。
                yy, xx = np.ogrid[ry0:ry1, rx0:rx1]
                dist = np.sqrt((xx - icx) ** 2 + (yy - icy) ** 2)
                # sc：取样区域的 S 通道。
                sc = s[ry0:ry1, rx0:rx1]
                # center_mask：中心 0.4R 圆盘；rim_mask：0.85~1.15R 外圈。
                center_mask = dist <= radius * self.RATIO_CENTER
                rim_mask = ((dist >= radius * self.RATIO_RIM[0])
                            & (dist <= radius * self.RATIO_RIM[1]))
                # center_s：中心 S 中位数；rim_s：外圈 S 的 60 分位。
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
                    # d_val：中心/外圈 S 比值，灯亮时明显小于 1。
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

    # 按当前判据值、迟滞阈值和连续帧确认，更新单盏灯的状态。
    def _lens_state_vote(self, color, value, is_hue=False):
        # 按判据值 + 迟滞 + 连续帧确认更新单盏灯的亮灭。
        # cur：当前稳定状态；cand：本帧判据给出的候选状态。
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


# 输出类：负责图上标注、控制台状态打印和图片/视频保存。
class Output:
    # 输出类：图上标注 + 控制台数据 + 结果保存。
    #
    # source:
    # "camera"  按空格把当前标注帧存为 save_dir/annotated_XX.png
    # "image"   直接写 save_dir/<stem>_annotated.png
    # "video"   每帧写入 save_dir/<stem>_annotated.mp4，结束时调 close()

    FONT = cv2.FONT_HERSHEY_SIMPLEX   # 统一字体
    FONT_SCALE = 0.7                  # 标签字号
    FONT_THICK = 2                    # 标签笔画粗细
    LINE_THICK = 2                    # 十字架/圆圈的线宽

    ORDER = LightDetector.LAMP_ORDER   # 标注/打印顺序：黄、绿、红（与灯箱朝向无关）
    # 每盏灯在图上的颜色（BGR）。
    LAMP_COLORS = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
    }
    ON_COLOR = (0, 255, 0)            # 亮：绿色标签
    OFF_COLOR = (180, 180, 180)       # 灭：灰色标签
    UNKNOWN_COLOR = (0, 165, 255)     # 未知：橙色标签
    TEXT_BG = (0, 0, 0)               # 文字/线条的黑边颜色
    FPS_PRINT_INTERVAL = 1.0  # 状态不变时，至少每秒打一行帧率

    # 初始化输出目录、视频写入器和帧率统计状态。
    def __init__(self, source="camera", input_path=None,
                 save_dir=RESULTS_DIR, fps=CAPTURE_FPS):
        if source not in ("camera", "image", "video"):
            raise ValueError('source 必须是 "camera"/"image"/"video"')
        # source：数据源类型；input_path：图片/视频路径；save_dir：输出目录。
        self.source = source
        self.input_path = input_path
        self.save_dir = save_dir
        # video_fps：写视频时使用的帧率。
        self.video_fps = fps

        # writer：VideoWriter；video_path：正在写的视频路径。
        self.writer = None
        self.video_path = None
        # saved_image：图片模式是否已经保存过，避免重复写。
        self.saved_image = False
        # last_states：上一次打印时的三灯状态，用来判断状态是否变化。
        self.last_states = None

        # out_fps：平滑后的输出帧率；_last_time/_last_print：时间戳。
        self.out_fps = None
        self._last_time = time.monotonic()
        self._last_print = 0.0

        os.makedirs(self.save_dir, exist_ok=True)

    # 板块 1：图上标注
    # 在图上画每个灯的十字架、范围圆和状态标签，返回标注后的新图。
    def draw(self, frame, states, detail):
        # 在图上画灯的位置/大小/范围/亮灭，返回标注后的新图。
        #
        # 十字架交点=圆心，臂长=半径，所以位置和大小一个标记就够；
        # 外面再画一个圆把灯的范围圈出来。
        # img：在副本上画，避免修改原始帧。
        img = frame.copy()

        # color：当前灯的颜色；d：该灯的判据和几何信息。
        for color in self.ORDER:
            d = None if detail is None else detail.get(color)
            if not d:
                continue

            # state：该灯当前状态；cx/cy：圆心整数坐标；r：半径。
            state = "unknown" if states is None else states.get(color, "unknown")
            cx, cy = (int(round(v)) for v in d["center"])
            r = max(1, int(round(d["radius"])))
            # lamp_bgr：该灯对应的绘制颜色。
            lamp_bgr = self.LAMP_COLORS[color]

            # 十字架：臂长 = 半径
            self._line(img, (cx - r, cy), (cx + r, cy), lamp_bgr)
            self._line(img, (cx, cy - r), (cx, cy + r), lamp_bgr)
            # 圈出灯的实际范围
            self._circle(img, (cx, cy), r, lamp_bgr)

            # state_text：图上显示的状态文本；label：完整标签。
            state_text = "?" if state == "unknown" else state.upper()
            label = f"{color.upper()} {state_text} r={r}"
            self._label(img, label, (cx - r, cy - r - 8),
                        self._state_color(state))

        return img

    # 板块 2：控制台数据
    # 打印帧率和每盏灯的圆心、半径、判据值、阈值；状态不变时按秒限流。
    def report(self, states, detail, force=False):
        # 输出帧率、每个灯的状态、圆心、半径、判据值和阈值。
        #
        # 每帧都会更新输出帧率；默认在状态变化或距上次打印超过
        # FPS_PRINT_INTERVAL 秒时打印，force=True 每帧都打印。
        # now：当前时间；dt：距上一帧的时间差；inst：瞬时帧率。
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            inst = 1.0 / dt
            self.out_fps = (inst if self.out_fps is None
                            else 0.9 * self.out_fps + 0.1 * inst)

        if states is None:
            states = {}
        # key：三灯状态组成的元组；changed：和上次打印相比是否变化。
        key = tuple(states.get(c, "unknown") for c in self.ORDER)
        changed = key != self.last_states
        self.last_states = key
        if (not force and not changed
                and now - self._last_print < self.FPS_PRINT_INTERVAL):
            return
        self._last_print = now

        # modes：本帧用到的判据类型（ratio/H）；parts：每盏灯的输出片段。
        modes = []
        parts = []
        for color in self.ORDER:
            d = None if detail is None else detail.get(color)
            state = states.get(color, "unknown")
            if not d:
                parts.append(f"{color}:{state}")
                continue
            # mode：该灯用的判据；判据值/阈值从 detail 里取。
            mode = d.get("mode")
            if mode and mode not in modes:
                modes.append(mode)
            cx, cy = d["center"]
            if mode == "H":
                # hv：红灯色相判据值，None 时显示 --。
                hv = "--" if d["value"] is None else f"{d['value']:+.0f}"
                parts.append(
                    f"{color}:{state} c=({cx:.0f},{cy:.0f}) "
                    f"r={d['radius']:.1f} "
                    f"H={hv}/{d['threshold']:.0f} "
                    f"S={d['S']:.0f} frac={d['frac']:.2f} "
                    f"px={d['pixels']}"
                )
            elif mode == "ratio":
                # rv：黄/绿中心外圈比值，None 时显示 --。
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

        # fps_text：帧率文本；mode_text：本帧判据类型文本；line：完整输出行。
        fps_text = "--" if self.out_fps is None else f"{self.out_fps:.1f}"
        mode_text = "+".join(modes) if modes else "-"
        line = f"[{mode_text}] fps={fps_text} | " + " | ".join(parts)
        print(line, flush=True)

    # 板块 3：保存
    # 按数据源选择保存方式：摄像头编号图、图片标注图或视频文件。
    def save(self, frame):
        # 保存当前标注帧。摄像头由空格触发，图片/视频自动触发。
        if self.source == "camera":
            return self._save_numbered(frame)
        if self.source == "image":
            return self._save_image(frame)
        return self._save_video(frame)

    # 按 p 拍照：把原始帧存到 photos/photo_XX.png，供离线标定。
    def snapshot(self, frame):
        # 按 p 拍照：把原始帧存成 photos/photo_XX.png，用于离线标定。
        os.makedirs(PHOTO_DIR, exist_ok=True)
        # index：下一个照片编号；img_path：照片完整路径。
        index = next_index(PHOTO_DIR, prefix="photo_")
        img_path = os.path.join(PHOTO_DIR, f"photo_{index:02d}.png")
        if not cv2.imwrite(img_path, frame):
            print(f"拍照保存失败: {img_path}")
            return None
        print(f"已拍照: {img_path}")
        return img_path

    # 处理摄像头/视频按键：空格保存当前帧，q/Esc 返回 True 表示退出。
    def handle_key(self, key, frame):
        # 摄像头模式按键：空格保存当前帧。返回 True 表示 q/Esc 退出。
        if self.source == "camera" and key == ord(" "):
            self.save(frame)
        return key in (ord("q"), 27)

    # 视频模式收尾：释放 VideoWriter；其他模式无操作。
    def close(self):
        # 视频模式收尾；摄像头/图片模式无操作。
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"已保存视频: {self.video_path}")

    # 摄像头模式：按已有编号递增保存 annotated_XX.png。
    def _save_numbered(self, frame):
        # index：下一个可用编号；path：输出路径。
        index = next_index(self.save_dir)
        path = os.path.join(self.save_dir, f"annotated_{index:02d}.png")
        if not cv2.imwrite(path, frame):
            print(f"保存失败: {path}")
            return None
        print(f"已保存: {path}")
        return path

    # 图片模式：保存为 <stem>_annotated.png，每张图只保存一次。
    def _save_image(self, frame):
        if self.saved_image:
            return None
        # stem：输入图片的文件名（不含扩展名）；path：输出路径。
        stem = os.path.splitext(
            os.path.basename(self.input_path or "image"))[0]
        path = os.path.join(self.save_dir, f"{stem}_annotated.png")
        if not cv2.imwrite(path, frame):
            print(f"保存失败: {path}")
            return None
        self.saved_image = True
        print(f"已保存: {path}")
        return path

    # 视频模式：第一帧创建 mp4 writer，之后逐帧写入。
    def _save_video(self, frame):
        if self.writer is None:
            # stem：输入视频名；path：输出视频路径；h/w：帧高宽。
            stem = os.path.splitext(
                os.path.basename(self.input_path or "video"))[0]
            path = os.path.join(self.save_dir, f"{stem}_annotated.mp4")
            h, w = frame.shape[:2]
            # fourcc：mp4v 编码器，兼容性最好。
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

    # 画线：先画粗黑边再画彩色线，白底上也看得清。
    @classmethod
    def _line(cls, img, p1, p2, color):
        # 先黑后彩，白面板上也能看清
        cv2.line(img, p1, p2, cls.TEXT_BG, cls.LINE_THICK + 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, cls.LINE_THICK, cv2.LINE_AA)

    # 画圆：先画粗黑边再画彩色圆。
    @classmethod
    def _circle(cls, img, center, r, color):
        cv2.circle(img, center, r, cls.TEXT_BG, cls.LINE_THICK + 2, cv2.LINE_AA)
        cv2.circle(img, center, r, color, cls.LINE_THICK, cv2.LINE_AA)

    # 画文字标签：先画黑底，再把文字限制在画面内。
    @classmethod
    def _label(cls, img, text, org, color):
        # tw/th：文字宽高；base：基线以下的高度。
        (tw, th), base = cv2.getTextSize(
            text, cls.FONT, cls.FONT_SCALE, cls.FONT_THICK)
        # x/y：把标签限制在画面内后的左上角坐标。
        x = min(max(0, org[0]), max(0, img.shape[1] - tw - 6))
        y = min(max(th + 4, org[1]), img.shape[0] - base - 2)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base),
                      cls.TEXT_BG, -1)
        cv2.putText(img, text, (x + 3, y - 2), cls.FONT, cls.FONT_SCALE,
                    color, cls.FONT_THICK, cv2.LINE_AA)

    # 按 on/off/unknown 返回标签颜色。
    @classmethod
    def _state_color(cls, state):
        if state == "on":
            return cls.ON_COLOR
        if state == "off":
            return cls.OFF_COLOR
        return cls.UNKNOWN_COLOR


# 命令行读取函数
# 解析命令行参数：--image / --video / --camera 三选一，--show 控制是否显示窗口。
def parse_args():
    # parser：命令行解析器；source：三选一的输入源参数组。
    parser = argparse.ArgumentParser(description="红绿灯检测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="输入图片路径")
    source.add_argument("--video", help="输入视频路径")
    source.add_argument("--camera", type=int, help="摄像头索引，如 0")
    parser.add_argument("--show", action="store_true", help="显示结果窗口")
    return parser.parse_args()


# 扫描目录里已有的同前缀 PNG，返回下一个可用编号，避免覆盖旧结果。
def next_index(save_dir, prefix="annotated_"):
    # 从已有文件名接着编号，避免重跑时覆盖之前的图片。
    # used：已经用过的编号列表；name/stem/ext：目录里的文件名、主干、扩展名。
    used = []
    for name in os.listdir(save_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".png" and stem.startswith(prefix):
            try:
                used.append(int(stem[len(prefix):]))
            except ValueError:
                pass
    return max(used, default=0) + 1


# 主入口：创建检测器和输出对象，按摄像头/图片/视频三种模式运行。
def main():
    # args：命令行参数；detector：灯箱检测器；pre：预处理器。
    args = parse_args()

    detector = LightDetector()
    pre = Preprocessor()

    # 单帧完整流水线：预处理 -> 灯罩几何定位 -> 判亮灭 -> 标注 -> 打印。
    def process(frame, out):
        # frame：滤波后的 BGR 帧；hsv：对应的 HSV 图。
        frame = pre.apply_gaussian(frame)

        # 每帧只转一次 BGR->HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lenses：三灯位置；states：亮灭状态；detail：判据值和阈值。
        lenses = detector.detect_lenses(hsv=hsv)
        states, detail = detector.judge_lens_state(hsv, lenses)

        # annotated：画好标注的结果图。
        annotated = out.draw(frame, states, detail)
        out.report(states, detail)
        return annotated

    # ---------- 摄像头 ----------
    if args.camera is not None:
        # cap：摄像头读取器；优先 V4L2 后端。
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
        if not pre.wait_for_start(cap):
            cap.release()
            cv2.destroyAllWindows()
            return
        cv2.destroyAllWindows()
        detector.reset_states()

        # out：摄像头输出对象，按空格保存编号图。
        out = Output(source="camera")
        try:
            while True:
                # ok：是否读到帧；frame：当前原始画面。
                ok, frame = cap.read()
                if not ok:
                    print("读取画面失败")
                    break

                # annotated：当前帧的标注结果；key：当前按键。
                annotated = process(frame, out)
                cv2.imshow("detect", pre.preview(annotated))
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
        # frame：读入的 BGR 图片；读失败直接返回。
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"无法读取图片: {args.image}")
            return
        print(f"图片: {args.image} ({frame.shape[1]}x{frame.shape[0]})")

        out = Output(source="image", input_path=args.image)
        annotated = process(frame, out)
        out.save(annotated)

        if args.show:
            cv2.imshow("detect", pre.preview(annotated))
            print("按任意键退出")
            cv2.waitKey(0)
        out.close()
        cv2.destroyAllWindows()
        return

    # ---------- 视频 ----------
    # cap：视频读取器；fps：源视频帧率，读不到时用采集默认值。
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法读取视频: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = CAPTURE_FPS

    # ok：是否读到首帧；frame：首帧图像。
    ok, frame = cap.read()
    if not ok:
        print(f"视频没有可读帧: {args.video}")
        cap.release()
        return
    print(f"视频: {args.video} ({frame.shape[1]}x{frame.shape[0]}"
          f"@{fps:.1f}fps)")

    # out：视频输出对象，逐帧写入 mp4。
    out = Output(source="video", input_path=args.video, fps=fps)
    try:
        while ok:
            # annotated：当前帧标注结果，写入视频。
            annotated = process(frame, out)
            out.save(annotated)

            if args.show:
                cv2.imshow("detect", pre.preview(annotated))
                if out.handle_key(cv2.waitKey(1) & 0xFF, annotated):
                    break
            ok, frame = cap.read()
    finally:
        cap.release()
        out.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
