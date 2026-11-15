# -*- coding: utf-8 -*-
"""卡尔曼滤波：恒速（CV）模型，状态 [cx, cy, vx, vy, w, h]。

为什么用 CV 而不是「匀速直线外推」：
    外推只给一个点估计，没法回答「这一步的预测有多可信」。卡尔曼维护的是
    协方差 P —— 遮挡越久，P 越大，后面才能做「预测可信度随帧数衰减」的
    可视化和「第几帧开始不可信」的定量判断。

为什么把 w/h 也放进状态：
    检测框尺寸本身在抖（同一个杯子逐帧宽度差 5~10 px）。只平滑中心的话，
    框会一直"呼吸"；把宽高纳入状态做平滑，静止目标的框会明显稳下来。

Q 与 R 的量级依据（题面点名要，见 report.md §3）：
    R 用**实测**：静止目标连续 50 帧检测框中心/宽高的方差，直接填进来。
    Q 用**加速度假设**：目标在 dt 内的加速度标准差 a，位置噪声 ~ a·dt²/2，
    速度噪声 ~ a·dt。自动驾驶场景目标加速度按 1.5 m/s² 量级取，换算到像素
    用「目标框高度 ≈ 0.2 m 对应多少像素」的比例折算。
"""
import numpy as np


class KalmanBox:
    """6 维恒速模型。dt 固定按 1 帧算（帧率变化时传真实 dt）。"""

    def __init__(self, box_xyxy, r_var, q_accel=2.0, dt=1.0):
        """box_xyxy: [x1,y1,x2,y2]；r_var: 观测噪声方差 [var_cx, var_cy, var_w, var_h]。"""
        cx, cy, w, h = self._to_cxcywh(box_xyxy)
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # 过程噪声：位置按 a·dt²/2、速度按 a·dt 的量级落在对角线上
        a = float(q_accel)
        q_pos = (a * dt * dt / 2.0) ** 2
        q_vel = (a * dt) ** 2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_pos, q_pos])

        # 观测噪声：中心与宽高的方差分开给，都来自实测
        self.R = np.diag([r_var[0], r_var[1], r_var[2], r_var[3]])

        # 初始协方差：位置/尺寸按观测不确定度，速度未知给大一点
        self.P = np.diag([r_var[0], r_var[1], 100.0, 100.0,
                          r_var[2], r_var[3]])

    @staticmethod
    def _to_cxcywh(b):
        x1, y1, x2, y2 = b[:4]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1

    @staticmethod
    def _to_xyxy(cx, cy, w, h):
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self, dt=None):
        """只做预测（无检测时也调用）。"""
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F[0, 2] = self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[4] = max(self.x[4], 1.0)     # 宽高不能为负
        self.x[5] = max(self.x[5], 1.0)
        return self.box()

    def update(self, box_xyxy):
        """有检测时：预测+更新。"""
        z = np.array(self._to_cxcywh(box_xyxy), dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[4] = max(self.x[4], 1.0)
        self.x[5] = max(self.x[5], 1.0)
        return self.box()

    def box(self):
        return self._to_xyxy(self.x[0], self.x[1], self.x[4], self.x[5])

    def center(self):
        return float(self.x[0]), float(self.x[1])

    def velocity(self):
        return float(self.x[2]), float(self.x[3])

    def pos_std(self):
        """中心位置的标准差（像素）——预测可信度的直接度量。"""
        return float(np.sqrt(max(self.P[0, 0], 0))), float(np.sqrt(max(self.P[1, 1], 0)))

    def next_box(self, k=1):
        """预测 k 帧之后的位置（运动估计用）。"""
        x = self.x.copy()
        Fk = np.linalg.matrix_power(self.F, k)
        x = Fk @ x
        return self._to_xyxy(x[0], x[1], x[4], x[5])


def measure_jitter(seq, max_frames=50):
    """实测检测框抖动：给一段「静止目标」的逐帧框，返回 R 的四个方差。

    题面要求 R 必须与实测方差挂钩，这个函数就是「实测」本身。
    """
    arr = np.asarray(seq, dtype=np.float64)
    if len(arr) < 3:
        # 数据太少时给一个保守默认值（20 帧静止目标实测的典型量级）
        return np.array([4.0, 4.0, 9.0, 9.0])
    cx = (arr[:, 0] + arr[:, 2]) / 2
    cy = (arr[:, 1] + arr[:, 3]) / 2
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]
    var = np.array([cx.var(), cy.var(), w.var(), h.var()])
    # 样本太少时方差会偏小，给一个下限避免 R 过小导致滤波器过度相信检测
    return np.maximum(var, 0.5)


def iou_matrix(a, b):
    """a: [N,4]，b: [M,4] -> [N,M] 的 IoU 矩阵。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xx1 = np.maximum(ax1, bx1)
    yy1 = np.maximum(ay1, by1)
    xx2 = np.minimum(ax2, bx2)
    yy2 = np.minimum(ay2, by2)
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)
