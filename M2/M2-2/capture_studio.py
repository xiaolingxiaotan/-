# -*- coding: utf-8 -*-
"""M2-2 现场精度实测拍摄台：选择窗口 + 空格拍摄 + 一键汇总填表。

用途
----
按题目要求的三组实验逐组拍照：距离线性度、姿态响应、畸变开/关。
拍完把实测数字填进 report.md §7 和 results/accuracy.md 的空表。

设计约束（重要）
--------------
1. **不修改 pose_estimate.py 的任何一行**。本文件只 import 它的现成函数，
   采集时用的算法与现场演示用的算法是同一套 process_frame，不存在
   "采集一套、演示另一套"的风险，也不动现有正常光亮的准确度。
2. **不产生任何测量值**。所有数字都来自 process_frame 的真实输出；
   没拍够的组在表里留空并打印警告，绝不补数、绝不外推。
3. **照片只追加不覆盖**。重拍是把旧目录整体改名存档，不删现场原图。
4. 填表前自动备份原文件；表格已有数据时默认拒绝覆盖（--force 才覆盖）。
5. **畸变 / 非畸变是两套独立流程**，选择窗口可切换预览看哪一套；无论选哪个，
   同一帧的两套结果都会存进 measure.csv，填表用的 |Δt| 不会缺数据。
6. 远距离只在快路径「未检出棋盘格」时自动重试一次全分辨率兜底；平时与现场
   演示走完全相同的检测顺序，**不放宽任何阈值**。
7. **除距离线性度四档外，全部组固定在 0.50 m 基准线**：零位、姿态、畸变都不
   挪相机距离，只换靶标角度或画面位置，让两套 dist 的差与角度误差不掺进距离变化。

运行
----
    python3 capture_studio.py                  打开选择窗口（默认）
    python3 capture_studio.py --finalize       只汇总并填表，不开窗口
    python3 capture_studio.py --summary        只打印各组进度/统计，不写文件
    python3 capture_studio.py --dry-run        配合 --finalize：只打印将写入的内容
    python3 capture_studio.py --selftest       自检表格定位与渲染，不写任何文件
    python3 capture_studio.py --camera 1       指定摄像头索引
    python3 capture_studio.py --flow off       窗口默认选中的流程：both / on / off
    python3 capture_studio.py --force          表格已有数据时仍覆盖

产物
----
    results/measure/<组名>/raw_*.png    未标注原始帧（PNG 无损，可离线重算）
    results/measure/<组名>/axis_*.png   带角点/坐标轴/数值的标注图
    results/measure/<组名>/measure.csv  每帧真值 + 畸变/非畸变两套实测值 + 所选流程
"""

import argparse
import csv
import glob
import math
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import pose_estimate as pe

# ---------------------------------------------------------------------------
# 常量区
# ---------------------------------------------------------------------------
# HERE：本脚本所在目录，保证从任何工作目录启动都能找到标定和报告。
HERE = os.path.dirname(os.path.abspath(__file__))
# MEASURE_DIR：本次正式实测的根目录。和旧的 results/capture/ 物理隔离，
# 避免把早期试拍（distance_030 那 10 张）混进正式统计。
MEASURE_DIR = os.path.join(HERE, "results", "measure")
# REPORT_PATH / ACCURACY_PATH：两张要填的空表所在文件。
REPORT_PATH = os.path.join(HERE, "report.md")
ACCURACY_PATH = os.path.join(HERE, "results", "accuracy.md")

# FRAMES_PER_GROUP：每组要拍的有效帧数，题目要求"各 5 帧"。
FRAMES_PER_GROUP = 5
# CAPTURE_FPS：摄像头目标帧率，与 pose_estimate.py 保持一致。
CAPTURE_FPS = pe.CAPTURE_FPS
# WARMUP_FRAMES：开摄像头后先丢掉的帧数，等自动曝光/画面稳定再允许拍。
WARMUP_FRAMES = 15
# WINDOW_NAME：OpenCV 预览窗口名。
WINDOW_NAME = "M2-2 Measure"
# FONT：OpenCV 字体（不含中文，所以预览窗口里的提示一律用英文）。
FONT = pe.FONT
# MAX_REPROJ_PX：与主程序同一阈值，超过它的帧不算有效帧。
MAX_REPROJ_PX = pe.MAX_REPROJ_PX

# DISTANCES：距离线性度的四档真值，单位米。
DISTANCES: Tuple[float, ...] = (0.30, 0.50, 0.80, 1.20)
# AXES：姿态响应的三个轴，以及中文说明（只给选择窗口看）。
AXES: Tuple[Tuple[str, str], ...] = (
    ("yaw", "偏航 · 绕竖直轴左右转"),
    ("pitch", "俯仰 · 绕水平轴上下翻"),
    ("roll", "横滚 · 绕光轴平面内转"),
)
# ANGLES：姿态响应的三档真值，单位度。
ANGLES: Tuple[int, ...] = (30, 60, 90)
# DISTORT_CASES：畸变对照表里的两行，与 report.md §7.3 / accuracy.md §3.2 一一对应。
# 每项 = (距离 m, 画面位置)。全部固定在 0.50 m 基准线，只留「中心 / 边缘」一个变量：
# 同一帧两种 dist 的差不会被距离变化污染，得到的就是纯畸变影响。距离变化单独由
# DISTANCES 那四档（距离线性度）负责。
DISTORT_CASES: Tuple[Tuple[float, str], ...] = (
    (0.50, "中心"),
    (0.50, "边缘"),
)

# FLOW_CHOICES：选择窗口里的三档流程。两套结果永远都存，flow 只决定预览看哪一套。
FLOW_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("both", "同帧两套对照（默认，填表用）"),
    ("on", "畸变流程（带 k1/k2/k3）"),
    ("off", "非畸变流程（零畸变）"),
)
FLOW_LABELS: Dict[str, str] = dict(FLOW_CHOICES)

# CSV_HEADER：每组 measure.csv 的表头。所有组共用一套列，flow 记录窗口里选的流程。
CSV_HEADER: Tuple[str, ...] = (
    "index", "timestamp", "wall_time", "group", "kind", "truth", "unit",
    "axis", "position", "flow", "detected", "reason",
    "x_m", "y_m", "z_m", "distance_m", "roll_deg", "pitch_deg", "yaw_deg",
    "reproj_px", "detector", "raw", "annotated",
    "nodist_x_m", "nodist_y_m", "nodist_z_m", "nodist_distance_m",
    "nodist_roll_deg", "nodist_pitch_deg", "nodist_yaw_deg", "nodist_reproj_px",
    "dt_mm", "droll_deg", "dpitch_deg", "dyaw_deg",
)


# ---------------------------------------------------------------------------
# 组定义
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupSpec:
    """GroupSpec：一个实验组的定义（选择窗口里的一行）。所有字段都是"这组是什么"，
    不含任何测量结果；填表程序靠它知道每组对应表格的哪一行。"""

    key: str          # key：组名，同时也是 results/measure/ 下的目录名
    kind: str         # kind：zero / distance / pose / distort
    category: str     # category：选择窗口里的大类标题
    label: str        # label：中文显示名
    truth: float      # truth：本组真值；距离 m、角度 °；zero/distort 用 0 占位
    unit: str         # unit：真值单位（m / °）
    axis: str = ""    # axis：姿态组对应的轴（yaw/pitch/roll）
    position: str = ""  # position：畸变组的画面位置（中心/边缘）
    frames: int = FRAMES_PER_GROUP  # frames：本组要拍的有效帧数


def build_groups() -> List[GroupSpec]:
    """build_groups：按题目三张表的行顺序生成全部实验组。

    顺序刻意排成"先零位基线 → 距离 → 姿态 → 畸变"：现场先拍正对，
    后面算姿态误差才有零位可扣，不会出现把 89° 零位当成 0° 的错。
    """
    groups: List[GroupSpec] = [
        GroupSpec(
            key="pose_zero", kind="zero", category="0 · 先拍零位基线",
            label="正对零位（三个角的起跑线，务必先拍）", truth=0.0, unit="°",
        ),
    ]
    # 距离线性度：4 档 × 5 帧。
    for distance in DISTANCES:
        groups.append(GroupSpec(
            key=f"distance_{int(round(distance * 100)):03d}",
            kind="distance", category="1 · 距离线性度",
            label=f"距离 {distance:.2f} m", truth=distance, unit="m",
        ))
    # 姿态响应：3 轴 × 3 角度 × 5 帧。
    for axis, axis_cn in AXES:
        for angle in ANGLES:
            groups.append(GroupSpec(
                key=f"{axis}_{angle:03d}", kind="pose",
                category=f"2 · 姿态响应（{axis_cn}）",
                label=(
                    f"{axis} {angle}°（90° 掠射，检不出可跳过）"
                    if angle == 90 and axis != "roll" else f"{axis} {angle}°"
                ),
                truth=float(angle), unit="°",
                axis=axis,
            ))
    # 畸变开/关：2 组 × 5 帧（0.50 m 中心 / 边缘），每帧同时解两套 PnP。
    for distance, position in DISTORT_CASES:
        tag = "center" if position == "中心" else "edge"
        groups.append(GroupSpec(
            key=f"distort_{int(round(distance * 100)):03d}_{tag}",
            kind="distort", category="3 · 畸变开 / 关",
            label=f"{distance:.2f} m · {position}", truth=distance, unit="m",
            position=position,
        ))
    return groups


# GROUPS：全部实验组（单一数据源，选择窗口和填表程序共用）。
GROUPS: List[GroupSpec] = build_groups()
# GROUP_BY_KEY：按组名快速查定义。
GROUP_BY_KEY: Dict[str, GroupSpec] = {g.key: g for g in GROUPS}


# ---------------------------------------------------------------------------
# 存储：每组一个目录，一个 measure.csv
# ---------------------------------------------------------------------------
class GroupStore:
    """GroupStore：负责拍照落盘与进度统计。

    不缓存任何数据在内存里，每次都从 measure.csv 重新读，这样中途关掉程序、
    换一台电脑接着拍、或者手工删掉几张坏帧，进度都不会算错。
    """

    def __init__(self, root: str = MEASURE_DIR) -> None:
        # root：实测根目录。
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def dir_for(self, spec: GroupSpec) -> str:
        """dir_for：该组的目录路径（调用方负责创建）。"""
        return os.path.join(self.root, spec.key)

    def csv_path(self, spec: GroupSpec) -> str:
        """csv_path：该组清单文件路径。"""
        return os.path.join(self.dir_for(spec), "measure.csv")

    def rows(self, spec: GroupSpec) -> List[Dict[str, str]]:
        """rows：读该组全部已拍帧；没有清单就返回空列表。"""
        path = self.csv_path(spec)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def valid_rows(self, spec: GroupSpec) -> List[Dict[str, str]]:
        """valid_rows：只保留"检出成功且重投影误差没超标"的帧。

        题目明确要求"只统计 detected=1 的帧"；重投影误差超标的帧是坏数据，
        按 accuracy.md §0 的约定也不能直接算进精度。
        """
        valid: List[Dict[str, str]] = []
        for row in self.rows(spec):
            if str(row.get("detected", "")) != "1":
                continue
            try:
                if float(row.get("reproj_px", "")) > MAX_REPROJ_PX:
                    continue
            except (TypeError, ValueError):
                continue
            valid.append(row)
        return valid

    def progress(self, spec: GroupSpec) -> Tuple[int, int]:
        """progress：返回 (有效帧数, 总帧数)。有效帧数就是进度。"""
        all_rows = self.rows(spec)
        return len(self.valid_rows(spec)), len(all_rows)

    def next_index(self, spec: GroupSpec) -> int:
        """next_index：下一个文件序号。扫 raw_*.png 而不是数 CSV 行，
        这样手工补图、清单缺行也绝不会覆盖已有照片。"""
        pattern = os.path.join(self.dir_for(spec), "raw_*.png")
        biggest = 0
        for name in glob.glob(pattern):
            match = re.search(r"raw_(\d+)\.png$", name)
            if match:
                biggest = max(biggest, int(match.group(1)))
        return biggest + 1

    def append(
        self,
        spec: GroupSpec,
        row: Dict[str, str],
        frame: np.ndarray,
        vis: np.ndarray,
    ) -> str:
        """append：保存一帧（原图 + 标注图 + 清单一行），返回原始帧文件名。

        检出失败的帧也照样保存，只是 detected=0，方便事后回看是光照还是角度问题；
        它不会计入有效帧，选择窗口会提示继续补拍。
        """
        folder = self.dir_for(spec)
        os.makedirs(folder, exist_ok=True)
        index = self.next_index(spec)
        stem = f"{index:03d}"
        raw_name = f"raw_{stem}.png"
        axis_name = f"axis_{stem}.png"
        pe.write_image(os.path.join(folder, raw_name), frame)
        pe.write_image(os.path.join(folder, axis_name), vis)

        row = dict(row)
        row.update({
            "index": stem,
            "timestamp": f"{time.time():.6f}",
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "group": spec.key,
            "kind": spec.kind,
            "truth": f"{spec.truth:g}",
            "unit": spec.unit,
            "axis": spec.axis,
            "position": spec.position,
            "raw": raw_name,
            "annotated": axis_name,
        })

        path = self.csv_path(spec)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, extrasaction="ignore")
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        return raw_name

    def archive_group(self, spec: GroupSpec) -> str:
        """archive_group：重拍本组时把旧目录整体改名存档，不删任何照片。"""
        folder = self.dir_for(spec)
        if not os.path.isdir(folder):
            return ""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(self.root, f"{spec.key}_old_{stamp}")
        suffix = 1
        while os.path.exists(backup):
            backup = os.path.join(self.root, f"{spec.key}_old_{stamp}_{suffix}")
            suffix += 1
        shutil.move(folder, backup)
        return backup


# ---------------------------------------------------------------------------
# 测量：只调主程序的算法，不自己重写任何一步
# ---------------------------------------------------------------------------
def _fmt(value: Optional[float], digits: int = 6) -> str:
    """_fmt：把浮点数写成定长字符串；None / NaN 一律写成空单元格。"""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _pose_fields(result: pe.PoseResult, prefix: str = "") -> Dict[str, str]:
    """_pose_fields：把 PoseResult 的位姿展开成 CSV 单元格。

    prefix：字段名前缀，带畸变用空前缀，不带畸变用 "nodist_"。
    检测失败时全部留空，绝不用 0 冒充。
    """
    keys = ("x_m", "y_m", "z_m", "distance_m", "roll_deg", "pitch_deg",
            "yaw_deg", "reproj_px")
    if not result.ok or result.tvec is None:
        return {f"{prefix}{key}": "" for key in keys}
    x, y, z = (float(v) for v in result.tvec.ravel())
    values = (x, y, z, result.distance, result.roll, result.pitch,
              result.yaw, result.reproj_rmse)
    return {f"{prefix}{key}": _fmt(value) for key, value in zip(keys, values)}


def build_row(
    spec: GroupSpec,
    result: pe.PoseResult,
    no_dist_result: Optional[pe.PoseResult] = None,
    flow: str = "both",
) -> Dict[str, str]:
    """build_row：把一帧的测量结果整理成一行 CSV 数据。

    result：带畸变流程的结果（主结果，detected 与填表都以它为准）。
    no_dist_result：同一帧零畸变流程的结果；两套都跑，绝不缺列。
    flow：窗口里选的预览流程，只作记录，不改变两套数据的计算。
    """
    row: Dict[str, str] = {
        "detected": "1" if result.ok else "0",
        "reason": "" if result.ok else result.reason,
        "detector": result.detector or "",
        "flow": flow,
    }
    row.update(_pose_fields(result))

    # 同一帧、同一批角点，只把 dist 换成零再解一次 PnP，
    # 然后算两套结果的差（这正是题目 §2.5 要求的"同一帧对比"）。
    if no_dist_result is not None:
        row.update(_pose_fields(no_dist_result, prefix="nodist_"))
        if result.ok and no_dist_result.ok and result.tvec is not None \
                and no_dist_result.tvec is not None:
            delta_t = np.linalg.norm(result.tvec - no_dist_result.tvec)
            row.update({
                "dt_mm": _fmt(float(delta_t) * 1000.0),
                "droll_deg": _fmt(result.roll - no_dist_result.roll),
                "dpitch_deg": _fmt(result.pitch - no_dist_result.pitch),
                "dyaw_deg": _fmt(result.yaw - no_dist_result.yaw),
            })
    return row


def _run_pipeline(
    frame: np.ndarray,
    calib: pe.Calibration,
    use_distortion: bool,
    refine: bool,
    first_diag_dark: bool,
    full_fallback: bool,
) -> pe.PoseResult:
    """_run_pipeline：两套流程共用的底座，只调主程序 process_frame，不重写算法。

    full_fallback：打开"全分辨率经典 + 全分辨率 SB"两档兜底。平时必须是 False，
    这样正常光亮下的检测顺序与现场演示逐字一致；只有远距离快路径检不出时才开。
    """
    return pe.process_frame(
        frame, calib,
        use_distortion=use_distortion,
        refine=refine,
        full_classic_fallback=full_fallback,
        full_sb_fallback=full_fallback,
        first_diag_dark=first_diag_dark,
    )


def solve_with_distortion(
    frame: np.ndarray,
    calib: pe.Calibration,
    refine: bool,
    first_diag_dark: bool,
    full_fallback: bool = False,
) -> pe.PoseResult:
    """畸变流程：用 M2-1 标定的 [k1,k2,p1,p2,k3] 解 PnP、画轴（现场演示那一套）。"""
    return _run_pipeline(
        frame, calib, True, refine, first_diag_dark, full_fallback
    )


def solve_without_distortion(
    frame: np.ndarray,
    calib: pe.Calibration,
    refine: bool,
    first_diag_dark: bool,
    full_fallback: bool = False,
) -> pe.PoseResult:
    """非畸变流程：dist 全置零，其余（角点、顺序、flag、精化、阈值）完全相同。

    与畸变流程唯一的区别就是 dist，所以两套结果的差才是"畸变的影响"，
    不会混进检测或阈值差异。
    """
    return _run_pipeline(
        frame, calib, False, refine, first_diag_dark, full_fallback
    )


def solve_pair(
    frame: np.ndarray,
    calib: pe.Calibration,
    refine: bool,
    first_diag_dark: bool,
) -> Tuple[pe.PoseResult, pe.PoseResult]:
    """同帧两套：同一帧、同一批角点，只换 dist 解两次，供 |Δt| / Δ角 对比。

    两套都走 solve_flow，所以远距离时两套都享有同样的检测兜底，
    不会出现"一套检出、一套检不出"导致 Δ 算不出来。
    """
    return (
        solve_flow(frame, calib, True, refine, first_diag_dark),
        solve_flow(frame, calib, False, refine, first_diag_dark),
    )


def solve_flow(
    frame: np.ndarray,
    calib: pe.Calibration,
    use_distortion: bool,
    refine: bool,
    first_diag_dark: bool,
) -> pe.PoseResult:
    """按指定流程解一帧；只在"未检出棋盘格"时自动重试一次全分辨率兜底。

    1.20 m 时每格约 19 px（SB@0.4 后约 7.6 px），合成验证能稳定检出，
    但余量不大。这里不动 MAX_REPROJ_PX：放宽阈值只会放进坏帧，
    真正该补的是检测兜底，而且只在快路径失败时才触发。
    """
    solve = solve_with_distortion if use_distortion else solve_without_distortion
    result = solve(frame, calib, refine, first_diag_dark, full_fallback=False)
    if not result.ok and "未检出" in result.reason:
        retry = solve(frame, calib, refine, first_diag_dark, full_fallback=True)
        if retry.ok:
            retry.detector = f"{retry.detector}+full"
            return retry
    return result


# ---------------------------------------------------------------------------
# 预览窗口
# ---------------------------------------------------------------------------
def _draw_hud(
    vis: np.ndarray,
    spec: GroupSpec,
    done: int,
    result: pe.PoseResult,
    warmup: bool,
    flow: str = "both",
) -> None:
    """_draw_hud：在预览图上叠英文提示（OpenCV 字体不含中文）。"""
    height = vis.shape[0]
    state = "WARMING UP - WAIT" if warmup else ("NG - NOT COUNTED" if not result.ok else "OK")
    lines = [
        f"[{spec.key}]  flow={flow.upper()}  SPACE=shoot  ESC=back  {done}/{spec.frames}",
        f"state={state}  truth={spec.truth:g}{spec.unit}",
    ]
    if result.ok:
        lines.append(
            f"dist={result.distance:.4f}m  r={result.roll:+.1f} "
            f"p={result.pitch:+.1f} y={result.yaw:+.1f} "
            f"reproj={result.reproj_rmse:.2f}px"
        )
    else:
        lines.append(f"reason: {result.reason[:48]}")

    overlay = vis.copy()
    cv2.rectangle(overlay, (10, height - 20 - 26 * len(lines)),
                  (min(vis.shape[1] - 10, 950), height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (20, height - 20 - 26 * (len(lines) - 1 - i)),
                    FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def shoot_group(
    spec: GroupSpec,
    calib: pe.Calibration,
    store: GroupStore,
    camera: int,
    refine: bool,
    first_diag_dark: bool,
    reset_exposure: bool,
    flow: str = "both",
) -> Tuple[int, str]:
    """shoot_group：打开一个组的预览窗口，按空格拍一帧，拍满或按 Esc 返回。

    返回 (本次新拍的有效帧数, 状态说明)。摄像头在每次拍摄时开、拍完就放，
    选择窗口停留期间不占用设备。
    """
    # cap：优先 V4L2 后端，与 pose_estimate.run_camera 完全一致。
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        return 0, f"打不开摄像头 {camera}"

    try:
        # MJPG + 标定分辨率：避免默认 640×480 / YUYV 导致内参不匹配或绿屏。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib.height)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 内参是在固定焦距下标定的
        if reset_exposure:
            # V4L2 的自动曝光是"设备级状态"，重启程序不会自己恢复；
            # 这里传原始枚举 3=auto（OpenCV 文档里的 0.75 在本机不生效）。
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        # 丢掉开头若干帧，等自动曝光收敛、画面稳定，避免拍到发暗的帧。
        for _ in range(WARMUP_FRAMES):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            return 0, "摄像头读帧失败"

        # calib_cam：按实际分辨率缩放内参（本机标定即 1920×1080，缩放比为 1）。
        calib_cam = pe.scale_calibration(calib, frame.shape[1], frame.shape[0])
        print(f"[camera] {frame.shape[1]}x{frame.shape[0]}  group={spec.key}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        width = min(1280, frame.shape[1])
        cv2.resizeWindow(WINDOW_NAME, width, int(width * frame.shape[0] / frame.shape[1]))

        done, _ = store.progress(spec)
        fresh = 0
        shot_last = ""
        warmup_left = 0
        try:
            while True:
                # preview_with_dist：预览看哪一套流程；选 off 时看零畸变那套。
                preview_with_dist = flow != "off"
                shown = solve_flow(
                    frame, calib_cam, preview_with_dist, refine, first_diag_dark
                )
                dist_used = (calib_cam.dist if preview_with_dist
                             else np.zeros(5, dtype=np.float64))
                vis = pe.annotate_frame(frame, shown, calib_cam, dist_used, fps=None)
                _draw_hud(vis, spec, done, shown, warmup_left > 0, flow)
                if shot_last:
                    cv2.putText(vis, shot_last, (20, 40), FONT, 0.7,
                                (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if warmup_left > 0:
                    warmup_left -= 1
                elif key in (32, ord("c")):
                    # 两套流程在同一帧各解一次；无论窗口选的是哪一套，
                    # CSV 里带畸变/零畸变两列都齐全，填表的 |Δt| 不会缺数据。
                    result_on, result_off = solve_pair(
                        frame, calib_cam, refine, first_diag_dark
                    )
                    row = build_row(spec, result_on, result_off, flow)
                    raw_name = store.append(spec, row, frame, vis)
                    if row["detected"] == "1":
                        done += 1
                        fresh += 1
                        shot_last = f"SAVED {raw_name}  {done}/{spec.frames}"
                        print(f"[shot] {spec.key} {raw_name} OK  {done}/{spec.frames}")
                    else:
                        shot_last = f"SAVED {raw_name}  NG - NOT COUNTED"
                        print(f"[shot] {spec.key} {raw_name} NG  {result_on.reason}")
                    if done >= spec.frames:
                        break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break
        finally:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass
        return fresh, "完成" if done >= spec.frames else "中途返回"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# 汇总：所有数字都从 measure.csv 读出来算
# ---------------------------------------------------------------------------
def _numbers(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    """_numbers：从若干行里取某一列的数值，坏值直接跳过。"""
    out: List[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def _wrap_deg(value: float) -> float:
    """_wrap_deg：把角度归一化到 (-180, 180]，处理 179°→-179° 这类跨界。"""
    return (value + 180.0) % 360.0 - 180.0


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """_mean_std：均值和样本标准差（n<2 时标准差为 None）。"""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def summarize(store: GroupStore) -> Dict[str, object]:
    """summarize：把全部组的清单汇总成"三张表要用的数字"。

    返回结构是纯 dict/list，方便 --summary 打印、--finalize 填表、--selftest 造数据。
    缺数据的组用 None 占位，渲染时保留原来的空行，绝不补数。
    """
    warnings: List[str] = []
    by_key: Dict[str, List[Dict[str, str]]] = {
        spec.key: store.valid_rows(spec) for spec in GROUPS
    }

    # 零位基线：正对时三个角的均值。姿态误差必须扣掉它才对，
    # 否则"零位本来就是 89°"会被算成 89° 的测量误差。
    zero_rows = by_key.get("pose_zero", [])
    zero: Dict[str, Optional[float]] = {}
    for axis in ("roll", "pitch", "yaw"):
        vals = _numbers(zero_rows, f"{axis}_deg")
        zero[axis] = _mean_std(vals)[0]
    if any(zero[axis] is None for axis in ("roll", "pitch", "yaw")):
        warnings.append("缺少零位基线 pose_zero：姿态表改填原始输出值，误差含义受限")
    if zero_rows and len(zero_rows) < FRAMES_PER_GROUP:
        warnings.append(f"pose_zero 只有 {len(zero_rows)} 帧有效（建议 {FRAMES_PER_GROUP} 帧）")

    # 距离线性度：每档取最后 5 个有效帧。
    distance: List[Dict[str, object]] = []
    for spec in GROUPS:
        if spec.kind != "distance":
            continue
        rows = by_key[spec.key][-FRAMES_PER_GROUP:]
        vals = _numbers(rows, "distance_m")
        mean, std = _mean_std(vals)
        err_mm = None if mean is None else (mean - spec.truth) * 1000.0
        rel = None if err_mm is None else abs(err_mm) / (spec.truth * 1000.0) * 100.0
        distance.append({
            "label": f"{spec.truth:.2f}", "truth": spec.truth,
            "frames": vals, "n": len(vals), "mean": mean,
            "std_mm": None if std is None else std * 1000.0,
            "err_mm": err_mm, "rel_pct": rel,
        })
        if len(vals) < FRAMES_PER_GROUP:
            warnings.append(f"{spec.key} 只有 {len(vals)} 帧有效（需要 {FRAMES_PER_GROUP} 帧）")

    # 姿态响应：每档取最后 5 帧对应轴的角度，扣零位后与实际角比较。
    pose: List[Dict[str, object]] = []
    for spec in GROUPS:
        if spec.kind != "pose":
            continue
        rows = by_key[spec.key][-FRAMES_PER_GROUP:]
        raw_vals = _numbers(rows, f"{spec.axis}_deg")
        baseline = zero.get(spec.axis)
        vals = [_wrap_deg(v - baseline) if baseline is not None else v for v in raw_vals]
        mean, _ = _mean_std(vals)
        err = None if mean is None else _wrap_deg(mean - spec.truth)
        # 检出率：总帧数里 detected=1 的比例，90° 侧视经常检不到，要如实记。
        total = len(store.rows(spec))
        rate = None if total == 0 else len(by_key[spec.key]) / total
        pose.append({
            "axis": spec.axis, "angle": int(spec.truth),
            "frames": vals, "n": len(vals), "mean": mean, "err": err,
            "detected_rate": rate,
        })
        if len(vals) < FRAMES_PER_GROUP:
            warnings.append(f"{spec.key} 只有 {len(vals)} 帧有效（需要 {FRAMES_PER_GROUP} 帧）")

    # 畸变开/关：每帧已经存好两套结果和四个差值，取均值即可。
    distort: List[Dict[str, object]] = []
    for spec in GROUPS:
        if spec.kind != "distort":
            continue
        rows = by_key[spec.key][-FRAMES_PER_GROUP:]
        item: Dict[str, object] = {
            "distance": spec.truth, "position": spec.position, "n": len(rows),
        }
        for out_key, csv_key in (
            ("dt_mm", "dt_mm"), ("droll", "droll_deg"),
            ("dpitch", "dpitch_deg"), ("dyaw", "dyaw_deg"),
        ):
            item[out_key] = _mean_std(_numbers(rows, csv_key))[0]
        # 两套 t 和两套欧拉角，供 accuracy.md §3.2 的宽表使用（取均值）。
        for prefix, out_prefix in (("", "with"), ("nodist_", "without")):
            for axis in ("x", "y", "z"):
                item[f"{out_prefix}_t{axis}"] = _mean_std(
                    _numbers(rows, f"{prefix}{axis}_m")
                )[0]
            for axis in ("roll", "pitch", "yaw"):
                item[f"{out_prefix}_{axis}"] = _mean_std(
                    _numbers(rows, f"{prefix}{axis}_deg")
                )[0]
        distort.append(item)
        if len(rows) < FRAMES_PER_GROUP:
            warnings.append(f"{spec.key} 只有 {len(rows)} 帧有效（需要 {FRAMES_PER_GROUP} 帧）")

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zero": zero, "distance": distance, "pose": pose,
        "distort": distort, "warnings": warnings,
        "total_valid": sum(len(v) for v in by_key.values()),
    }


def print_summary(summary: Dict[str, object], store: GroupStore) -> None:
    """print_summary：--summary 用，逐组打印进度和已算出的统计。"""
    print(f"实测根目录: {store.root}")
    print(f"有效帧合计: {summary['total_valid']}")
    print("-" * 74)
    for spec in GROUPS:
        valid, total = store.progress(spec)
        flag = "OK " if valid >= spec.frames else "   "
        print(f"{flag}{spec.key:<24} {valid}/{spec.frames} 有效  "
              f"(共 {total} 帧)  {spec.label}")
    print("-" * 74)
    zero = summary["zero"]
    print("零位基线 (roll/pitch/yaw): " + " / ".join(
        "—" if zero.get(a) is None else f"{zero[a]:+.2f}°"
        for a in ("roll", "pitch", "yaw")
    ))
    print("距离线性度:")
    for item in summary["distance"]:  # type: ignore[union-attr]
        if item["mean"] is None:
            print(f"  {item['label']} m : 暂无有效帧")
        else:
            print(f"  {item['label']} m : n={item['n']}  mean={item['mean']:.4f} m  "
                  f"std={item['std_mm']:.2f} mm  err={item['err_mm']:+.2f} mm  "
                  f"rel={item['rel_pct']:.2f}%")
    print("姿态响应（已扣零位）:")
    for item in summary["pose"]:  # type: ignore[union-attr]
        if item["mean"] is None:
            print(f"  {item['axis']:<6}{item['angle']:>3}° : 暂无有效帧")
        else:
            rate = item["detected_rate"]
            print(f"  {item['axis']:<6}{item['angle']:>3}° : n={item['n']}  "
                  f"mean={item['mean']:+.2f}°  err={item['err']:+.2f}°  "
                  f"检出率={'—' if rate is None else format(rate * 100, '.0f') + '%'}")
    print("畸变开/关（带畸变 − 不带畸变）:")
    for item in summary["distort"]:  # type: ignore[union-attr]
        if item["dt_mm"] is None:
            print(f"  {item['distance']:.2f} m {item['position']} : 暂无有效帧")
        else:
            print(f"  {item['distance']:.2f} m {item['position']} : n={item['n']}  "
                  f"|Δt|={item['dt_mm']:.2f} mm  Δr={item['droll']:+.2f}°  "
                  f"Δp={item['dpitch']:+.2f}°  Δy={item['dyaw']:+.2f}°")
    if summary["warnings"]:
        print("-" * 74)
        print("提示:")
        for text in summary["warnings"]:  # type: ignore[union-attr]
            print("  ! " + text)


# ---------------------------------------------------------------------------
# 填表：按"表格块"精确定位，只换数据行，表头和分隔行原样保留
# ---------------------------------------------------------------------------
def _newline_of(line: str) -> str:
    """_newline_of：沿用原文件的换行风格（LF / CRLF）。"""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def replace_table(
    text: str,
    header_prefix: str,
    data_lines: Sequence[Optional[str]],
) -> Tuple[str, int]:
    """replace_table：把指定表格的数据行换成 data_lines，返回 (新文本, 替换行数)。

    header_prefix：表头行的开头字符串，必须在文件里唯一。
    data_lines：与表格数据行一一对应；None 表示"保留原来的空行"（缺数据时不动）。
    表头行、分隔行、以及表格以外的任何内容都不会被改写。
    """
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(header_prefix)]
    if len(hits) != 1:
        raise ValueError(f"表头定位失败（找到 {len(hits)} 处）: {header_prefix}")
    start = hits[0]
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    block = lines[start:end]
    if len(block) < 3:
        raise ValueError(f"表格块不足三行（表头/分隔/数据）: {header_prefix}")
    header_count = len(block) - 2  # 数据行数
    if len(data_lines) == header_count:
        # 正常情况：逐行替换，None 表示"这格没数据，保留原来的空行"。
        replaced = 0
        new_block = list(block[:2])
        for old_line, new_line in zip(block[2:], data_lines):
            if new_line is None:
                new_block.append(old_line)
            else:
                new_block.append(new_line + _newline_of(old_line))
                replaced += 1
        return "".join(lines[:start] + new_block + lines[end:]), replaced

    # 行数不一致：一般是模板漏行（report.md §7.2 的表格少了 pitch 90）。
    # 空表可以按题目要求的行集合重建数据区；已有手填数据时拒绝，避免丢记录。
    if _table_has_data(text, header_prefix):
        raise ValueError(
            f"表格行数与待填行数不一致，且已有数据，拒绝重建: {header_prefix}"
        )
    print(f"[fill] 表格行数 {header_count} → {len(data_lines)}（补齐模板漏行）: "
          f"{header_prefix}")
    n_cols = block[1].strip().strip("|").count("|") + 1
    nl = _newline_of(block[2]) if len(block) > 2 else "\n"
    new_block = list(block[:2])
    replaced = 0
    for new_line in data_lines:
        if new_line is None:
            new_block.append("| " + " | ".join([""] * n_cols) + " |" + nl)
        else:
            new_block.append(new_line + nl)
            replaced += 1
    return "".join(lines[:start] + new_block + lines[end:]), replaced


def ensure_annotation(
    text: str,
    anchor: str,
    note: str,
    before: bool = False,
) -> str:
    """ensure_annotation：在指定行附近补一句口径说明，已存在则不动（幂等）。

    只用于把"表里这些数字是什么口径"讲清楚，不改动任何原有文字。
    """
    marker = note.strip()
    if marker in text:
        return text
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() != anchor:
            continue
        nl = _newline_of(line) or "\n"
        lines.insert(i if before else i + 1, nl + marker + nl)
        return "".join(lines)
    return text


def fill_blank_lines(text: str, mapping: Dict[str, Sequence[str]]) -> str:
    """fill_blank_lines：替换结论段里形如 "____" 的占位。

    mapping：{行前缀: [依次填入的值...]}；一行有多个占位时按顺序填。
    找不到的前缀直接跳过，不会因为报告改了几句话就让整个填表失败。
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for prefix, values in mapping.items():
            if not stripped.startswith(prefix):
                continue
            for value in values:
                if "______" in lines[i]:
                    lines[i] = lines[i].replace("______", str(value), 1)
            break
    return "".join(lines)


def _f(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    """_f：表格里的数字格式化；None 用 "—" 表示"这一格还没数据"。"""
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _frames5(values: Sequence[float], digits: int = 4) -> List[str]:
    """_frames5：补齐到 5 个单元格，缺的留空（保持表格列数不变）。"""
    cells = [f"{v:.{digits}f}" for v in list(values)[:FRAMES_PER_GROUP]]
    return cells + [""] * (FRAMES_PER_GROUP - len(cells))


def _triple(values: Dict[str, object], keys: Sequence[str], digits: int) -> str:
    """_triple：把三个数写成 "(a,b,c)"，任一缺失就整格留 "—"。"""
    if any(values.get(key) is None for key in keys):
        return "—"
    return "(" + ",".join(f"{float(values[key]):.{digits}f}" for key in keys) + ")"


def build_table_edits(summary: Dict[str, object]) -> Dict[str, List[Tuple[str, List[Optional[str]]]]]:
    """build_table_edits：按两张文档里的六张表，生成"表头前缀 → 新数据行"。

    缺数据的行返回 None，replace_table 会保留原来的空单元格。
    """
    distance = {item["label"]: item for item in summary["distance"]}  # type: ignore[union-attr]
    pose = {(item["axis"], item["angle"]): item for item in summary["pose"]}  # type: ignore[union-attr]
    distort = {(f"{item['distance']:.2f}", item["position"]): item
               for item in summary["distort"]}  # type: ignore[union-attr]

    report: List[Tuple[str, List[Optional[str]]]] = []
    accuracy: List[Tuple[str, List[Optional[str]]]] = []

    # --- report.md §7.1 距离线性度（4 行 × 5 列）---
    rep_dist: List[Optional[str]] = []
    for spec in GROUPS:
        if spec.kind != "distance":
            continue
        item = distance[f"{spec.truth:.2f}"]
        if item["mean"] is None:
            rep_dist.append(None)
        else:
            rep_dist.append(
                f"| {item['label']} | {_f(item['mean'], 4)} | "
                f"{_f(item['std_mm'])} | {_f(item['err_mm'], signed=True)} | "
                f"{_f(item['rel_pct'])}% |"
            )
    report.append(("| 实际距离 (m) |", rep_dist))

    # --- accuracy.md §1.2 距离线性度（4 行 × 10 列，含 5 帧原始值）---
    acc_dist: List[Optional[str]] = []
    for spec in GROUPS:
        if spec.kind != "distance":
            continue
        item = distance[f"{spec.truth:.2f}"]
        if item["mean"] is None:
            acc_dist.append(None)
        else:
            cells = " | ".join(_frames5(item["frames"]))
            acc_dist.append(
                f"| {item['truth']:.3f} | {cells} | {_f(item['mean'], 4)} | "
                f"{_f(item['std_mm'])} | {_f(item['err_mm'], signed=True)} | "
                f"{_f(item['rel_pct'])}% |"
            )
    accuracy.append(("| 实际距离 | 第 1 帧 |", acc_dist))

    # --- report.md §7.2 姿态响应（9 行 × 4 列）---
    rep_pose: List[Optional[str]] = []
    # --- accuracy.md §2.2 姿态响应（9 行 × 9 列，含 5 帧原始值）---
    acc_pose: List[Optional[str]] = []
    for spec in GROUPS:
        if spec.kind != "pose":
            continue
        item = pose[(spec.axis, int(spec.truth))]
        if item["mean"] is None:
            rep_pose.append(None)
            acc_pose.append(None)
        else:
            rep_pose.append(
                f"| {spec.axis} | {int(spec.truth)} | "
                f"{_f(item['mean'], signed=True)} | {_f(item['err'], signed=True)} |"
            )
            cells = " | ".join(_frames5(item["frames"], digits=2))
            acc_pose.append(
                f"| {spec.axis} | {int(spec.truth)} | {cells} | "
                f"{_f(item['mean'], signed=True)} | {_f(item['err'], signed=True)} |"
            )
    report.append(("| 轴 | 实际角 (°) |", rep_pose))
    accuracy.append(("| 轴 | 实际角 | 第 1 帧 |", acc_pose))

    # --- report.md §7.3 畸变的影响（2 行 × 6 列）---
    rep_distor: List[Optional[str]] = []
    # --- accuracy.md §3.2 畸变对照（2 行 × 10 列）---
    acc_distor: List[Optional[str]] = []
    for spec in GROUPS:
        if spec.kind != "distort":
            continue
        item = distort[(f"{spec.truth:.2f}", spec.position)]
        if item["dt_mm"] is None:
            rep_distor.append(None)
            acc_distor.append(None)
        else:
            rep_distor.append(
                f"| {item['distance']:.2f} | {item['position']} | "
                f"{_f(item['dt_mm'])} | {_f(item['droll'], signed=True)} | "
                f"{_f(item['dpitch'], signed=True)} | {_f(item['dyaw'], signed=True)} |"
            )
            t_with = _triple(item, ("with_tx", "with_ty", "with_tz"), 4)
            t_without = _triple(item, ("without_tx", "without_ty", "without_tz"), 4)
            r_with = _triple(item, ("with_roll", "with_pitch", "with_yaw"), 2)
            r_without = _triple(item, ("without_roll", "without_pitch", "without_yaw"), 2)
            acc_distor.append(
                f"| {item['distance']:.2f} | {item['position']} | {t_with} | "
                f"{t_without} | {_f(item['dt_mm'])} | {r_with} | {r_without} | "
                f"{_f(item['droll'], signed=True)} | "
                f"{_f(item['dpitch'], signed=True)} | "
                f"{_f(item['dyaw'], signed=True)} |"
            )
    # 注意：report.md §7.3 表头把竖线转义成了 \|，前缀只取到转义前的位置，
    # 避免在 Python 字符串里和反斜杠较劲；该前缀在 report.md 里唯一。
    report.append(("| 距离 (m) | 画面位置 | ", rep_distor))
    accuracy.append(("| 距离 (m) | 画面位置 | 带畸变 t", acc_distor))

    return {"report": report, "accuracy": accuracy}


def _conclusion_edits(summary: Dict[str, object]) -> Dict[str, Sequence[str]]:
    """_conclusion_edits：能由数据直接算出的结论，替掉 accuracy.md 里的 ____ 占位。

    需要人判断的项（例如"是否满足现场要求"）给客观数据 + 明确标注"现场确认"，
    不替用户下结论。
    """
    mapping: Dict[str, Sequence[str]] = {}
    distance = summary["distance"]  # type: ignore[assignment]
    pose = summary["pose"]  # type: ignore[assignment]
    distort = summary["distort"]  # type: ignore[assignment]

    errs = [abs(item["err_mm"]) for item in distance if item["err_mm"] is not None]
    rels = [item["rel_pct"] for item in distance if item["rel_pct"] is not None]
    if errs:
        mapping["- 0.30–1.20 m 内最大绝对误差："] = [f"{max(errs):.2f}"]
        mapping["- 最大相对误差："] = [f"{max(rels):.2f}"]
        # 单调性：按距离顺序看 |err| 是否逐档不减。
        ordered = [item["err_mm"] for item in distance]
        if all(v is not None for v in ordered):
            magnitude = [abs(v) for v in ordered]
            mono = all(b >= a - 1e-9 for a, b in zip(magnitude, magnitude[1:]))
            mapping["- 误差随距离是否单调增大："] = ["是" if mono else "否（有起伏）"]
        worst = max(errs)
        worst_rel = max(rels)
        ok = worst <= 10.0 and worst_rel <= 2.0
        mapping["- 是否满足现场要求（例如 ±10 mm / ±2%）："] = [
            ("满足" if ok else "不满足") + "（按 ±10 mm / ±2% 自评，阈值以现场为准）"
        ]

    # 单调响应：每个轴的 30/60/90 输出是否逐档递增。
    by_axis: Dict[str, List[Optional[float]]] = {a: [] for a, _ in AXES}
    for axis, _ in AXES:
        for angle in ANGLES:
            by_axis[axis].append(pose_axis_value(pose, axis, angle))
    has_full_pose = all(all(v is not None for v in vals) for vals in by_axis.values())
    if has_full_pose:
        mono_all = all(
            all(b > a for a, b in zip(vals, vals[1:]))  # type: ignore[arg-type]
            for vals in by_axis.values()
        )
        mapping["- 三个轴是否都能单调响应："] = [
            "是" if mono_all else "否（见下表逐档值）"
        ]
    low_err = [abs(item["err"]) for item in pose
               if item["angle"] in (30, 60) and item["err"] is not None]
    if low_err:
        mapping["- 30°/60° 最大误差："] = [f"{max(low_err):.2f}"]
    at90 = [item for item in pose if item["angle"] == 90]
    detected = [item for item in at90 if item["mean"] is not None]
    rates = [item["detected_rate"] for item in at90 if item["detected_rate"] is not None]
    if rates:
        stable = len(detected) == len(at90) and min(rates) >= 0.8
        mapping["- 90° 是否可稳定检出："] = [
            "是" if stable else "否",
            f"{min(rates) * 100:.0f}%",
        ]
    zero = summary["zero"]  # type: ignore[assignment]
    if any(zero.get(a) is not None for a in ("roll", "pitch", "yaw")):
        mapping["- 正对时零位偏差（roll/pitch/yaw）："] = [
            "—" if zero.get(a) is None else f"{zero[a]:+.2f}"
            for a in ("roll", "pitch", "yaw")
        ]

    centers = [item["dt_mm"] for item in distort
               if item["position"] == "中心" and item["dt_mm"] is not None]
    edges = [item["dt_mm"] for item in distort
             if item["position"] == "边缘" and item["dt_mm"] is not None]
    if centers:
        mapping["- 中心区域 |Δt| 最大："] = [f"{max(centers):.2f}"]
    if edges:
        mapping["- 边缘区域 |Δt| 最大："] = [f"{max(edges):.2f}"]
    if centers and edges:
        mapping["- 边缘是否明显大于中心："] = [
            "是" if max(edges) > max(centers) * 1.5 else "否（差异不明显）"
        ]
        mapping["- 结论：现场是否有必要保留畸变模型："] = [
            "有必要（边缘差值更大），阈值以现场确认" if max(edges) > max(centers) * 1.5
            else "差异不明显，建议现场再确认"
        ]
    # 角度最大差：取三轴差值绝对值的最大。
    angles_all = [abs(item[key]) for item in distort
                  for key in ("droll", "dpitch", "dyaw") if item[key] is not None]
    if angles_all:
        for prefix, subset in (
            ("- 中心区域 |Δt| 最大：", "中心"),
            ("- 边缘区域 |Δt| 最大：", "边缘"),
        ):
            vals = [abs(item[key]) for item in distort if item["position"] == subset
                    for key in ("droll", "dpitch", "dyaw") if item[key] is not None]
            if vals and prefix in mapping:
                # 该行形如 "- 中心区域 |Δt| 最大：______ mm，角度最大差：______°"
                mapping[prefix] = [mapping[prefix][0], f"{max(vals):.2f}"]
    return mapping


def pose_axis_value(pose: Sequence[Dict[str, object]], axis: str, angle: int) -> Optional[float]:
    """pose_axis_value：从姿态汇总里取 (轴, 角度) 的输出均值。"""
    for item in pose:
        if item["axis"] == axis and item["angle"] == angle:
            return item["mean"]  # type: ignore[return-value]
    return None


def _key_columns(header_prefix: str) -> int:
    """_key_columns：表格前几列是题目给的"键"，本来就有值，不算手填数据。

    距离表只有 1 列键（实际距离）；姿态表/畸变表有 2 列（轴+实际角、距离+位置）。
    """
    if "实际角" in header_prefix or "画面位置" in header_prefix:
        return 2
    return 1


def _table_has_data(text: str, header_prefix: str) -> bool:
    """_table_has_data：目标表格是否已经有手填的数字（防止覆盖现场记录）。"""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(header_prefix)]
    if len(hits) != 1:
        return False
    key_columns = _key_columns(header_prefix)
    i = hits[0] + 2  # 跳过表头和分隔行
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if any(cell for cell in cells[key_columns:]):
            return True
        i += 1
    return False


def fill_tables(
    summary: Dict[str, object],
    dry_run: bool = False,
    force: bool = False,
    paths: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """fill_tables：把汇总结果写进 report.md §7 和 results/accuracy.md。

    dry_run=True 时只算出新文本、打印 diff，不写任何文件。
    返回 {"files": [...], "replaced": n, "skipped": [...], "dry_run": bool}。
    """
    target = paths or {"report": REPORT_PATH, "accuracy": ACCURACY_PATH}
    # 没拍就绝不改文档：避免误点"生成实测表"把空表改成带占位符的样子。
    if int(summary.get("total_valid", 0)) == 0 and not force:
        return {
            "files": [], "replaced": 0,
            "skipped": ["results/measure/ 里还没有有效实测帧，未改动任何文档"],
            "dry_run": dry_run,
        }
    edits = build_table_edits(summary)
    conclusions = _conclusion_edits(summary)

    result: Dict[str, object] = {
        "files": [], "replaced": 0, "skipped": [], "dry_run": dry_run,
    }
    for name in ("report", "accuracy"):
        path = target[name]
        with open(path, "r", encoding="utf-8") as handle:
            original = handle.read()

        # 已有手填数据时默认拒写，避免把现场记的数冲掉。
        dirty = [prefix for prefix, _ in edits[name]
                 if _table_has_data(original, prefix)]
        if dirty and not force:
            result["skipped"].append(  # type: ignore[union-attr]
                f"{os.path.basename(path)} 已有数据：{dirty}（加 --force 才覆盖）"
            )
            continue

        text = original
        replaced = 0
        for prefix, data_lines in edits[name]:
            text, count = replace_table(text, prefix, data_lines)
            replaced += count
        if name == "accuracy":
            text = fill_blank_lines(text, conclusions)
            # 两条口径说明：表里数字怎么来的，答辩会被追问，必须写进文档。
            text = ensure_annotation(
                text, "计算公式：",
                "> 表中 5 帧原始值与均值单位 m；标准差、绝对误差单位 mm，"
                "相对误差单位 %；只统计 detected=1 且重投影误差达标的帧。",
                before=True,
            )
            text = ensure_annotation(
                text, "右手定则，拇指指向该轴正方向，四指弯曲方向为正转。",
                "> 本表「第 1~5 帧 / 均值」已扣除正对零位基线（pose_zero 组），"
                "填的是相对零位的转角响应；误差 = 该响应 − 实际角。"
                "未拍零位基线时填原始输出值，误差栏仅供参考。",
            )
        result["replaced"] = int(result["replaced"]) + replaced  # type: ignore[arg-type]
        result["files"].append({  # type: ignore[union-attr]
            "path": path, "replaced_rows": replaced, "changed": text != original,
        })

        if dry_run:
            print(f"\n===== 预览（不会写入）: {path} =====")
            _print_diff(original, text)
        else:
            if text != original:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                shutil.copy2(path, f"{path}.bak-{stamp}")
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                print(f"[fill] {path}  替换 {replaced} 行  "
                      f"(已备份 .bak-{stamp})")
            else:
                print(f"[fill] {path}  无变化")
    return result


def _print_diff(old: str, new: str) -> None:
    """_print_diff：只打印真正发生变化的行，方便填表前肉眼核对。"""
    import difflib
    diff = difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", n=0,
        fromfile="原文件", tofile="填表后",
    )
    shown = 0
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        print(line)
        shown += 1
        if shown >= 80:
            print("... (diff 过长，已截断)")
            break
    if shown == 0:
        print("(没有任何变化)")


# ---------------------------------------------------------------------------
# 自检：造一批明显的假数据，验证表格定位/渲染，不写任何文件
# ---------------------------------------------------------------------------
def selftest() -> int:
    """selftest：只验证"填表程序能不能正确找到并替换六张表"，不碰实测数据。

    用的全是 1.11 / 2.22 这类一眼假的值，且 dry_run=True，
    因此不会写进 report.md / accuracy.md，也不会留下任何文件。
    """
    print("[selftest] 以下数字全部是自检假数据，不会写入任何文件")
    distance = []
    for i, spec in enumerate([g for g in GROUPS if g.kind == "distance"]):
        frames = [spec.truth + 0.001 * (j + 1) for j in range(5)]
        distance.append({
            "label": f"{spec.truth:.2f}", "truth": spec.truth,
            "frames": frames, "n": 5, "mean": statistics.fmean(frames),
            "std_mm": 1.11 + i, "err_mm": 2.22 + i, "rel_pct": 0.33 + i,
        })
    pose = []
    for spec in [g for g in GROUPS if g.kind == "pose"]:
        frames = [float(spec.truth) + 0.1 * j for j in range(5)]
        pose.append({
            "axis": spec.axis, "angle": int(spec.truth), "frames": frames,
            "n": 5, "mean": statistics.fmean(frames), "err": 0.55,
            "detected_rate": 1.0,
        })
    distort = []
    for spec in [g for g in GROUPS if g.kind == "distort"]:
        item: Dict[str, object] = {
            "distance": spec.truth, "position": spec.position, "n": 5,
            "dt_mm": 3.33, "droll": 0.11, "dpitch": -0.22, "dyaw": 0.33,
        }
        for out_prefix in ("with", "without"):
            for axis, base in (("x", 0.01), ("y", 0.02), ("z", 0.30)):
                item[f"{out_prefix}_t{axis}"] = base
            for axis in ("roll", "pitch", "yaw"):
                item[f"{out_prefix}_{axis}"] = 1.0
        distort.append(item)
    summary: Dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zero": {"roll": 1.0, "pitch": 2.0, "yaw": 3.0},
        "distance": distance, "pose": pose, "distort": distort,
        "warnings": [], "total_valid": 999,
    }
    try:
        outcome = fill_tables(summary, dry_run=True, force=True)
    except Exception as exc:  # noqa: BLE001 - 自检要把任何定位失败都摊开
        print(f"[selftest] 失败: {type(exc).__name__}: {exc}")
        return 1
    replaced = int(outcome["replaced"])
    files = outcome["files"]  # type: ignore[assignment]
    print(f"\n[selftest] 两张文件共替换 {replaced} 行，涉及 {len(files)} 个文件")
    if outcome["skipped"]:
        print(f"[selftest] 跳过: {outcome['skipped']}")
    # 6 张表：距离 4 + 姿态 9 + 畸变 2，两张文档各一份 = 30 行。
    expected = 4 + 4 + 9 + 9 + 2 + 2
    if replaced != expected:
        print(f"[selftest] 失败: 期望替换 {expected} 行，实际 {replaced} 行")
        return 1
    # 顺带验证两条口径说明的锚点还在（报告改版后锚点会失效，要能提前发现）。
    with open(ACCURACY_PATH, "r", encoding="utf-8") as handle:
        sample = handle.read()
    for anchor, marker, before in (
        ("计算公式：", "> 表中 5 帧原始值与均值单位 m", True),
        ("右手定则，拇指指向该轴正方向，四指弯曲方向为正转。",
         "> 本表「第 1~5 帧 / 均值」已扣除正对零位基线", False),
    ):
        rendered = ensure_annotation(sample, anchor, marker, before=before)
        if marker not in rendered:
            print(f"[selftest] 失败: annotation anchor missing -> {anchor}")
            return 1
    print("[selftest] annotation anchors OK")
    print("[selftest] 通过：六张表全部定位成功，行数正确，未写入任何文件")
    return 0


# ---------------------------------------------------------------------------
# 选择窗口（tkinter，仅在 --gui 时才 import）
# ---------------------------------------------------------------------------
def run_gui(
    calib: pe.Calibration,
    store: GroupStore,
    camera: int,
    refine: bool,
    first_diag_dark: bool,
    reset_exposure: bool,
    force: bool,
    flow: str = "both",
) -> None:
    """run_gui：选择窗口主循环。窗口里只做"选组 / 拍摄 / 生成实测表"三件事。

    flow：窗口打开时默认选中的流程，仍可在窗口里随时改。
    """
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("缺少 tkinter，无法打开选择窗口；可改用 --finalize") from exc

    class Selector(tk.Tk):
        """Selector：左侧分组清单 + 底部操作按钮。"""

        def __init__(self) -> None:
            super().__init__()
            self.title("M2-2 现场精度实测 · 拍摄台")
            self.geometry("900x700")
            self.minsize(760, 580)
            # item_to_key：Treeview 行 id → 组名。
            self.item_to_key: Dict[str, str] = {}
            # flow：预览用哪一套流程；两套结果都存，切换不影响填表数据。
            self.flow = tk.StringVar(value=flow)
            self._build()
            self.refresh()

        def _build(self) -> None:
            """_build：搭界面骨架。"""
            top = ttk.Frame(self, padding=(12, 10))
            top.pack(fill="x")
            ttk.Label(top, text="M2-2 现场精度实测 · 拍摄台",
                      font=("Sans", 14, "bold")).pack(anchor="w")
            ttk.Label(
                top,
                text=(f"相机 /dev/video{camera}　内参 {os.path.basename(calib.path)}　"
                      f"{calib.width}×{calib.height}　自动对焦已关"),
            ).pack(anchor="w", pady=(2, 0))
            ttk.Label(
                top,
                text=("顺序：先拍【零位基线】→ 摆好靶标选一组 → 开始拍摄 → "
                      "等画面稳定后按空格拍 5 帧 → 拍满自动回到这里"),
                foreground="#555555",
            ).pack(anchor="w", pady=(2, 0))
            ttk.Label(
                top,
                text=("零位 / 姿态 / 畸变全部在 0.50 m 基准线拍；只有距离线性度四档另量。"
                      "yaw/pitch 的 90° 检不出可跳过，表格会如实记「不可测」。"),
                foreground="#555555",
            ).pack(anchor="w", pady=(2, 0))

            flow_bar = ttk.LabelFrame(
                top, text="解算流程（两套结果都会存，这里只切换预览画面）",
                padding=(8, 4),
            )
            flow_bar.pack(fill="x", pady=(6, 0))
            for key, text in FLOW_CHOICES:
                ttk.Radiobutton(
                    flow_bar, text=text, value=key, variable=self.flow,
                ).pack(side="left", padx=(0, 16))

            middle = ttk.Frame(self, padding=(12, 6))
            middle.pack(fill="both", expand=True)
            columns = ("group", "truth", "progress", "state")
            self.tree = ttk.Treeview(middle, columns=columns, show="tree headings",
                                     selectmode="browse", height=20)
            self.tree.heading("#0", text="实验组 / 类别")
            self.tree.heading("group", text="目录名")
            self.tree.heading("truth", text="真值")
            self.tree.heading("progress", text="有效帧")
            self.tree.heading("state", text="状态")
            self.tree.column("#0", width=300, anchor="w")
            self.tree.column("group", width=190, anchor="w")
            self.tree.column("truth", width=90, anchor="center")
            self.tree.column("progress", width=90, anchor="center")
            self.tree.column("state", width=90, anchor="center")
            scroll = ttk.Scrollbar(middle, orient="vertical",
                                   command=self.tree.yview)
            self.tree.configure(yscrollcommand=scroll.set)
            self.tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            self.tree.bind("<Double-1>", lambda _e: self.start_shooting())

            bottom = ttk.Frame(self, padding=(12, 8))
            bottom.pack(fill="x")
            ttk.Button(bottom, text="开始拍摄选中组",
                       command=self.start_shooting).pack(side="left")
            ttk.Button(bottom, text="重拍本组（旧数据存档）",
                       command=self.restart_group).pack(side="left", padx=(8, 0))
            ttk.Button(bottom, text="刷新进度",
                       command=self.refresh).pack(side="left", padx=(8, 0))
            ttk.Button(bottom, text="生成实测表",
                       command=self.finalize).pack(side="right")
            ttk.Button(bottom, text="退出",
                       command=self.destroy).pack(side="right", padx=(0, 8))

            self.status = tk.StringVar(value="就绪")
            ttk.Label(self, textvariable=self.status, padding=(12, 4),
                      foreground="#333333").pack(fill="x")

        def selected(self) -> Optional[GroupSpec]:
            """selected：当前选中的组；选中的是类别节点时返回 None。"""
            item = self.tree.focus()
            key = self.item_to_key.get(item)
            return GROUP_BY_KEY.get(key) if key else None

        def refresh(self) -> None:
            """refresh：重新扫一遍磁盘，重建清单并更新进度。"""
            self.tree.delete(*self.tree.get_children())
            self.item_to_key.clear()
            parents: Dict[str, str] = {}
            total_ok = 0
            for spec in GROUPS:
                if spec.category not in parents:
                    parents[spec.category] = self.tree.insert(
                        "", "end", text=spec.category, open=True
                    )
                valid, total = store.progress(spec)
                state = "完成" if valid >= spec.frames else ("进行中" if valid else "未拍")
                if valid >= spec.frames:
                    total_ok += 1
                item = self.tree.insert(
                    parents[spec.category], "end", text=spec.label,
                    values=(spec.key, f"{spec.truth:g} {spec.unit}",
                            f"{valid}/{spec.frames}", state),
                )
                self.item_to_key[item] = spec.key
            self.status.set(
                f"已完成 {total_ok}/{len(GROUPS)} 组　"
                f"数据目录 results/measure/　按空格拍帧，Esc 提前返回"
            )

        def start_shooting(self) -> None:
            """start_shooting：隐藏选择窗口，进 OpenCV 预览，回来刷新进度。"""
            spec = self.selected()
            if spec is None:
                messagebox.showinfo("提示", "请先在列表里选中一个实验组")
                return
            valid, _ = store.progress(spec)
            if valid >= spec.frames:
                if not messagebox.askyesno(
                    "已完成", f"【{spec.label}】已有 {valid} 帧有效数据，还要继续补拍吗？"
                ):
                    return
            self.withdraw()
            try:
                fresh, note = shoot_group(
                    spec, calib, store, camera, refine,
                    first_diag_dark, reset_exposure, self.flow.get(),
                )
            except Exception as exc:  # noqa: BLE001 - 现场不能因为一帧异常整个崩掉
                fresh, note = 0, f"{type(exc).__name__}: {exc}"
            finally:
                self.deiconify()
            self.refresh()
            self.status.set(f"【{spec.label}】本次新增 {fresh} 帧有效数据（{note}）")
            if "打不开" in note or "失败" in note:
                messagebox.showerror("拍摄出错", note)

        def restart_group(self) -> None:
            """restart_group：把本组旧目录整体改名存档，进度清零，绝不删照片。"""
            spec = self.selected()
            if spec is None:
                messagebox.showinfo("提示", "请先选中一个实验组")
                return
            if not messagebox.askyesno(
                "重拍本组",
                f"【{spec.label}】的旧照片会整体改名存档（不会删除），本组进度清零。\n确定吗？",
            ):
                return
            backup = store.archive_group(spec)
            self.refresh()
            self.status.set(f"已存档旧数据：{os.path.basename(backup) if backup else '（本来就是空的）'}")

        def finalize(self) -> None:
            """finalize：汇总并填表；缺数据的组会列出来让用户确认。"""
            summary = summarize(store)
            missing = [w for w in summary["warnings"]]  # type: ignore[union-attr]
            message = (
                f"有效帧合计 {summary['total_valid']} 帧。\n"
                f"将把实测数字填进 report.md §7 和 results/accuracy.md。\n"
                f"没拍够的组会留空，不会补数。\n"
            )
            if missing:
                message += f"\n有 {len(missing)} 条数据提示：\n" + "\n".join(
                    "  · " + w for w in missing[:8]
                )
                if len(missing) > 8:
                    message += f"\n  · ...还有 {len(missing) - 8} 条"
            if not messagebox.askyesno("生成实测表", message + "\n\n继续吗？"):
                return
            try:
                outcome = fill_tables(summary, force=force)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("填表失败", f"{type(exc).__name__}: {exc}")
                return
            skipped = outcome["skipped"]  # type: ignore[assignment]
            text = f"共替换 {outcome['replaced']} 行。"
            if skipped:
                text += "\n\n跳过的文件：\n" + "\n".join(skipped)
            messagebox.showinfo("填表完成", text)
            self.status.set(text.replace("\n", " "))

    try:
        app = Selector()
        app.mainloop()
    except tk.TclError as exc:
        raise RuntimeError(
            f"打不开图形窗口（{exc}）。WSL 请确认 WSLg 可用；"
            f"纯命令行可用 --finalize / --summary"
        ) from exc


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """parse_args：默认打开选择窗口；--finalize/--summary/--selftest 为无窗口模式。"""
    parser = argparse.ArgumentParser(
        description="M2-2 现场精度实测拍摄台（选择窗口 + 空格拍摄 + 自动填表）"
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--calib", default=pe.DEFAULT_CALIB,
                        help="M2-1 的标定 YAML（默认复用主程序的路径）")
    parser.add_argument("--square-mm", type=float, default=None,
                        help="重新量过方格边长时覆盖；默认读 YAML")
    parser.add_argument("--corner-diag", choices=("dark", "light"), default="dark",
                        help="首角点主对角线颜色，与主程序默认一致：dark")
    parser.add_argument("--flow", choices=("both", "on", "off"), default="both",
                        help="窗口默认选中的解算流程；both=同帧两套（填表用）")
    parser.add_argument("--no-refine", action="store_true",
                        help="关闭 solvePnPRefineLM（默认开启，与现场演示一致）")
    parser.add_argument("--no-exposure-reset", action="store_true",
                        help="不重设 V4L2 自动曝光；默认启动时设 3=auto")
    parser.add_argument("--finalize", action="store_true",
                        help="只汇总并填表，不打开窗口")
    parser.add_argument("--summary", action="store_true",
                        help="只打印各组进度与统计，不写文件")
    parser.add_argument("--selftest", action="store_true",
                        help="自检表格定位与渲染，使用假数据且绝不写文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="配合 --finalize：只打印将写入的内容，不落盘")
    parser.add_argument("--force", action="store_true",
                        help="目标表格已有数据时仍然覆盖")
    parser.add_argument("--measure-dir", default=MEASURE_DIR,
                        help=f"实测数据根目录，默认 {MEASURE_DIR}")
    return parser.parse_args(argv)


def main() -> int:
    """main：按参数分流。无窗口模式不需要 DISPLAY，也不会 import tkinter。"""
    args = parse_args()
    if args.selftest:
        return selftest()

    if args.finalize or args.summary:
        store = GroupStore(args.measure_dir)
        summary = summarize(store)
        if args.summary:
            print_summary(summary, store)
            return 0
        outcome = fill_tables(summary, dry_run=args.dry_run, force=args.force)
        for text in outcome["skipped"]:  # type: ignore[union-attr]
            print("[skip] " + text)
        print(f"[done] 替换 {outcome['replaced']} 行，dry_run={args.dry_run}")
        return 0

    calib = pe.load_calibration(args.calib, args.square_mm)
    store = GroupStore(args.measure_dir)
    print(f"[calib] {calib.path}  {calib.width}x{calib.height}  "
          f"square={calib.square_m * 1000:.1f}mm")
    try:
        run_gui(
            calib=calib, store=store, camera=args.camera,
            refine=not args.no_refine,
            first_diag_dark=(args.corner_diag == "dark"),
            reset_exposure=not args.no_exposure_reset,
            force=args.force, flow=args.flow,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
