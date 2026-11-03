import argparse
import csv
import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIB = os.path.normpath(
    os.path.join(HERE, "..", "M2-1", "camera_calibration.yaml")
)
DEFAULT_SAVE_DIR = os.path.join(HERE, "results")

CAPTURE_FPS = 15
SB_PRIMARY_SCALE = 0.4
CLASSIC_SCALE = 0.6
SB_FALLBACK_SCALE = 0.5
CORNER_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
REPROJ_WARN_PX = 2.0
MAX_REPROJ_PX = 3.0
SIGNATURE_MIN_CONTRAST = 20.0

# 物体系的“正对相机”姿态：物体系 X 向右、Y 向上、Z 指向相机时，
# 该矩阵把物体系坐标映射到相机坐标（相机 Y 向下、Z 向前）。
NOMINAL_R = np.diag([1.0, -1.0, -1.0])

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Calibration:
    path: str
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    cols: int
    rows: int
    square_m: float
    calib_rmse: float = 0.0


@dataclass
class PoseSolution:
    rvec: np.ndarray
    tvec: np.ndarray
    reproj_rmse: float


@dataclass
class PoseResult:
    ok: bool
    corners: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    roll: float = float("nan")
    pitch: float = float("nan")
    yaw: float = float("nan")
    distance: float = float("nan")
    reproj_rmse: float = float("nan")
    solution_errors: Tuple[float, ...] = ()
    num_solutions: int = 0
    detector: str = "none"
    flipped: bool = False
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2-2 单目视觉位姿估计（棋盘格 + solvePnP）"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="摄像头索引，例如 0")
    source.add_argument("--video", help="视频文件路径")
    source.add_argument(
        "--image",
        nargs="+",
        help="一张或多张图片，也可以给目录（自动读取 jpg/png）",
    )

    parser.add_argument(
        "--calib",
        default=DEFAULT_CALIB,
        help="M2-1 的 camera_calibration.yaml（默认 ../M2-1/camera_calibration.yaml）",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=None,
        help="棋盘格实测边长（mm），默认从标定 YAML 读取；只有重新量过才需要覆盖",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="图片模式下显示窗口；摄像头/视频默认显示",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开窗口，只处理/保存（摄像头模式会一直处理到 Ctrl+C）",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="结果图目录，默认脚本同级 results/",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="可选：把每帧位姿写入 CSV（便于精度实验）",
    )
    parser.add_argument(
        "--no-distort",
        action="store_true",
        help="解 PnP 和画轴时使用零畸变（用于对比实验）",
    )
    parser.add_argument(
        "--compare-distort",
        action="store_true",
        help="图片模式：同一帧分别用带/不带畸变解 PnP，打印差值",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="不做 solvePnPRefineLM 精化",
    )
    parser.add_argument(
        "--corner-diag",
        choices=("dark", "light"),
        default="dark",
        help="首角点主对角线两侧方块应为深色还是浅色；默认 dark，对应 M2-1 的棋盘格",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="摄像头/视频模式每帧打印位姿",
    )
    return parser.parse_args()


def _fs_real(node: cv2.FileNode, key: str) -> float:
    child = node.getNode(key)
    if child.empty():
        raise KeyError(key)
    return float(child.real())


def _fs_ints(node: cv2.FileNode, key: str) -> List[int]:
    child = node.getNode(key)
    if child.empty():
        raise KeyError(key)
    return [int(round(child.at(i).real())) for i in range(child.size())]


def load_calibration(path: str, square_mm_override: Optional[float]) -> Calibration:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到标定文件: {path}")

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法读取标定 YAML: {path}")

    try:
        camera = fs.getNode("camera")
        if camera.empty():
            raise ValueError("YAML 缺少 camera 节点")

        intr = camera.getNode("intrinsics")
        fx = _fs_real(intr, "fx")
        fy = _fs_real(intr, "fy")
        cx = _fs_real(intr, "cx")
        cy = _fs_real(intr, "cy")

        dist_node = camera.getNode("distortion")
        dist = np.array(
            [_fs_real(dist_node, k) for k in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64,
        )

        width, height = _fs_ints(camera, "resolution")
        board = camera.getNode("calib_board")
        inner = _fs_ints(board, "inner_corners")
        if len(inner) != 2:
            raise ValueError("calib_board.inner_corners 必须是 [cols, rows]")
        cols, rows = int(inner[0]), int(inner[1])
        square_mm = _fs_real(board, "square_size_mm")
        if square_mm_override is not None:
            if abs(square_mm_override - square_mm) > 0.5:
                print(
                    f"[warn] CLI 棋盘格边长 {square_mm_override:.2f} mm "
                    f"与 YAML 的 {square_mm:.2f} mm 相差较大，确认量的是同一块板"
                )
            square_mm = float(square_mm_override)

        try:
            calib_rmse = _fs_real(camera, "reprojection_error")
        except KeyError:
            calib_rmse = 0.0
    finally:
        fs.release()

    if cols < 2 or rows < 2:
        raise ValueError(f"棋盘格内角点数异常: {cols} x {rows}")
    if square_mm <= 0:
        raise ValueError(f"棋盘格边长异常: {square_mm} mm")

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    calib = Calibration(
        path=path,
        K=K,
        dist=dist,
        width=width,
        height=height,
        cols=cols,
        rows=rows,
        square_m=square_mm / 1000.0,
        calib_rmse=calib_rmse,
    )
    rmse_text = f"{calib_rmse:.3f} px" if calib_rmse > 0 else "unknown"
    print(f"[calib] {path}")
    print(
        f"        {width}x{height}  fx={fx:.2f} fy={fy:.2f} "
        f"cx={cx:.2f} cy={cy:.2f}"
    )
    print(
        f"        board: (cols,rows)=({cols},{rows})  内角点 {cols}x{rows}  "
        f"square={square_mm:.3f} mm  calib_rmse={rmse_text}"
    )
    return calib


def scale_calibration(calib: Calibration, width: int, height: int) -> Calibration:
    if width == calib.width and height == calib.height:
        return calib

    sx = width / float(calib.width)
    sy = height / float(calib.height)
    K = calib.K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    print(
        f"[calib] 输入分辨率 {width}x{height} 与标定 {calib.width}x{calib.height} 不同，"
        f"按 sx={sx:.4f}, sy={sy:.4f} 缩放内参"
    )
    if abs(sx - sy) > 0.01 * max(sx, sy):
        print("[warn] 宽高比不一致，内参缩放只是近似，建议用标定分辨率采集")
    return Calibration(
        path=calib.path,
        K=K,
        dist=calib.dist.copy(),
        width=width,
        height=height,
        cols=calib.cols,
        rows=calib.rows,
        square_m=calib.square_m,
        calib_rmse=calib.calib_rmse,
    )


def make_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """
    物体系：原点在棋盘几何中心；X 沿长边向右；Y 沿短边向上；Z 垂直板面指向相机。
    M2-1 的 YAML 写的是 (cols, rows) = (6, 9)：cols 沿短边、rows 沿长边。
    检测器返回的角点按 k = j*cols + i 排列：i 沿短边（下→上），j 沿长边（左→右）。
    """
    n = cols * rows
    k = np.arange(n)
    i = k % cols
    j = k // cols

    objp = np.zeros((n, 3), dtype=np.float32)
    objp[:, 0] = (j - (rows - 1) / 2.0) * square_m
    objp[:, 1] = (i - (cols - 1) / 2.0) * square_m
    return objp


def _classic_detect(
    gray: np.ndarray, pattern: Tuple[int, int], scale: float
) -> Optional[np.ndarray]:
    if scale == 1.0:
        work = gray
    else:
        work = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(work, pattern, flags=flags)
    if not ok or corners is None:
        return None

    if scale != 1.0:
        corners = corners.reshape(-1, 2) / scale
        corners = cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2).astype(np.float32),
            (11, 11),
            (-1, -1),
            CORNER_SUBPIX_CRITERIA,
        )
    else:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), CORNER_SUBPIX_CRITERIA
        )
    return corners.reshape(-1, 2)


def _sb_detect(
    gray: np.ndarray, pattern: Tuple[int, int], scale: float
) -> Optional[np.ndarray]:
    if scale == 1.0:
        small = gray
    else:
        small = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, corners = cv2.findChessboardCornersSB(small, pattern)
    if not ok or corners is None:
        return None

    corners = corners.reshape(-1, 2)
    if scale != 1.0:
        corners = corners / scale
        corners = cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2).astype(np.float32),
            (11, 11),
            (-1, -1),
            CORNER_SUBPIX_CRITERIA,
        ).reshape(-1, 2)
    return corners


def detect_corners(
    gray: np.ndarray,
    cols: int,
    rows: int,
    full_classic_fallback: bool,
    full_sb_fallback: bool,
) -> Tuple[Optional[np.ndarray], str]:
    """
    SB 在 0.4 倍分辨率上最稳也够快（1080p 约 30 ms），角点再回全分辨率
    做 cornerSubPix；失败才退回经典检测 / 半分辨率 SB。
    摄像头和视频只做快路径与半分辨率兜底；图片模式允许全分辨率兜底。
    """
    pattern = (cols, rows)
    expected = cols * rows

    corners = _sb_detect(gray, pattern, SB_PRIMARY_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"SB@{SB_PRIMARY_SCALE:g}"

    corners = _classic_detect(gray, pattern, CLASSIC_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"classic@{CLASSIC_SCALE:g}"

    if full_classic_fallback:
        corners = _classic_detect(gray, pattern, 1.0)
        if corners is not None and len(corners) == expected:
            return corners, "classic"

    corners = _sb_detect(gray, pattern, SB_FALLBACK_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"SB@{SB_FALLBACK_SCALE:g}"

    if full_sb_fallback:
        corners = _sb_detect(gray, pattern, 1.0)
        if corners is not None and len(corners) == expected:
            return corners, "SB"

    return None, "none"


def canonicalize_corners(
    gray: np.ndarray,
    corners: np.ndarray,
    cols: int,
    first_diag_dark: bool,
) -> Tuple[np.ndarray, bool]:
    """
    棋盘格是 180° 对称的，检测器可能从相反的角开始返回。
    看首角点周围四个方块：若“主对角线”两侧的亮度关系与约定不符，就把整个
    角点数组倒序。这样 P0 始终跟随同一块物理角点，经典检测器和 SB 也能统一。
    """
    c = corners.reshape(-1, 2)
    if len(c) < cols + 1:
        return c, False

    u = c[1] - c[0]
    v = c[cols] - c[0]
    su = float(np.linalg.norm(u))
    sv = float(np.linalg.norm(v))
    if su < 1.0 or sv < 1.0:
        return c, False
    u = u / su
    v = v / sv

    values: Dict[Tuple[int, int], float] = {}
    h, w = gray.shape[:2]
    for a in (-1, 1):
        for b in (-1, 1):
            p = c[0] + 0.45 * su * a * u + 0.45 * sv * b * v
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
                return c, False
            patch = gray[y - 2 : y + 3, x - 2 : x + 3]
            values[(a, b)] = float(np.median(patch))

    main = values[(-1, -1)] + values[(1, 1)]
    anti = values[(1, -1)] + values[(-1, 1)]
    if abs(main - anti) < SIGNATURE_MIN_CONTRAST:
        return c, False

    main_is_dark = main < anti
    if main_is_dark != first_diag_dark:
        return c[::-1].copy(), True
    return c, False


def reprojection_rmse(
    objp: np.ndarray,
    imgp: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    diff = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def solve_pose(
    objp: np.ndarray,
    imgp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    refine: bool,
) -> Tuple[PoseSolution, List[PoseSolution]]:
    """
    平面靶标用 IPPE：solvePnPGeneric 会返回两个解，用重投影 RMSE 选小的。
    再用 solvePnPRefineLM 做一次 LM 精化（只有误差不变大才采用）。
    """
    pts = imgp.reshape(-1, 1, 2).astype(np.float64)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objp, pts, K, dist, flags=cv2.SOLVEPNP_IPPE
    )

    solutions: List[PoseSolution] = []
    if n and rvecs is not None and tvecs is not None:
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            solutions.append(
                PoseSolution(
                    rvec=rvec,
                    tvec=tvec,
                    reproj_rmse=reprojection_rmse(objp, imgp, rvec, tvec, K, dist),
                )
            )

    if not solutions:
        ok, rvec, tvec = cv2.solvePnP(
            objp, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP 失败")
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        solutions = [
            PoseSolution(
                rvec=rvec,
                tvec=tvec,
                reproj_rmse=reprojection_rmse(objp, imgp, rvec, tvec, K, dist),
            )
        ]

    best = min(solutions, key=lambda s: s.reproj_rmse)
    if refine:
        try:
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                objp, pts, K, dist, best.rvec.copy(), best.tvec.copy()
            )
            rvec_ref = np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1)
            tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)
            rmse_ref = reprojection_rmse(
                objp, imgp, rvec_ref, tvec_ref, K, dist
            )
            if rmse_ref < best.reproj_rmse:
                best = PoseSolution(rvec_ref, tvec_ref, rmse_ref)
        except cv2.error:
            pass

    return best, solutions


def rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    rvec -> R（物体系到相机）-> 相对“正对相机”姿态的偏差 Rdev = R * Rnom^T。
    分解顺序：Rdev = Ry(yaw) * Rx(pitch) * Rz(roll)
    即内旋 Y-X-Z（等价外旋 Z-X-Y）。
    """
    R, _ = cv2.Rodrigues(rvec)
    rdev = R @ NOMINAL_R.T

    sin_pitch = float(np.clip(-rdev[1, 2], -1.0, 1.0))
    pitch = float(np.degrees(np.arcsin(sin_pitch)))
    yaw = float(np.degrees(np.arctan2(rdev[0, 2], rdev[2, 2])))
    roll = float(np.degrees(np.arctan2(rdev[1, 0], rdev[1, 1])))
    return roll, pitch, yaw


def process_frame(
    frame: np.ndarray,
    calib: Calibration,
    use_distortion: bool,
    refine: bool,
    full_classic_fallback: bool,
    full_sb_fallback: bool,
    first_diag_dark: bool,
) -> PoseResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, detector = detect_corners(
        gray,
        calib.cols,
        calib.rows,
        full_classic_fallback,
        full_sb_fallback,
    )
    if corners is None:
        return PoseResult(ok=False, detector=detector, reason="未检出棋盘格")

    corners, flipped = canonicalize_corners(
        gray, corners, calib.cols, first_diag_dark
    )
    objp = make_object_points(calib.cols, calib.rows, calib.square_m)
    dist = calib.dist if use_distortion else np.zeros(5, dtype=np.float64)

    try:
        best, solutions = solve_pose(objp, corners, calib.K, dist, refine)
    except Exception as exc:  # noqa: BLE001 - 现场显示失败原因比直接崩掉好
        return PoseResult(
            ok=False,
            corners=corners,
            detector=detector,
            flipped=flipped,
            reason=f"PnP 失败: {exc}",
        )

    if best.reproj_rmse > MAX_REPROJ_PX:
        return PoseResult(
            ok=False,
            corners=corners,
            detector=detector,
            flipped=flipped,
            reproj_rmse=best.reproj_rmse,
            solution_errors=tuple(s.reproj_rmse for s in solutions),
            num_solutions=len(solutions),
            reason=f"重投影误差 {best.reproj_rmse:.1f} px 过大",
        )

    roll, pitch, yaw = rotation_to_euler(best.rvec)
    return PoseResult(
        ok=True,
        corners=corners,
        rvec=best.rvec,
        tvec=best.tvec,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        distance=float(np.linalg.norm(best.tvec)),
        reproj_rmse=best.reproj_rmse,
        solution_errors=tuple(s.reproj_rmse for s in solutions),
        num_solutions=len(solutions),
        detector=detector,
        flipped=flipped,
    )


def _font_scale(frame: np.ndarray) -> float:
    return 0.8 if frame.shape[1] >= 1280 else 0.55


def _draw_panel(
    frame: np.ndarray,
    lines: Sequence[Tuple[str, Tuple[int, int, int]]],
    origin: Tuple[int, int] = (20, 20),
) -> None:
    scale = _font_scale(frame)
    thickness = 2
    line_height = int(round(34 * scale / 0.8))
    sizes = [
        cv2.getTextSize(text, FONT, scale, thickness)[0] for text, _ in lines
    ]
    width = max(size[0] for size in sizes) + 20
    height = line_height * len(lines) + 10
    x0, y0 = origin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, (text, color) in enumerate(lines):
        y = y0 + line_height * (index + 1) - 6
        cv2.putText(
            frame, text, (x0 + 10, y), FONT, scale, color, thickness, cv2.LINE_AA
        )


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    scale = _font_scale(frame)
    text = f"FPS: {fps:.1f}"
    size = cv2.getTextSize(text, FONT, scale, 2)[0]
    x = frame.shape[1] - size[0] - 30
    y = 45
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x - 8, y - size[1] - 8), (x + size[0] + 8, y + 10), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 255, 0), 2, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    result: PoseResult,
    calib: Calibration,
    dist: np.ndarray,
    fps: Optional[float] = None,
) -> np.ndarray:
    vis = frame.copy()

    if result.corners is not None:
        pts = np.round(result.corners).astype(np.int32)
        for point in pts:
            cv2.circle(vis, tuple(point), 3, (0, 255, 0), -1, cv2.LINE_AA)
        p0 = tuple(pts[0])
        cv2.circle(vis, p0, 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "P0",
            (p0[0] + 8, p0[1] - 8),
            FONT,
            _font_scale(frame),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if result.ok and result.rvec is not None and result.tvec is not None:
        axis_len = 0.5 * max(1, calib.rows - 1) * calib.square_m
        cv2.drawFrameAxes(
            vis,
            calib.K,
            dist,
            result.rvec,
            result.tvec,
            float(axis_len),
            3,
        )
        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float32),
            result.rvec,
            result.tvec,
            calib.K,
            dist,
        )
        origin_px = tuple(np.round(origin.reshape(2)).astype(np.int32))
        cv2.circle(vis, origin_px, 4, (0, 255, 255), -1, cv2.LINE_AA)

        x, y, z = result.tvec.ravel()
        lines = [
            (
                f"[dist: {result.distance:.3f} m, yaw: {result.yaw:+.1f} deg, "
                f"pitch: {result.pitch:+.1f} deg]",
                (0, 255, 255),
            ),
            (
                f"[roll: {result.roll:+.1f} deg, x: {x:+.3f} m, "
                f"y: {y:+.3f} m, z: {z:+.3f} m]",
                (255, 255, 255),
            ),
            (
                f"[reproj: {result.reproj_rmse:.2f} px, "
                f"solutions: {result.num_solutions}, detector: {result.detector}]",
                (200, 200, 200),
            ),
        ]
        if result.reproj_rmse > REPROJ_WARN_PX:
            lines.append(
                (
                    f"[low confidence: reproj > {REPROJ_WARN_PX:.0f} px]",
                    (0, 0, 255),
                )
            )
    else:
        lines = [(f"[target not found: {result.reason}]", (0, 0, 255))]

    _draw_panel(vis, lines)
    if fps is not None:
        _draw_fps(vis, fps)
    return vis


def print_pose_result(name: str, result: PoseResult) -> None:
    if not result.ok:
        print(f"[{name}] 未得到位姿: {result.reason} (detector={result.detector})")
        return

    x, y, z = result.tvec.ravel()
    errors = ", ".join(f"{e:.2f}" for e in result.solution_errors)
    print(
        f"[{name}] detector={result.detector} flipped={result.flipped} "
        f"rmse={result.reproj_rmse:.2f}px solutions={result.num_solutions}"
        + (f" ({errors} px)" if errors else "")
    )
    print(
        f"  tvec [m] = ({x:+.4f}, {y:+.4f}, {z:+.4f})  "
        f"distance={result.distance:.4f} m"
    )
    print(
        f"  euler [deg] roll={result.roll:+.2f} pitch={result.pitch:+.2f} "
        f"yaw={result.yaw:+.2f}"
    )


def print_distortion_comparison(calib: Calibration, result: PoseResult) -> None:
    if not result.ok or result.corners is None:
        return

    objp = make_object_points(calib.cols, calib.rows, calib.square_m)
    zero = np.zeros(5, dtype=np.float64)
    try:
        on, _ = solve_pose(objp, result.corners, calib.K, calib.dist, True)
        off, _ = solve_pose(objp, result.corners, calib.K, zero, True)
    except Exception as exc:  # noqa: BLE001
        print(f"[distortion] 对比失败: {exc}")
        return

    roll_on, pitch_on, yaw_on = rotation_to_euler(on.rvec)
    roll_off, pitch_off, yaw_off = rotation_to_euler(off.rvec)
    delta_t = float(np.linalg.norm(on.tvec - off.tvec))

    def fmt(tvec: np.ndarray, roll: float, pitch: float, yaw: float) -> str:
        x, y, z = tvec.ravel()
        return (
            f"t=({x:+.4f},{y:+.4f},{z:+.4f}) m "
            f"euler=({roll:+.1f},{pitch:+.1f},{yaw:+.1f}) deg"
        )

    print("[distortion] same frame, IPPE + LM")
    print(f"  with    : {fmt(on.tvec, roll_on, pitch_on, yaw_on)}")
    print(f"  without : {fmt(off.tvec, roll_off, pitch_off, yaw_off)}")
    print(
        f"  delta   : |dt|={delta_t * 1000:.1f} mm, "
        f"droll={roll_on - roll_off:+.2f} deg, "
        f"dpitch={pitch_on - pitch_off:+.2f} deg, "
        f"dyaw={yaw_on - yaw_off:+.2f} deg"
    )


class CsvLogger:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._file = None
        self._writer = None
        if not path:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "source",
                "frame",
                "timestamp",
                "detected",
                "x_m",
                "y_m",
                "z_m",
                "distance_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "reproj_px",
                "num_solutions",
                "detector",
            ]
        )

    def log(self, source: str, frame_index: int, timestamp: float, result: PoseResult) -> None:
        if self._writer is None:
            return
        if result.ok and result.tvec is not None:
            x, y, z = result.tvec.ravel()
            row = [
                source,
                frame_index,
                f"{timestamp:.6f}",
                1,
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{result.distance:.6f}",
                f"{result.roll:.6f}",
                f"{result.pitch:.6f}",
                f"{result.yaw:.6f}",
                f"{result.reproj_rmse:.6f}",
                result.num_solutions,
                result.detector,
            ]
        else:
            row = [
                source,
                frame_index,
                f"{timestamp:.6f}",
                0,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                result.detector,
            ]
        self._writer.writerow(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def read_image(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str, image: np.ndarray) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buffer.tofile(path)


def expand_images(items: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for item in items:
        if os.path.isdir(item):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(glob.glob(os.path.join(item, ext)))
        else:
            paths.append(item)
    paths = sorted(dict.fromkeys(paths))
    return paths


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w.-]+", "_", stem)
    return stem or "image"


def run_images(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    paths = expand_images(args.image)
    if not paths:
        print("没有找到图片")
        return

    show = args.show and not args.headless
    if show:
        cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("M2-2 Pose", 1280, 720)

    for index, path in enumerate(paths):
        frame = read_image(path)
        if frame is None:
            print(f"跳过无法读取的图片: {path}")
            continue

        calib_i = scale_calibration(calib, frame.shape[1], frame.shape[0])
        result = process_frame(
            frame,
            calib_i,
            use_distortion=not args.no_distort,
            refine=not args.no_refine,
            full_classic_fallback=True,
            full_sb_fallback=True,
            first_diag_dark=(args.corner_diag == "dark"),
        )
        dist_used = (
            calib_i.dist
            if not args.no_distort
            else np.zeros(5, dtype=np.float64)
        )
        vis = annotate_frame(frame, result, calib_i, dist_used)

        out_path = os.path.join(
            args.save_dir, f"axis_{index:02d}_{safe_stem(path)}.png"
        )
        write_image(out_path, vis)
        print(f"[save] {out_path}")
        print_pose_result(path, result)

        if args.compare_distort:
            print_distortion_comparison(calib_i, result)

        logger.log(path, index, time.time(), result)

        if show:
            cv2.imshow("M2-2 Pose", vis)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break


def _open_window(frame: np.ndarray) -> None:
    cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)
    height, width = frame.shape[:2]
    if width > 1280:
        cv2.resizeWindow("M2-2 Pose", 1280, int(1280 * height / width))
    else:
        cv2.resizeWindow("M2-2 Pose", width, height)


def run_video(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[video] {args.video}  {frame_width}x{frame_height}"
        f"@{source_fps:.1f}fps"
    )
    calib_v = scale_calibration(calib, frame_width, frame_height)
    show = not args.headless

    if show:
        cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)

    fps_smooth: Optional[float] = None
    last_time = time.perf_counter()
    frame_index = 0
    window_ready = not show

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if show and not window_ready:
                _open_window(frame)
                window_ready = True

            result = process_frame(
                frame,
                calib_v,
                use_distortion=not args.no_distort,
                refine=not args.no_refine,
                full_classic_fallback=False,
                full_sb_fallback=False,
                first_diag_dark=(args.corner_diag == "dark"),
            )
            dist_used = (
                calib_v.dist
                if not args.no_distort
                else np.zeros(5, dtype=np.float64)
            )
            vis = annotate_frame(frame, result, calib_v, dist_used, fps=fps_smooth)

            now = time.perf_counter()
            instant = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_smooth = (
                instant if fps_smooth is None else 0.9 * fps_smooth + 0.1 * instant
            )

            logger.log(args.video, frame_index, time.time(), result)
            if args.verbose or (frame_index % 30 == 0):
                if result.ok:
                    print(
                        f"[video] frame={frame_index} fps={fps_smooth:.1f} "
                        f"dist={result.distance:.3f}m "
                        f"yaw={result.yaw:+.1f} pitch={result.pitch:+.1f} "
                        f"rmse={result.reproj_rmse:.2f}px det={result.detector}"
                    )
                else:
                    print(f"[video] frame={frame_index} {result.reason}")

            if show:
                cv2.imshow("M2-2 Pose", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = os.path.join(
                        args.save_dir,
                        time.strftime("axis_video_%Y%m%d_%H%M%S.png"),
                    )
                    write_image(path, vis)
                    print(f"[save] {path}")

            frame_index += 1
    finally:
        cap.release()


def run_camera(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return

    # MJPG + 标定分辨率，避免相机默认给出 640x480 导致内参不匹配
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib.height)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("[camera] 关闭自动对焦（内参是在固定焦距下标定的）")

    ok, frame = cap.read()
    if not ok or frame is None:
        print("摄像头读帧失败")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"[camera] {args.camera}  {frame_width}x{frame_height}"
        f"@{actual_fps:.1f}fps (target {CAPTURE_FPS})"
    )
    calib_cam = scale_calibration(calib, frame_width, frame_height)
    show = not args.headless

    if show:
        _open_window(frame)

    fps_smooth: Optional[float] = None
    last_time = time.perf_counter()
    frame_index = 0

    try:
        while True:
            result = process_frame(
                frame,
                calib_cam,
                use_distortion=not args.no_distort,
                refine=not args.no_refine,
                full_classic_fallback=False,
                full_sb_fallback=False,
                first_diag_dark=(args.corner_diag == "dark"),
            )
            dist_used = (
                calib_cam.dist
                if not args.no_distort
                else np.zeros(5, dtype=np.float64)
            )
            vis = annotate_frame(
                frame, result, calib_cam, dist_used, fps=fps_smooth
            )

            now = time.perf_counter()
            instant = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_smooth = (
                instant if fps_smooth is None else 0.9 * fps_smooth + 0.1 * instant
            )

            logger.log("camera", frame_index, time.time(), result)
            if args.verbose or (frame_index % 30 == 0):
                if result.ok:
                    print(
                        f"[camera] frame={frame_index} fps={fps_smooth:.1f} "
                        f"dist={result.distance:.3f}m "
                        f"yaw={result.yaw:+.1f} pitch={result.pitch:+.1f} "
                        f"rmse={result.reproj_rmse:.2f}px det={result.detector}"
                    )
                else:
                    print(f"[camera] frame={frame_index} {result.reason}")

            if show:
                cv2.imshow("M2-2 Pose", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = os.path.join(
                        args.save_dir,
                        time.strftime("axis_cam_%Y%m%d_%H%M%S.png"),
                    )
                    write_image(path, vis)
                    print(f"[save] {path}")

            ok, frame = cap.read()
            if not ok or frame is None:
                print("摄像头读帧失败，退出")
                break
            frame_index += 1
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    if args.compare_distort and not args.image:
        print("[warn] --compare-distort 只在 --image 模式生效，已忽略")
        args.compare_distort = False

    calib = load_calibration(args.calib, args.square_mm)
    logger = CsvLogger(args.csv)
    try:
        if args.image:
            run_images(args, calib, logger)
        elif args.video:
            run_video(args, calib, logger)
        else:
            run_camera(args, calib, logger)
    finally:
        logger.close()
        cv2.destroyAllWindows()
        if args.csv:
            print(f"[csv] {args.csv}")


if __name__ == "__main__":
    main()
