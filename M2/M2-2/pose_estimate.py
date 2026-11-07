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

# HERE：当前脚本所在目录，用来拼默认路径，保证从任何工作目录运行都能找到文件。
HERE = os.path.dirname(os.path.abspath(__file__))
# DEFAULT_CALIB：默认标定文件，指向同级 M2-1 的 camera_calibration.yaml。
DEFAULT_CALIB = os.path.normpath(
    os.path.join(HERE, "..", "M2-1", "camera_calibration.yaml")
)
# DEFAULT_SAVE_DIR：默认结果图目录，位于脚本同级的 results/。
DEFAULT_SAVE_DIR = os.path.join(HERE, "results")

# CAPTURE_FPS：摄像头采集目标帧率；处理帧率会高于它，留出余量。
CAPTURE_FPS = 15
# SB_PRIMARY_SCALE：SB 检测器首选缩放比；0.4 倍找大致位置，再回全分辨率精化。
SB_PRIMARY_SCALE = 0.4
# CLASSIC_SCALE：经典检测器兜底缩放比；0.6 倍兼顾速度和成功率。
CLASSIC_SCALE = 0.6
# SB_FALLBACK_SCALE：SB 第二次尝试的缩放比；比首选更慢但更容易检出。
SB_FALLBACK_SCALE = 0.5
# CORNER_SUBPIX_CRITERIA：cornerSubPix 的终止条件：最多 30 次迭代或精度 0.001 像素。
CORNER_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
# REPROJ_WARN_PX：重投影误差超过该值时，画面上加“低置信度”警告。
REPROJ_WARN_PX = 2.0
# MAX_REPROJ_PX：重投影误差超过该值时直接拒绝，不显示错误位姿。
MAX_REPROJ_PX = 3.0
# SIGNATURE_MIN_CONTRAST：P0 周围主/副对角线的最小亮度差；太小就放弃归一化。
SIGNATURE_MIN_CONTRAST = 20.0

# 物体系的“正对相机”姿态：物体系 X 向右、Y 向上、Z 指向相机时，
# 该矩阵把物体系坐标映射到相机坐标（相机 Y 向下、Z 向前）。
NOMINAL_R = np.diag([1.0, -1.0, -1.0])

# FONT：全程序统一使用的 OpenCV Hershey 字体（不含中文）。
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Calibration：一份标定参数，包含内参、畸变、分辨率、棋盘格规格和标定误差。
@dataclass
class Calibration:
    path: str                  # path：标定 YAML 的路径，便于打印和排查
    K: np.ndarray              # K：3x3 相机内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
    dist: np.ndarray           # dist：畸变系数 [k1,k2,p1,p2,k3]
    width: int                 # width：标定时的图像宽度（像素）
    height: int                # height：标定时的图像高度（像素）
    cols: int                  # cols：棋盘格短边方向的内角点数（YAML 里的第一个数）
    rows: int                  # rows：棋盘格长边方向的内角点数（YAML 里的第二个数）
    square_m: float            # square_m：方格边长，单位米
    calib_rmse: float = 0.0    # calib_rmse：M2-1 标定报告的重投影误差，仅用于显示


# PoseSolution：PnP 的一个候选解，包含旋转、平移和该解的重投影误差。
@dataclass
class PoseSolution:
    rvec: np.ndarray           # rvec：旋转向量（Rodrigues 形式），3x1
    tvec: np.ndarray           # tvec：物体系原点在相机坐标系下的平移，3x1，单位米
    reproj_rmse: float         # reproj_rmse：该解的重投影均方根误差，单位像素


# PoseResult：单帧处理的最终结果；ok=False 时只有 reason/detector 等诊断字段有效。
@dataclass
class PoseResult:
    ok: bool                                      # ok：是否得到可信位姿
    corners: Optional[np.ndarray] = None          # corners：归一化后的 54 个图像角点
    rvec: Optional[np.ndarray] = None             # rvec：选中的旋转向量
    tvec: Optional[np.ndarray] = None             # tvec：选中的平移向量，单位米
    roll: float = float("nan")                    # roll：绕光轴的横滚角，单位度
    pitch: float = float("nan")                   # pitch：绕相机 X 轴的俯仰角，单位度
    yaw: float = float("nan")                     # yaw：绕相机 Y 轴的偏航角，单位度
    distance: float = float("nan")                # distance：相机到棋盘中心的欧氏距离，单位米
    reproj_rmse: float = float("nan")             # reproj_rmse：选中解的重投影误差，单位像素
    solution_errors: Tuple[float, ...] = ()       # solution_errors：IPPE 每个候选解的误差
    num_solutions: int = 0                        # num_solutions：IPPE 返回的候选解数量
    detector: str = "none"                        # detector：实际使用的检测器名称
    flipped: bool = False                         # flipped：角点数组是否被倒序过
    reason: str = ""                              # reason：失败原因，成功时为空



# 解析命令行参数：--camera / --video / --image 三选一，其余是标定、保存和调试选项。
def parse_args() -> argparse.Namespace:
    # parser：命令行解析器，负责生成帮助信息和默认值。
    parser = argparse.ArgumentParser(
        description="M2-2 单目视觉位姿估计（棋盘格 + solvePnP）"
    )
    # source：互斥参数组，--camera/--video/--image 必须且只能选一个。
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


# 从 cv2.FileStorage 的节点里读取一个标量实数；节点不存在时抛 KeyError。
def _fs_real(node: cv2.FileNode, key: str) -> float:
    # child：key 对应的子节点；不存在时 empty() 为 True。
    child = node.getNode(key)
    if child.empty():
        raise KeyError(key)
    return float(child.real())


# 从 cv2.FileStorage 的节点里读取整数序列，例如 resolution 和 inner_corners。
def _fs_ints(node: cv2.FileNode, key: str) -> List[int]:
    # child：key 对应的序列节点，例如 resolution 或 inner_corners。
    child = node.getNode(key)
    if child.empty():
        raise KeyError(key)
    return [int(round(child.at(i).real())) for i in range(child.size())]


# 读取 M2-1 的 camera_calibration.yaml，构造 Calibration；不写死任何内参。
def load_calibration(path: str, square_mm_override: Optional[float]) -> Calibration:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到标定文件: {path}")

    # fs：OpenCV 的 FileStorage 对象，用只读方式打开 YAML。
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法读取标定 YAML: {path}")

    try:
        # camera：YAML 的 camera 根节点，下面挂 intrinsics/distortion/resolution 等。
        camera = fs.getNode("camera")
        if camera.empty():
            raise ValueError("YAML 缺少 camera 节点")

        # intr：内参节点，包含 fx/fy/cx/cy。
        intr = camera.getNode("intrinsics")
        fx = _fs_real(intr, "fx")   # fx：x 方向焦距，像素
        fy = _fs_real(intr, "fy")   # fy：y 方向焦距，像素
        cx = _fs_real(intr, "cx")   # cx：主点 x 坐标，像素
        cy = _fs_real(intr, "cy")   # cy：主点 y 坐标，像素

        # dist_node：畸变节点；dist 按 [k1,k2,p1,p2,k3] 顺序拼成一维数组。
        dist_node = camera.getNode("distortion")
        dist = np.array(
            [_fs_real(dist_node, k) for k in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64,
        )

        # width/height：标定图像分辨率，用于后续判断是否需要缩放内参。
        width, height = _fs_ints(camera, "resolution")
        # board：棋盘格节点，记录内角点数量和方格边长。
        board = camera.getNode("calib_board")
        # inner：内角点数量 [cols, rows]，和 OpenCV 的 (cols, rows) 顺序一致。
        inner = _fs_ints(board, "inner_corners")
        if len(inner) != 2:
            raise ValueError("calib_board.inner_corners 必须是 [cols, rows]")
        # cols：短边方向内角点数；rows：长边方向内角点数。
        cols, rows = int(inner[0]), int(inner[1])
        # square_mm：YAML 记录的方格边长，单位 mm。
        square_mm = _fs_real(board, "square_size_mm")
        if square_mm_override is not None:
            if abs(square_mm_override - square_mm) > 0.5:
                print(
                    f"[warn] CLI 棋盘格边长 {square_mm_override:.2f} mm "
                    f"与 YAML 的 {square_mm:.2f} mm 相差较大，确认量的是同一块板"
                )
            # 现场重新量过方格时，以命令行值为准。
            square_mm = float(square_mm_override)

        try:
            # calib_rmse：M2-1 标定时报告的重投影误差，只用于打印。
            calib_rmse = _fs_real(camera, "reprojection_error")
        except KeyError:
            calib_rmse = 0.0
    finally:
        fs.release()

    if cols < 2 or rows < 2:
        raise ValueError(f"棋盘格内角点数异常: {cols} x {rows}")
    if square_mm <= 0:
        raise ValueError(f"棋盘格边长异常: {square_mm} mm")

    # K：3x3 内参矩阵，按针孔模型 [[fx,0,cx],[0,fy,cy],[0,0,1]] 构造。
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    # calib：把 YAML 里的参数打包成 Calibration；square_m 由 mm 转成米。
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
    # rmse_text：控制台显示用的标定误差文本；YAML 没写时显示 unknown。
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


# 输入分辨率与标定分辨率不同时，按 sx/sy 缩放 fx、fy、cx、cy，返回新的 Calibration。
def scale_calibration(calib: Calibration, width: int, height: int) -> Calibration:
    if width == calib.width and height == calib.height:
        return calib

    # sx/sy：输入图相对标定图的横向、纵向缩放比。
    sx = width / float(calib.width)
    sy = height / float(calib.height)
    # K：复制一份内参再缩放，避免修改调用方传进来的 calib.K。
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


# 按物体系生成棋盘格全部内角点的 3D 坐标（米，z=0），顺序必须和检测器返回的角点一一对应。
def make_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    # 物体系：原点在棋盘几何中心；X 沿长边向右；Y 沿短边向上；Z 垂直板面指向相机。
    # M2-1 的 YAML 写的是 (cols, rows) = (6, 9)：cols 沿短边、rows 沿长边。
    # 检测器返回的角点按 k = j*cols + i 排列：i 沿短边（下→上），j 沿长边（左→右）。
    # n：内角点总数，54。
    n = cols * rows
    # k：0..n-1 的线性下标；i 是组内下标（沿短边），j 是组号（沿长边）。
    k = np.arange(n)
    i = k % cols
    j = k // cols

    # objp：n 个 3D 物点，先全部置零，再填 X/Y。
    objp = np.zeros((n, 3), dtype=np.float32)
    # X 以棋盘中心为原点，沿长边展开；Y 沿短边展开；Z 保持 0。
    objp[:, 0] = (j - (rows - 1) / 2.0) * square_m
    objp[:, 1] = (i - (cols - 1) / 2.0) * square_m
    return objp


# 用经典 findChessboardCorners 在 scale 倍灰度图上找角点，再回原图做 cornerSubPix。
def _classic_detect(
    gray: np.ndarray, pattern: Tuple[int, int], scale: float
) -> Optional[np.ndarray]:
    # work：实际送进检测器的灰度图；scale=1 时直接复用原图，避免多余拷贝。
    if scale == 1.0:
        work = gray
    else:
        work = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    # flags：自适应阈值 + 图像归一化 + 快速预检，兼顾暗光/强光和速度。
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    # ok：是否检出；corners：检测器返回的角点，形状 (N,1,2)。
    ok, corners = cv2.findChessboardCorners(work, pattern, flags=flags)
    if not ok or corners is None:
        return None

    if scale != 1.0:
        # 先把缩放图上的角点除回原图坐标，再在原图灰度上做亚像素精化。
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


# 用 findChessboardCornersSB 在 scale 倍灰度图上找角点，再回原图做 cornerSubPix。
def _sb_detect(
    gray: np.ndarray, pattern: Tuple[int, int], scale: float
) -> Optional[np.ndarray]:
    # small：SB 检测用的缩放图；scale=1 时直接复用原图。
    if scale == 1.0:
        small = gray
    else:
        small = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    # ok：是否检出；corners：SB 返回的角点，形状 (N,1,2)。
    ok, corners = cv2.findChessboardCornersSB(small, pattern)
    if not ok or corners is None:
        return None

    # 统一拉平成 (N,2)，后续 PnP 都用这个形状。
    corners = corners.reshape(-1, 2)
    if scale != 1.0:
        # 缩放图角点除回原图坐标，再在原图上做亚像素精化。
        corners = corners / scale
        corners = cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2).astype(np.float32),
            (11, 11),
            (-1, -1),
            CORNER_SUBPIX_CRITERIA,
        ).reshape(-1, 2)
    return corners


# 按 SB@0.4 -> classic@0.6 -> 可选全分辨率经典 -> SB@0.5 -> 可选全分辨率 SB 的顺序尝试。
def detect_corners(
    gray: np.ndarray,
    cols: int,
    rows: int,
    full_classic_fallback: bool,
    full_sb_fallback: bool,
) -> Tuple[Optional[np.ndarray], str]:
    # SB 在 0.4 倍分辨率上最稳也够快（1080p 约 30 ms），角点再回全分辨率
    # 做 cornerSubPix；失败才退回经典检测 / 半分辨率 SB。
    # 摄像头和视频只做快路径与半分辨率兜底；图片模式允许全分辨率兜底。
    # pattern：传给 OpenCV 的 (cols, rows) 棋盘格尺寸。
    pattern = (cols, rows)
    # expected：应有的内角点总数，用来防止检测器返回半块棋盘。
    expected = cols * rows

    # 第一次尝试：SB 在 0.4 倍图上检测，速度最快。
    corners = _sb_detect(gray, pattern, SB_PRIMARY_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"SB@{SB_PRIMARY_SCALE:g}"

    # 第二次尝试：经典检测器 0.6 倍，SB 失败时更稳。
    corners = _classic_detect(gray, pattern, CLASSIC_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"classic@{CLASSIC_SCALE:g}"

    if full_classic_fallback:
        # 图片模式允许更慢的全分辨率经典检测。
        corners = _classic_detect(gray, pattern, 1.0)
        if corners is not None and len(corners) == expected:
            return corners, "classic"

    # 第三次尝试：SB 半分辨率，比 0.4 倍更容易检出。
    corners = _sb_detect(gray, pattern, SB_FALLBACK_SCALE)
    if corners is not None and len(corners) == expected:
        return corners, f"SB@{SB_FALLBACK_SCALE:g}"

    if full_sb_fallback:
        # 图片模式最后兜底：全分辨率 SB，最慢但成功率最高。
        corners = _sb_detect(gray, pattern, 1.0)
        if corners is not None and len(corners) == expected:
            return corners, "SB"

    # 所有路径都失败：返回 None，detector 记为 none。
    return None, "none"


# 用 P0 周围四个方格的黑白签名统一角点顺序，避免 SB 从对角返回导致 180° 翻转。
def canonicalize_corners(
    gray: np.ndarray,
    corners: np.ndarray,
    cols: int,
    first_diag_dark: bool,
) -> Tuple[np.ndarray, bool]:
    # 棋盘格是 180° 对称的，检测器可能从相反的角开始返回。
    # 看首角点周围四个方块：若“主对角线”两侧的亮度关系与约定不符，就把整个
    # 角点数组倒序。这样 P0 始终跟随同一块物理角点，经典检测器和 SB 也能统一。
    # c：拉平后的角点数组，形状 (N,2)，方便按下标取点。
    c = corners.reshape(-1, 2)
    if len(c) < cols + 1:
        return c, False

    # u：从 P0 指向 P1 的单位方向（沿短边）；v：从 P0 指向 P[cols] 的单位方向（沿长边）。
    u = c[1] - c[0]
    v = c[cols] - c[0]
    # su/sv：两个方向的像素间距，用来决定采样偏移和判断角点是否太密。
    su = float(np.linalg.norm(u))
    sv = float(np.linalg.norm(v))
    if su < 1.0 or sv < 1.0:
        return c, False
    u = u / su
    v = v / sv

    # values：四个对角采样点的亮度，键 (a,b) 表示在 u/v 方向上的正负号。
    values: Dict[Tuple[int, int], float] = {}
    # h/w：图像高宽，用于防止采样点越界。
    h, w = gray.shape[:2]
    for a in (-1, 1):
        for b in (-1, 1):
            # p：P0 周围 0.45 个方格处的采样点，避开角点本身的灰度过渡区。
            p = c[0] + 0.45 * su * a * u + 0.45 * sv * b * v
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
                return c, False
            # patch：5x5 邻域，取中位数比单像素更抗噪声。
            patch = gray[y - 2 : y + 3, x - 2 : x + 3]
            values[(a, b)] = float(np.median(patch))

    # main/anti：主对角线、副对角线两个方格的亮度和。
    main = values[(-1, -1)] + values[(1, 1)]
    anti = values[(1, -1)] + values[(-1, 1)]
    if abs(main - anti) < SIGNATURE_MIN_CONTRAST:
        return c, False

    # main_is_dark：主对角线是否更暗；和约定不符说明检测器从对角开始，需要倒序。
    main_is_dark = main < anti
    if main_is_dark != first_diag_dark:
        return c[::-1].copy(), True
    return c, False


# 把 3D 物点按当前 rvec/tvec 投影回图像，计算与检测角点的均方根误差（像素）。
def reprojection_rmse(
    objp: np.ndarray,
    imgp: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    # projected：3D 物点投影回图像的像素坐标；第二返回值是雅可比，这里不用。
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    # diff：每个点的投影误差向量 (du,dv)。
    diff = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
    # 先对每个点求平方和，再取平均、开方，得到像素单位的 RMSE。
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# 平面靶标用 IPPE 求解，得到两个候选解；按重投影 RMSE 选优，再用 LM 精化。
def solve_pose(
    objp: np.ndarray,
    imgp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    refine: bool,
) -> Tuple[PoseSolution, List[PoseSolution]]:
    # 平面靶标用 IPPE：solvePnPGeneric 会返回两个解，用重投影 RMSE 选小的。
    # 再用 solvePnPRefineLM 做一次 LM 精化（只有误差不变大才采用）。
    # pts：把角点整理成 solvePnP 要求的 (N,1,2) 双精度数组。
    pts = imgp.reshape(-1, 1, 2).astype(np.float64)
    # n：IPPE 返回的解数量（平面靶标通常为 2）；rvecs/tvecs：对应的旋转和平移列表。
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objp, pts, K, dist, flags=cv2.SOLVEPNP_IPPE
    )

    # solutions：把每个候选解连同它的重投影误差一起存起来，便于比较和打印。
    solutions: List[PoseSolution] = []
    if n and rvecs is not None and tvecs is not None:
        for rvec, tvec in zip(rvecs, tvecs):
            # 每个解都统一成 (3,1) 列向量，避免后续广播出错。
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
        # IPPE 完全失败时退回迭代法；它需要初值，这里让 OpenCV 自己估计。
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

    # best：重投影误差最小的候选解，就是最终采用的位姿。
    best = min(solutions, key=lambda s: s.reproj_rmse)
    if refine:
        try:
            # rvec_ref/tvec_ref：LM 精化后的旋转和平移。
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                objp, pts, K, dist, best.rvec.copy(), best.tvec.copy()
            )
            rvec_ref = np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1)
            tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)
            # rmse_ref：精化后的重投影误差；只有变小才替换原解。
            rmse_ref = reprojection_rmse(
                objp, imgp, rvec_ref, tvec_ref, K, dist
            )
            if rmse_ref < best.reproj_rmse:
                best = PoseSolution(rvec_ref, tvec_ref, rmse_ref)
        except cv2.error:
            # 精化失败不致命，保留 IPPE 的原解。
            pass

    return best, solutions


# rvec -> R -> Rdev = R * Rnom^T，再按 Ry(yaw)Rx(pitch)Rz(roll) 分解出三个欧拉角。
def rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    # rvec -> R（物体系到相机）-> 相对“正对相机”姿态的偏差 Rdev = R * Rnom^T。
    # 分解顺序：Rdev = Ry(yaw) * Rx(pitch) * Rz(roll)
    # 即内旋 Y-X-Z（等价外旋 Z-X-Y）。
    # R：由旋转向量转换得到的 3x3 旋转矩阵（物体系 -> 相机）。
    R, _ = cv2.Rodrigues(rvec)
    # rdev：去掉“正对相机”基准姿态后的相对旋转，正对时为单位阵。
    rdev = R @ NOMINAL_R.T

    # sin_pitch：先夹到 [-1,1]，防止浮点误差让 arcsin 产生 NaN。
    sin_pitch = float(np.clip(-rdev[1, 2], -1.0, 1.0))
    # pitch：俯仰角，范围 [-90,90]。
    pitch = float(np.degrees(np.arcsin(sin_pitch)))
    # yaw：偏航角，atan2 自动处理象限。
    yaw = float(np.degrees(np.arctan2(rdev[0, 2], rdev[2, 2])))
    # roll：横滚角。
    roll = float(np.degrees(np.arctan2(rdev[1, 0], rdev[1, 1])))
    return roll, pitch, yaw


# 单帧完整流程：灰度化 -> 检测角点 -> 顺序归一化 -> PnP -> 误差检查 -> 欧拉角。
def process_frame(
    frame: np.ndarray,
    calib: Calibration,
    use_distortion: bool,
    refine: bool,
    full_classic_fallback: bool,
    full_sb_fallback: bool,
    first_diag_dark: bool,
) -> PoseResult:
    # gray：检测和亚像素精化都只用到灰度图。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners：检测到的角点；detector：实际命中的检测器名称，用于调试和 CSV。
    corners, detector = detect_corners(
        gray,
        calib.cols,
        calib.rows,
        full_classic_fallback,
        full_sb_fallback,
    )
    if corners is None:
        return PoseResult(ok=False, detector=detector, reason="未检出棋盘格")

    # flipped：角点数组是否因为黑白签名不符被整体倒序。
    corners, flipped = canonicalize_corners(
        gray, corners, calib.cols, first_diag_dark
    )
    # objp：与当前角点顺序严格对应的 54 个 3D 物点。
    objp = make_object_points(calib.cols, calib.rows, calib.square_m)
    # dist：本次 PnP 使用的畸变系数；--no-distort 时用全零。
    dist = calib.dist if use_distortion else np.zeros(5, dtype=np.float64)

    try:
        # best：最终采用的解；solutions：IPPE 的全部候选解。
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

    # roll/pitch/yaw：把选中的旋转向量分解成三个欧拉角。
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


# 按画面宽度选择字号：宽图用 0.8，小图用 0.55。
def _font_scale(frame: np.ndarray) -> float:
    return 0.8 if frame.shape[1] >= 1280 else 0.55


# 在左上角画半透明黑底信息面板，逐行写文字。
def _draw_panel(
    frame: np.ndarray,
    lines: Sequence[Tuple[str, Tuple[int, int, int]]],
    origin: Tuple[int, int] = (20, 20),
) -> None:
    # scale：字号；thickness：文字笔画粗细。
    scale = _font_scale(frame)
    thickness = 2
    # line_height：行高，随字号缩放，保证不同分辨率下间距一致。
    line_height = int(round(34 * scale / 0.8))
    # sizes：每行文字的像素宽高，用于计算面板总宽。
    sizes = [
        cv2.getTextSize(text, FONT, scale, thickness)[0] for text, _ in lines
    ]
    # width/height：面板的总宽和总高。
    width = max(size[0] for size in sizes) + 20
    height = line_height * len(lines) + 10
    # x0/y0：面板左上角坐标。
    x0, y0 = origin

    # overlay：先画在半透明副本上，再和原图混合，得到半透明黑底。
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, (text, color) in enumerate(lines):
        # y：当前行的基线纵坐标。
        y = y0 + line_height * (index + 1) - 6
        cv2.putText(
            frame, text, (x0 + 10, y), FONT, scale, color, thickness, cv2.LINE_AA
        )


# 在右上角画 FPS 数值。
def _draw_fps(frame: np.ndarray, fps: float) -> None:
    # scale：字号；text：要显示的 FPS 文本。
    scale = _font_scale(frame)
    text = f"FPS: {fps:.1f}"
    # size：文本像素宽高，用于把文字靠右对齐。
    size = cv2.getTextSize(text, FONT, scale, 2)[0]
    # x/y：文字左下角坐标，x 靠右留 30 像素边距，y 固定在顶部。
    x = frame.shape[1] - size[0] - 30
    y = 45
    # overlay：半透明黑底，避免白背景下绿字看不清。
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x - 8, y - size[1] - 8), (x + size[0] + 8, y + 10), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 255, 0), 2, cv2.LINE_AA)


# 在图像上画全部角点、P0、物体系坐标轴和信息面板，返回新图。
def annotate_frame(
    frame: np.ndarray,
    result: PoseResult,
    calib: Calibration,
    dist: np.ndarray,
    fps: Optional[float] = None,
) -> np.ndarray:
    # vis：在副本上画，避免修改原始帧。
    vis = frame.copy()

    if result.corners is not None:
        # pts：把浮点角点转成整数像素坐标，供 cv2.circle 使用。
        pts = np.round(result.corners).astype(np.int32)
        for point in pts:
            cv2.circle(vis, tuple(point), 3, (0, 255, 0), -1, cv2.LINE_AA)
        # p0：首角点，红色标出，方便检查角点顺序。
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
        # axis_len：坐标轴长度，取长边内角点跨度的一半，约 68 mm。
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
        # origin：物体系原点（棋盘中心）的投影像素坐标。
        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float32),
            result.rvec,
            result.tvec,
            calib.K,
            dist,
        )
        # origin_px：整数化的原点坐标，用黄色圆点标出。
        origin_px = tuple(np.round(origin.reshape(2)).astype(np.int32))
        cv2.circle(vis, origin_px, 4, (0, 255, 255), -1, cv2.LINE_AA)

        # x/y/z：平移向量的三个分量，单位米。
        x, y, z = result.tvec.ravel()
        # lines：信息面板的三行文字，每行是 (文本, BGR 颜色)。
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


# 把一帧的检测器、重投影误差、tvec、距离和欧拉角打印到控制台。
def print_pose_result(name: str, result: PoseResult) -> None:
    if not result.ok:
        print(f"[{name}] 未得到位姿: {result.reason} (detector={result.detector})")
        return

    # x/y/z：平移向量的三个分量；errors：每个候选解的误差文本。
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


# 同一帧分别用带畸变和零畸变解 PnP，打印 tvec/角度的差值。
def print_distortion_comparison(calib: Calibration, result: PoseResult) -> None:
    if not result.ok or result.corners is None:
        return

    # objp：物点；zero：全零畸变系数，代表“关掉畸变”的对照组。
    objp = make_object_points(calib.cols, calib.rows, calib.square_m)
    zero = np.zeros(5, dtype=np.float64)
    try:
        # on：带畸变的解；off：不带畸变的解；两者用同一组角点。
        on, _ = solve_pose(objp, result.corners, calib.K, calib.dist, True)
        off, _ = solve_pose(objp, result.corners, calib.K, zero, True)
    except Exception as exc:  # noqa: BLE001
        print(f"[distortion] 对比失败: {exc}")
        return

    # roll_on/pitch_on/yaw_on：带畸变解的角度；off 后缀同理。
    roll_on, pitch_on, yaw_on = rotation_to_euler(on.rvec)
    roll_off, pitch_off, yaw_off = rotation_to_euler(off.rvec)
    # delta_t：两种解的平移向量欧氏距离，单位米。
    delta_t = float(np.linalg.norm(on.tvec - off.tvec))

    # fmt：把一组 tvec + 欧拉角格式化成一行字符串。
    def fmt(tvec: np.ndarray, roll: float, pitch: float, yaw: float) -> str:
        # x/y/z：平移分量，格式化时统一保留四位小数。
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


# CSV 日志类：每帧写一行位姿，方便离线算均值、标准差和绘制曲线。
class CsvLogger:
    # 打开 CSV 文件并写表头；path 为空时不创建文件。
    def __init__(self, path: Optional[str]) -> None:
        # path：CSV 输出路径；self._file/_writer 是实际文件句柄和写入器。
        self.path = path
        self._file = None
        self._writer = None
        if not path:
            return
        # parent：CSV 所在目录，不存在就先创建。
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

    # 写入一帧结果；未检出时 detected=0，其余数值列留空。
    def log(self, source: str, frame_index: int, timestamp: float, result: PoseResult) -> None:
        if self._writer is None:
            return
        if result.ok and result.tvec is not None:
            # x/y/z：平移分量；row：按表头顺序拼出的一行数据。
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

    # 关闭 CSV 文件并清空文件句柄。
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


# 用 np.fromfile + imdecode 读图，兼容中文和空格路径。
def read_image(path: str) -> Optional[np.ndarray]:
    # data：把文件当字节流读进来，绕过 cv2.imread 对非 ASCII 路径的限制。
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# 用 imencode + tofile 写图，兼容中文和空格路径。
def write_image(path: str, image: np.ndarray) -> None:
    # parent：输出目录；ext：文件扩展名，决定编码格式。
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    # ok：编码是否成功；buffer：编码后的字节缓冲。
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buffer.tofile(path)


# 把目录展开成图片路径列表，去重并排序；非目录项原样保留。
def expand_images(items: Sequence[str]) -> List[str]:
    # paths：收集到的图片路径；item：命令行给的一项（目录或单张图）。
    paths: List[str] = []
    for item in items:
        if os.path.isdir(item):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(glob.glob(os.path.join(item, ext)))
        else:
            paths.append(item)
    # dict.fromkeys 去重，sorted 保证输出顺序稳定。
    paths = sorted(dict.fromkeys(paths))
    return paths


# 把文件名转成只含安全字符的 stem，供结果图命名使用。
def safe_stem(path: str) -> str:
    # stem：去掉目录和扩展名后的文件名；再把非字母数字/点/横线替换成下划线。
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w.-]+", "_", stem)
    return stem or "image"


# 图片模式：逐张处理、保存结果图、可选显示、写 CSV 和畸变对比。
def run_images(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    # paths：要处理的图片列表；show：是否弹窗显示（--headless 时不显示）。
    paths = expand_images(args.image)
    if not paths:
        print("没有找到图片")
        return

    # show：是否弹窗显示；--headless 时强制不显示。
    show = args.show and not args.headless
    if show:
        cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("M2-2 Pose", 1280, 720)

    # index：图片序号（用于结果图命名）；path：当前图片路径。
    for index, path in enumerate(paths):
        # frame：当前 BGR 图像；读失败就跳过。
        frame = read_image(path)
        if frame is None:
            print(f"跳过无法读取的图片: {path}")
            continue

        # calib_i：按当前图片分辨率缩放后的内参。
        calib_i = scale_calibration(calib, frame.shape[1], frame.shape[0])
        # result：当前帧的位姿结果。
        result = process_frame(
            frame,
            calib_i,
            use_distortion=not args.no_distort,
            refine=not args.no_refine,
            full_classic_fallback=True,
            full_sb_fallback=True,
            first_diag_dark=(args.corner_diag == "dark"),
        )
        # dist_used：画轴时实际使用的畸变系数，必须和 PnP 时保持一致。
        dist_used = (
            calib_i.dist
            if not args.no_distort
            else np.zeros(5, dtype=np.float64)
        )
        # vis：画好角点、坐标轴和信息面板的结果图。
        vis = annotate_frame(frame, result, calib_i, dist_used)

        # out_path：结果图路径，带序号和原图 stem，避免重名覆盖。
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
            # key：等待按键；q/Esc 提前结束批量处理。
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break


# 按画面宽高比创建预览窗口，宽图限制在 1280 像素。
def _open_window(frame: np.ndarray) -> None:
    cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)
    # height/width：输入帧的高宽，用来算预览窗口尺寸。
    height, width = frame.shape[:2]
    if width > 1280:
        cv2.resizeWindow("M2-2 Pose", 1280, int(1280 * height / width))
    else:
        cv2.resizeWindow("M2-2 Pose", width, height)


# 视频模式：逐帧处理、平滑统计 FPS、可选显示和按键保存。
def run_video(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    # cap：视频读取器；先按路径打开。
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    # source_fps：视频原始帧率；frame_width/frame_height：视频分辨率。
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[video] {args.video}  {frame_width}x{frame_height}"
        f"@{source_fps:.1f}fps"
    )
    # calib_v：按视频分辨率缩放后的内参；show：是否显示窗口。
    calib_v = scale_calibration(calib, frame_width, frame_height)
    # show：是否显示视频窗口；--headless 时只处理不显示。
    show = not args.headless

    if show:
        cv2.namedWindow("M2-2 Pose", cv2.WINDOW_NORMAL)

    # fps_smooth：指数平滑后的处理帧率；last_time：上一帧时间戳。
    fps_smooth: Optional[float] = None
    last_time = time.perf_counter()
    # frame_index：帧序号；window_ready：窗口是否已经按首帧尺寸创建。
    frame_index = 0
    window_ready = not show

    try:
        while True:
            # ok：是否读到帧；frame：当前 BGR 图像。
            ok, frame = cap.read()
            if not ok:
                break

            if show and not window_ready:
                _open_window(frame)
                window_ready = True

            # result：当前帧位姿；dist_used：画轴用的畸变系数。
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
            # vis：带标注的结果图。
            vis = annotate_frame(frame, result, calib_v, dist_used, fps=fps_smooth)

            # now：当前时间；instant：由两帧间隔算出的瞬时帧率。
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
                # key：q/Esc 退出，s 保存当前标注帧。
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


# 摄像头模式：配置 MJPG/分辨率/帧率，逐帧处理并显示。
def run_camera(
    args: argparse.Namespace,
    calib: Calibration,
    logger: CsvLogger,
) -> None:
    # cap：摄像头读取器；优先用 V4L2 后端，失败再退回默认后端。
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

    # ok：是否读到首帧；frame：首帧图像，用来确定实际分辨率。
    ok, frame = cap.read()
    if not ok or frame is None:
        print("摄像头读帧失败")
        cap.release()
        return

    # frame_height/frame_width：摄像头实际输出的高宽；actual_fps：实际帧率。
    frame_height, frame_width = frame.shape[:2]
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"[camera] {args.camera}  {frame_width}x{frame_height}"
        f"@{actual_fps:.1f}fps (target {CAPTURE_FPS})"
    )
    # calib_cam：按实际分辨率缩放后的内参；show：是否显示窗口。
    calib_cam = scale_calibration(calib, frame_width, frame_height)
    # show：是否显示摄像头窗口；--headless 时只处理不显示。
    show = not args.headless

    if show:
        _open_window(frame)

    # fps_smooth：平滑后的处理帧率；last_time：上一帧时间戳；frame_index：帧序号。
    fps_smooth: Optional[float] = None
    last_time = time.perf_counter()
    frame_index = 0

    try:
        while True:
            # result：当前帧位姿；dist_used：画轴用的畸变系数。
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
            # vis：带标注的结果图。
            vis = annotate_frame(
                frame, result, calib_cam, dist_used, fps=fps_smooth
            )

            # now：当前时间；instant：由两帧间隔算出的瞬时帧率。
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
                # key：q/Esc 退出，s 保存当前标注帧。
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


# 程序入口：解析参数、加载标定、创建 CSV 日志，再分发到图片/视频/摄像头模式。
def main() -> None:
    # args：命令行参数；calib：加载后的标定；logger：可选 CSV 日志。
    args = parse_args()
    if args.compare_distort and not args.image:
        print("[warn] --compare-distort 只在 --image 模式生效，已忽略")
        args.compare_distort = False

    # calib：从 M2-1 YAML 读到的标定参数；logger：按 --csv 创建日志器。
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
