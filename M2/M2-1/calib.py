import argparse
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(HERE, "calib_imgs")

BOARD_COLS = 6          # 横向角点数
BOARD_ROWS = 9          # 纵向角点数
SQUARE_SIZE_MM = 17.0   # 方格实际边长

RMSE_GOOD = 0.5         # 低于此值视为良好
RMSE_FAIL = 1.5         # 高于此值视为标定失败，需要重采

FX_FY_ERROR_RATIO = 0.05    # fx/fy 相对差超过 5% 视为异常，报错
K1_MIN = -0.3               # 普通 USB 相机 k1 的常见范围
K1_MAX = -0.05

# 命令行输入
def parse_args():
    parser = argparse.ArgumentParser(description="棋盘格相机标定")
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGE_DIR,
        help="标定图片目录，默认使用脚本同级的 calib_imgs",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "camera_calibration.yaml"),
        help="标定结果 YAML 输出路径",
    )
    return parser.parse_args()

#导入棋盘格图片
def load_images(image_dir):
    paths = []
    for ext in ("jpg", "jpeg", "png"):
        paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
    paths.sort()

    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"跳过无法读取的图片: {path}")
            continue
        images.append((os.path.basename(path), img))
    return images


#高斯滤波
def gaussian_blur(images, ksize=(3, 3), sigma=0):
    return [(name, cv2.GaussianBlur(img, ksize, sigma)) for name, img in images]


#确定三维坐标（棋盘平面，z=0）
def make_object_points():
    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE_MM
    return objp


#检测棋盘格角点（SB 自带亚像素，不需要 cornerSubPix）
def find_corners(images):
    pattern = (BOARD_COLS, BOARD_ROWS)
    found = []
    for name, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            found.append((name, corners))
        else:
            print(f"未检出棋盘格，跳过: {name}")
    return found


#相机标定（物点与像点一一对应，z=0 平面）
def calibrate_camera(found, objp, image_size):
    obj_points = [objp.copy() for _ in found]
    img_points = [corners.reshape(-1, 2) for _, corners in found]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


#内参合理性检查：fx≈fy；k1 符号与量级
def check_intrinsics(camera_matrix, dist_coeffs):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    rel = abs(fx - fy) / ((abs(fx) + abs(fy)) / 2)

    if rel > FX_FY_ERROR_RATIO:
        print(f"fx/fy: {fx:.1f} / {fy:.1f}，相对差 {rel * 100:.2f}%")
        print(f"  错误: fx 与 fy 相差过大（>{FX_FY_ERROR_RATIO * 100:.0f}%），"
              "说明角点检测或标定出了问题")
        fx_ok = False
    else:
        print(f"fx/fy: {fx:.1f} / {fy:.1f}，相对差 {rel * 100:.2f}% 正常")
        fx_ok = True

    k1 = dist_coeffs.ravel()[0]
    if K1_MIN <= k1 <= K1_MAX:
        print(f"k1: {k1:+.4f}，在常见范围内")
    else:
        print(f"k1: {k1:+.4f}，警告: 不在普通 USB 相机的常见范围"
              f"（{K1_MIN:.2f} ~ {K1_MAX:.2f}）")
        print("  可能原因: 棋盘纸不平、边缘样本不足，畸变被错误拟合")

    return fx_ok


#重投影误差报告
def reprojection_report(found, objp, camera_matrix, dist_coeffs, rvecs, tvecs):
    print("\n内参合理性检查:")
    fx_ok = check_intrinsics(camera_matrix, dist_coeffs)

    errors = []
    sq_sum = 0.0
    n_points = 0
    for (name, corners), rvec, tvec in zip(found, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            objp, rvec, tvec, camera_matrix, dist_coeffs
        )
        diff = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        sq = np.sum(diff ** 2, axis=1)
        errors.append((name, np.sqrt(np.mean(sq))))
        sq_sum += np.sum(sq)
        n_points += len(sq)

    overall = np.sqrt(sq_sum / n_points)
    print(f"\n重投影误差（RMSE）: {overall:.3f} px")
    print("各图最差 5 张:")
    for name, rmse in sorted(errors, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {rmse:.3f} px")

    rmse_ok = overall <= RMSE_FAIL
    if not rmse_ok:
        print(f"结果: 标定失败（>{RMSE_FAIL} px），需要重采")
        print("可能原因:")
        print("  - 棋盘纸不平（贴在硬纸板上、手持），z=0 假设不成立")
        print("  - 角点覆盖范围不足，边缘样本少，畸变被错误拟合")
        print("  - 拍摄时抖动/模糊，角点定位有系统偏差")
    elif overall > RMSE_GOOD:
        print(f"结果: 警告（>{RMSE_GOOD} px），可用但需说明原因")
        print("可能原因:")
        print("  - 棋盘纸有微小翘曲，物点 z=0 不完全成立")
        print("  - 部分图片位于画面边缘，畸变模型残差偏大")
        print("  - 手持拍摄的轻微运动模糊")
    else:
        print("结果: 良好")

    if not fx_ok:
        print("结论: 标定失败（fx/fy 相差过大）")
        return overall, False
    return overall, rmse_ok


#导出标定结果（YAML）
def save_calibration_yaml(path, camera_matrix, dist_coeffs, reprojection_error,
                          image_size, images_used):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()[:5]
    width, height = image_size

    text = f"""camera:
  model: pinhole
  resolution: [{width}, {height}]        # 标定时的分辨率（重要！内参与分辨率绑定）
  intrinsics:
    fx: {fx:.6f}
    fy: {fy:.6f}
    cx: {cx:.6f}
    cy: {cy:.6f}
  distortion:
    k1: {k1:.6f}
    k2: {k2:.6f}
    p1: {p1:.6f}
    p2: {p2:.6f}
    k3: {k3:.6f}
  reprojection_error: {reprojection_error:.6f}        # px
  calib_board:
    type: chessboard
    inner_corners: [{BOARD_COLS}, {BOARD_ROWS}]
    square_size_mm: {SQUARE_SIZE_MM}          # 卡尺量出来的实际值
  images_used: {images_used}
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n已导出标定结果: {path}")


def main():
    args = parse_args()
    images = load_images(args.images)
    print(f"从 {args.images} 读取到 {len(images)} 张图片")

    images = gaussian_blur(images)
    print(f"已对 {len(images)} 张图片做高斯滤波")

    objp = make_object_points()
    print(f"三维坐标: {objp.shape[0]} 个点（z=0），边长 {SQUARE_SIZE_MM} mm")
    print(f"  首点 {objp[0]}，末点 {objp[-1]}")

    found = find_corners(images)
    print(f"角点检测: {len(found)}/{len(images)} 张成功")
    if found:
        name, corners = found[0]
        print(f"  示例 {name}: {corners.shape[0]} 个角点，首点 {corners[0, 0]}")

    image_size = (images[0][1].shape[1], images[0][1].shape[0])
    _, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        found, objp, image_size
    )
    print("内参矩阵:")
    print(camera_matrix)
    print("畸变系数 [k1 k2 p1 p2 k3]:")
    print(dist_coeffs.ravel())

    overall, _ = reprojection_report(
        found, objp, camera_matrix, dist_coeffs, rvecs, tvecs
    )
    save_calibration_yaml(
        args.out, camera_matrix, dist_coeffs, overall, image_size, len(found)
    )


if __name__ == "__main__":
    main()
