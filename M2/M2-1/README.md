# M2-1 红绿灯检测

## 运行方式

### 标定（独立步骤）

```bash
python3 calib.py --images calib_imgs/ --out camera_calibration.yaml
```

`detect.py` 不使用标定文件，没有 `--calib` 参数。

### 检测：单图 / 视频 / 摄像头 三选一

```bash
# 单图：处理后在 results/normal/ 下直接生成 <原文件名>_annotated.png
python3 detect.py --image xxx.png
python3 detect.py --image xxx.png --show        # 额外弹窗预览，按任意键退出

# 视频：处理后直接写入 results/normal/<原文件名>_annotated.mp4
python3 detect.py --video xxx.mp4
python3 detect.py --video xxx.mp4 --show        # 弹窗预览，q/Esc 提前结束

# 摄像头
python3 detect.py --camera 0
```

摄像头模式始终显示窗口（需要按空格保存）；`--show` 只影响图片/视频。

## 摄像头模式操作

1. 先出现 `exposure` 窗口，提示 `LAMP BOX OUT OF VIEW`：让灯箱保持在画面外。
2. `r` 重新采样 L，空格确认开始，`q`/Esc 退出。
3. 检测窗口出现后：
   - 空格：把当前标注帧存为 `results/normal/annotated_XX.png`，编号接续已有文件；
   - `q`/Esc：退出。

## 输出说明

- 图上：每个灯画十字架（交点是圆心，臂长是半径）、范围圆、`颜色 ON/OFF r=半径` 文字；灯箱画矩形。
- 控制台：每行输出三个灯的状态、圆心、半径、判据值/阈值和像素数；默认只在三灯状态变化时打印。
- 结果目录：`results/normal/`。
  - 摄像头：`annotated_XX.png`，编号不覆盖已有文件。
  - 图片/视频：`<原文件名>_annotated.png/.mp4`，同名文件会被覆盖。

## 注意

- 图片/视频用第一帧初始化 L。如果第一帧里灯箱已经入画，单张图片无法修正背景光估计；视频会在后续带灯箱 mask 的更新中通过扩散修正。
- 摄像头固定 MJPG、1920×1080、15 fps（USB/IP 下 1080p 只能跑到 15 fps）。
