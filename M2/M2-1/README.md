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
2. 空格确认开始，`q`/Esc 退出。
3. 检测窗口出现后：
   - `p`：把当前标注帧存为 `results/normal/annotated_XX.png`，编号接续已有文件；
   - `q`/Esc：退出。

