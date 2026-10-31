# M2-1 红绿灯检测

## 运行方式

```bash
# 标定
python3 calib.py --images calib_imgs/ --out camera_calibration.yaml

# 检测：单图 / 视频 / 摄像头 三选一
python3 detect.py --image xxx.png --calib camera_calibration.yaml --show
python3 detect.py --video xxx.mp4 --calib camera_calibration.yaml
python3 detect.py --camera 0 --calib camera_calibration.yaml --show
```
