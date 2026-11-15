# M2-2 单目视觉位姿估计

棋盘格角点 + `solvePnP`，输出靶标相对相机的 6-DoF：位置 `(x, y, z)` 米、
姿态 `(roll, pitch, yaw)` 度，并在画面上画 X/Y/Z 三色坐标轴。

- 内参**只读** `../M2-1/camera_calibration.yaml`，不重新标定、不写死
- `--camera / --video / --image` 三种数据源共用同一套算法，结果可互相复算
- 现场可连拍，留下**未标注原始帧**和逐帧清单，任何一帧都能离线重算

> 本文件是「快速上手 + 排错」入口。原理、参数全表、flag 选型、失效情况看
> [`report.md`](report.md)；实测表格看 [`results/accuracy.md`](results/accuracy.md)；
> 靶标打印与量尺看 [`marker/README.md`](marker/README.md)。

---

## 快速开始

```bash
# 0) 依赖（WSL Ubuntu2204 通常已装系统包，可跳过）
pip install -r requirements.txt

# 1) 实时演示：摄像头 + 靶标
python3 pose_estimate.py --camera 0

# 2) 离线跑一批图，逐帧写 CSV
python3 pose_estimate.py --image ../M2-1/calib_imgs/ --headless \
    --csv results/pose_log.csv

# 3) 单张图看标注结果（--show 弹窗）
python3 pose_estimate.py --image xxx.png --show

# 4) 现场实拍一组：摆稳靶标、等 reproj 稳定后按空格连拍 5 张
python3 pose_estimate.py --camera 0 \
    --capture-group distance_030 --truth 0.30

# 5) 免按键：启动即自动拍完一组并退出
python3 pose_estimate.py --camera 0 \
    --capture-group distance_030 --truth 0.30 --capture-auto --headless
```

所有默认路径都相对**脚本所在目录**计算，从别的工作目录启动也能找到标定文件和
`results/`。中文路径已用 `np.fromfile + imdecode` / `imencode + tofile` 兼容。

## 目录结构

```text
M2-2/
├── pose_estimate.py          # 主程序（唯一代码文件）
├── report.md                 # 物体系 / PnP / 欧拉角 / 失效情况 / 必答问题
├── README.md                 # 本文件
├── requirements.txt          # 运行依赖
├── results/
│   ├── axis_*.png            # 坐标轴可视化
│   ├── accuracy.md           # 三组实测数据（现场填）
│   └── capture/<组名>/       # 现场实拍：raw_*.png + axis_*.png + capture.csv
└── marker/                   # 靶标 SVG、预览图、打印与物体系说明
```

## 常用参数

完整参数表见 `report.md` §0.1 / §0.2，这里只列最常改的几个：

| 参数 | 用途 |
| :-- | :-- |
| `--calib` | 指定 M2-1 的 YAML；不写就用 `../M2-1/camera_calibration.yaml` |
| `--square-mm` | 重新量过方格边长时覆盖 17.0 mm，**不要改 YAML 内参** |
| `--no-distort` / `--compare-distort` | 关畸变求解 / 同一帧带与不带畸变对比 |
| `--corner-diag dark\|light` | 首角点主对角线明暗；这块板用默认 `dark` |
| `--csv` | 整段视频流逐帧位姿日志 |
| `--capture-group` | **开关**：给了才启用现场拍照，并作为该组目录名 |
| `--truth` | 本组真值，只写进清单，不参与计算 |
| `--capture-burst` / `--capture-interval` | 连拍张数（默认 5）/ 两帧最小间隔（默认 0.3 s） |

按键：`q` / `Esc` 退出，`s` 存当前标注图，**空格或 `c`** 触发一组连拍。

## 现场实拍产物

```text
results/capture/<组名>/
├── raw_001.png ...   未标注原始帧（PNG 无损）：可 --image 重算 / --compare-distort
├── axis_001.png ...  带角点、P0、三色轴、信息面板的标注图
└── capture.csv       逐帧：真值、detected、失败原因、x/y/z、distance、
                      roll/pitch/yaw、reproj_px、num_solutions、detector、文件名
```

- 重跑同组会**自动接着编号**（`raw_006.png` 起），绝不覆盖现场旧照片。
- 检出失败也照实保存并记 `detected=0` + `reason`，检出率能如实统计。
- 离线复算示例：
  `python3 pose_estimate.py --image results/capture/<组名>/raw_001.png --compare-distort --headless`
- 拍照只在 `--camera` / `--video` 下生效；`--image` 模式给 `--capture-group` 会忽略。

## 开跑前 30 秒检查

1. 首行日志的分辨率是 `1920x1080`；不是就先查相机格式（内参与分辨率绑定）。
2. 靶标贴平硬纸板、**标题朝上、长边水平**，占画面约 1/3–2/3。
3. 画面第三行 `reproj` 稳定在 1–2 px 以内再按空格；经常 > 2 px 先查光照/倾斜/对焦。
4. 自动对焦已关（日志里有提示）；距离 0.3–1.2 m，别接近侧视。
5. 每换一档摆放换一个 `--capture-group`，`--truth` 填这一档量到的实际值。

## 常见问题

| 现象 | 原因 / 处理 |
| :-- | :-- |
| 启动报 `Input file is invalid` / `无法读取标定 YAML` | M2-1 的 YAML 缺 `%YAML:1.0` 头；代码已自动补头到临时文件再读。若仍失败，确认文件存在、UTF-8 编码、没被改动 |
| WSL / SSH 下窗口打不开 | 加 `--headless`；要拍照就再加 `--capture-auto`，不需要按键 |
| 按空格没反应 | 没给 `--capture-group`；或窗口没焦点（点一下画面）；或用了 `--headless` 却没开自动 |
| `reproj` 一直很大 / 显示未找到 | 光照过曝或阴影、靶标翘曲、距离太远、接近侧视；见 `report.md` §9 |
| `roll` 接近 ±90° 而不是 0 | 靶标被**竖放**（长边竖直）：物体系 X 跟着数组方向转了 90°，这是摆放的真实反映，`reproj` 仍正常、不是角点错乱。让**标题朝上、长边水平**重摆，别去改零位。见 `report.md` §3.2 |
| 距离整体按固定比例偏 | 方格边长量错（打印缩放）。量 100 mm 标尺，用 `--square-mm <实测值>`，不要改内参 |
| 出现内参缩放警告 | 输入分辨率不是 1920×1080；缩放只是近似，尽量用标定分辨率 |
| 想看逐帧数据而不是平均值 | `results/capture/<组名>/capture.csv`（拍照帧）或 `--csv` 指定的文件（全帧流） |
| 图片路径带中文读不了 | 代码已兼容；若仍报错，确认文件本身没损坏 |

## 与题目的对应

题目要求的「不能只报平均误差不给逐点数据」由 `capture.csv` / `--csv` 的逐帧记录
满足；「物体系与代码里的 3D 点定义一致」见 `report.md` §2 与
`make_object_points`；三组精度实测表格留在 `results/accuracy.md`，由现场实测填写。
