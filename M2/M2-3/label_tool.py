#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""M2-3 矩形框标注工具（单文件，界面基于 tkinter，无需额外安装图形库）。

用途
----
给自己采集的照片画「刚好贴住物体上下左右边缘」的矩形框，坐标是**原图像素**
且保留小数（亚像素），结果以 numpy 数组的形式序列化进一个 JSON 文件。

快速开始
--------
    python3 label_tool.py                      # 启动即打开 M2-3/data/raw（有图时）
    python3 label_tool.py --folder /path/to/photos
    python3 label_tool.py --folder 'C:\Users\me\Desktop\照片'
    python3 label_tool.py --font-size 18       # 界面字号调大（默认 14pt）
    python3 label_tool.py --self-test          # 不弹界面，自检核心逻辑

本机 WSL 目录（例如 M2-3/data/raw）
-----------------------------------
Windows 的原生目录选择框只能在「此电脑」的盘符树里走，**进不去 \\wsl$**，
所以 WSL 本机目录有专门的三条路：

    1) 不带参数启动：只要 data/raw 里有图，直接自动载入，不弹任何框；
    2) 「本机目录…」按钮：Tk 原生对话框，起点就是 M2-3，可一路点进 data/raw；
    3) 顶部输入框粘贴 /home/xiaolingxiaotan/... 或 ~/... 后回车。

选择 Windows 端的照片
---------------------
工具跑在 WSL 里，Windows 的 C:\ / F:\ / G:\ 挂在 /mnt/c、/mnt/f、/mnt/g。
三种入口都通向同一套加载逻辑：

    1) 「Windows 文件夹…」调 Windows 原生目录选择框（经 powershell.exe
       interop），选完直接得到 C:\ 路径并自动换算成 WSL 路径；
    2) 顶部输入框直接粘贴 C:\Users\...\照片（资源管理器复制的地址带引号也行），
       回车或点「打开」；粘贴单个图片路径则打开其所在目录并跳到该图；
    3) 「快捷位置」下拉直达各盘符、桌面、下载、图片、上次使用的目录、
       以及项目里的 data/raw。

--dialog 可强制选型：auto（默认，Windows 失败自动回退 WSL）/ windows / wsl。
程序内部一律用 WSL 路径读写，输出栏和剪贴板同时给出 C:\ 形式，方便在
资源管理器里定位 M2-3/annotations.json（固定写在 M2-3 目录下）。
照片放在 Windows 盘时建议 --image-mode none，避免大 JSON 跨 9p 写入变慢。

界面操作
--------
    选择文件夹      Windows 原生框 / 快捷位置 / 粘贴 C:\ 路径；自动按文件名排序
    滚轮            以鼠标位置为锚点缩放（X11 用 Button-4/5，Windows 用 MouseWheel）
    中键长按拖动    平移整张图片
    左键拖框内部    移动整个矩形框
    左键拖边框/角点 改变矩形框大小
    选标签          顶部「新框标签」下拉；开局不自动放任何框
    左键在空白处拖  新建一个矩形框（用当前选好的标签）
    一张图多类      反复「选标签 → 拖框」，每个框各有自己的数字类别
    方向键          按「微调步长」移动选中框（默认 0.1 px）
    Shift + 方向键  步长 ×10；Ctrl + 方向键 步长 ÷10
    0 ~ 9           有框时改选中框类别；无框时预设新框标签
    标签… 按钮      编辑每个数字 ID 后面的名字（可留空）/ 添加新类
    n / Delete      新建框 / 删除选中框
    f / PageUp / PageDown   适应窗口 / 上一张 / 下一张
    Enter / 确定    保存当前图并自动跳到下一张「未标注」的图
    点左侧列表      切回任意已标好的图，改完再按 Enter 即覆盖重标
    Ctrl + / -      放大 / 缩小界面字号（只改显示，不动标注坐标）

输出
----
产物固定写在 **M2-3 目录**下，不再跟着照片文件夹跑（题目 §9 交付物结构）：

    M2-3/annotations.json                全部标注（一个 JSON）
    M2-3/data/raw/<文件名>               原始采集图（题面 §9：抽检用）
    M2-3/data/labels/<文件名>.txt        同步导出的 YOLO txt（标注文件）
    M2-3/data/labels/preview/<文件名>.jpg 画好框的复核图（人工抽检用）
    M2-3/data/splits/                    train/val/test 清单（训练侧生成）

题面 §9 只要求 labels/ 放「标注文件」，所以 .txt 在 labels/ 顶层，
带框的复核图放在 labels/preview/ 子目录，两者不混在一起：训练脚本按
<文件名>.txt 找标注，不会被 preview 里的 .jpg 干扰。复核图最长边缩到
1280 px 存 JPEG，一张约 200~400 KB，不占空间。

界面顶部单独一行显示 JSON 完整路径，可点「复制路径 / 打开文件夹」；
每次保存和关闭窗口都会在终端打印路径。想换位置用 --output / --yolo-dir：

    python3 label_tool.py --folder photos/ --output /other/dir/ann.json

每个条目：
    image  : numpy 元信息 {path, shape:[H,W,3], dtype:"uint8", order:"HWC"}
    boxes  : numpy 风格二维数组 [[x1,y1,x2,y2,class_id], ...]，float，可含小数
    crops  : 每个框内的图像数组（numpy .tolist()），仅在 --image-mode crop 时写入

读回示例：
    import json, numpy as np
    doc = json.load(open("annotations.json", encoding="utf-8"))
    item = doc["items"][0]
    boxes = np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 5)
    crop = np.asarray(item["crops"][0], dtype=np.uint8)   # mode=crop 时

图像数组三档（--image-mode / 界面下拉）：
    crop（默认）只存框内区域，文件小，且「框 + 图片」都是 numpy 形式
    none  只存 shape/dtype/路径，JSON 最小
    full  存整张图的像素数组；1920×1080 约 20 MB/张，720 张会到十几 GB，慎用

默认同步写 YOLO txt（归一化 cx cy w h，6 位小数）到 data/labels/，
便于直接对接 §3 要求的标注格式；--yolo-dir 可改目录。
类别见下面「类别（数字 ID）」，默认 0~5 纯数字。

类别（数字 ID）
--------------
标注文件里 boxes[:,4] 存的是**数字类别 ID**，等于 classes 列表的下标；
名字只是写在数字后面的可选说明，随时可改、可留空，不影响已标注数据。
classes.txt 每行一类，支持：
    0             → ID 0，无名字
    0 书          → ID 0，名字「书」
    0:书 / 0,书   → 同上
    书            → 兼容旧格式，按行号自动编号
界面「标签…」按钮也能直接改名 / 添加新类，并写回 classes.txt。
JSON 里另有 class_ids / class_names，供 M2-4 的 detections JSON 对齐。

注意：读入时会自动做 EXIF 方向转正，标注坐标基于转正后的图像；
中文路径用 PIL 读取，不依赖 cv2.imread。

中文字体：WSL 默认不含中文字体，界面中文会显示成空白方块。启动时会自动把
Windows 自带字体引用到 ~/.local/share/fonts（只写用户目录、不进项目、不需要
sudo），并显式把界面字体设为含中文字形的字体族。若仍是空白，可用
--font-family 指定（SimHei / Noto Sans SC / Microsoft YaHei 等）。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
except ImportError:  # --self-test 允许在没有 tkinter 的环境跑
    tk = None
    tkfont = None
    filedialog = None
    messagebox = None
    ttk = None
    ImageTk = None


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 标签以「数字 ID」为准，名字只是写在数字后面的可选说明。
# 列表下标 == 类别 ID；classes.txt 可写 "0 书" / "0:书" / 纯 "0"。
DEFAULT_CLASSES: List[str] = [str(i) for i in range(6)]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_COLORS = ["#A32D2D", "#185FA5", "#0F6E56", "#854F0B", "#534AB7", "#993C1D"]
HANDLE_R = 7          # 控制点命中半径（屏幕像素）
EDGE_TOL = 4          # 边框命中容差（屏幕像素）
ZOOM_MIN, ZOOM_MAX = 0.02, 128.0
CANVAS_BG = "#E8EBEF"

# 项目根目录 = 本文件所在目录（M2-3）。标注产物一律固定落在这里，
# 不跟着「所选照片文件夹」跑，方便查找并对齐题目 §9 的交付物结构。
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "annotations.json")
DEFAULT_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
DEFAULT_LABELS_DIR = os.path.join(PROJECT_DIR, "data", "labels")
# 带框的可视化图单独放 labels/preview/：labels/ 顶层只留训练要读的 .txt，
# 免得训练侧把 .jpg 也当标注文件扫进去。
DEFAULT_PREVIEW_DIR = os.path.join(DEFAULT_LABELS_DIR, "preview")
DEFAULT_SPLITS_DIR = os.path.join(PROJECT_DIR, "data", "splits")
RUN_LOG = os.path.join(PROJECT_DIR, "label_tool_run.log")

# 画预览图用的中文字体：Windows 自带字体，经 /mnt/c 直接读，不用安装
PREVIEW_FONT_SOURCES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",            # 微软雅黑
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",          # 黑体
    "/mnt/c/Windows/Fonts/Deng.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


# ---------------------------------------------------------------------------
# Windows / WSL 跨系统路径
#
# 工具跑在 WSL 里，手机导出的照片却通常放在 Windows 盘（C:\、F:\、G:\）。
# WSL 把这些盘挂在 /mnt/<小写盘符>，程序内部一律用 WSL 路径读写文件，
# 只在界面/剪贴板上给出 Windows 形式，方便回资源管理器里找。
# ---------------------------------------------------------------------------

ON_WSL = (os.name != "nt" and os.path.isdir("/mnt/c/Windows"))
WIN_MOUNT_ROOT = "/mnt"
PS_DIALOG_TIMEOUT = 300              # 原生对话框最长等待秒数，超时自动回退
CANCELLED = "已取消"
LAST_FOLDER_FILE = os.path.expanduser("~/.cache/m2_3_label_tool/last_folder.txt")
_SKIP_USER_DIRS = {"all users", "default", "default user", "public",
                   "wsiaccount", "desktop.ini"}

# 不把 initial 路径拼进脚本字符串：Windows 用户名可能带引号（本机就是 j'c），
# 插值会破坏 PowerShell 语法。改成用环境变量 + WSLENV 传递。
PS_FOLDER_DIALOG = r'''
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = '选择照片所在文件夹（Windows 磁盘）'
$d.ShowNewFolderButton = $true
$d.RootFolder = [System.Environment+SpecialFolder]::MyComputer
$init = $env:M23_INITIAL
if (-not $init -or -not (Test-Path -LiteralPath $init)) {
    $init = [Environment]::GetFolderPath('MyPictures')
}
if (-not $init -or -not (Test-Path -LiteralPath $init)) {
    $init = $env:USERPROFILE
}
if ($init -and (Test-Path -LiteralPath $init)) { $d.SelectedPath = $init }
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.WindowState = 'Minimized'
$owner.Opacity = 0
$owner.Show()
$r = $d.ShowDialog($owner)
$owner.Close()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $d.SelectedPath
}
'''


def _powershell_exe() -> Optional[str]:
    """定位 Windows 的 powershell.exe（WSL 走 interop，Windows 本机直接调）。"""
    if os.name == "nt":
        return shutil.which("powershell.exe") or "powershell.exe"
    candidates = [shutil.which("powershell.exe"),
                  "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


def _find_drive_mount(letter: str) -> Optional[str]:
    """自定义 WSL 挂载点（非 /mnt/c）时，从 /proc/mounts 里找 X:\\ 的挂载位置。"""
    try:
        with open("/proc/mounts", encoding="utf-8", errors="replace") as f:
            for line in f:
                src, sep, rest = line.partition(" on ")
                if sep and src.upper().startswith(letter.upper() + ":\\"):
                    return rest.split(" ", 1)[0]
    except OSError:
        pass
    return None


def is_windows_drive(path: str) -> bool:
    """这个路径是否落在 Windows 盘上（决定要不要提示写盘慢、要不要显示 C:\\ 形式）。"""
    p = os.path.abspath(path)
    if ON_WSL:
        return re.match(r"^/mnt/[A-Za-z](?:/|$)", p) is not None
    return os.name == "nt" and re.match(r"^[A-Za-z]:[\\/]", p) is not None


def is_wsl_native(path: str) -> bool:
    """是否是 WSL 本机路径（/home/... 、~/... ），即不在任何 Windows 盘符下。

    Windows 原生目录框进不去 \\wsl$，起始位置是这类目录时必须换 Tk 对话框，
    否则用户看到的是「此电脑」里根本没有目标文件夹。
    """
    if not ON_WSL or not path:
        return False
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    return not re.match(r"^/mnt(?:/|$)", p)


def local_to_windows(path: str) -> str:
    """WSL 路径 → Windows 路径；WSL 本地路径转成 \\\\wsl$\\<发行版>\\... 。"""
    p = os.path.abspath(path)
    if os.name == "nt" or not ON_WSL:
        return os.path.normpath(p)
    m = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", p)
    if m:
        return m.group(1).upper() + ":\\" + (m.group(2) or "").replace("/", "\\")
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    if distro:
        return "\\\\wsl$\\" + distro + p.replace("/", "\\")
    return p


def windows_to_local(text: str) -> str:
    """把用户给的路径（Windows / WSL / UNC / 带引号）转成本地可访问路径。

    支持：C:\\Users\\...\\照片、C:/Users/...、\\\\?\\C:\\...、\\\\wsl$\\Ubuntu2204\\home\\...、
    /mnt/c/...、~/...、相对路径，以及资源管理器复制时带上的首尾双引号。
    """
    s = str(text).strip()
    # 引号只剥首尾成对/单引号，不动路径中间可能存在的引号字符
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'"):
        s = s[1:-1].strip()
    if not s:
        return ""
    s = re.sub(r"^\\\\[?.]\\", "", s)          # Win32 长路径前缀 \\?\ / \\.\
    m = re.match(r"^\\\\wsl\$\\[^\\]+[\\/](.*)$", s)
    if m:                                      # UNC 指向本机 WSL → 去掉发行版前缀
        s = "/" + m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        rest = m.group(2).replace("\\", "/") if ON_WSL else m.group(2)
        if ON_WSL:
            mount = os.path.join(WIN_MOUNT_ROOT, m.group(1).lower())
            if not os.path.isdir(mount):
                mount = _find_drive_mount(m.group(1)) or mount
            s = os.path.join(mount, rest)
        else:
            s = m.group(1).upper() + ":/" + rest
    if ON_WSL:
        s = s.replace("\\", "/")
    return os.path.abspath(os.path.expanduser(s))


def resolve_folder_arg(text: str) -> Tuple[str, Optional[str]]:
    """返回 (文件夹, 要跳转到的图片文件名)。传图片路径时自动打开其所在目录。"""
    p = windows_to_local(text)
    if os.path.isfile(p):
        name = os.path.basename(p) if os.path.splitext(p)[1].lower() in IMAGE_EXTS else None
        return os.path.dirname(p), name
    return p, None


def windows_default_initial() -> str:
    """Windows 对话框/WSL 对话框的默认起始位置：当前用户主目录。"""
    root = os.path.join(WIN_MOUNT_ROOT, "c", "Users")
    if ON_WSL and os.path.isdir(root):
        try:
            for name in sorted(os.listdir(root)):
                home = os.path.join(root, name)
                if (name.lower() not in _SKIP_USER_DIRS
                        and not name.endswith(".ini") and os.path.isdir(home)):
                    return home
        except OSError:
            pass
        return root
    return os.getcwd()


def remember_folder(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(LAST_FOLDER_FILE), exist_ok=True)
        with open(LAST_FOLDER_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.abspath(path))
    except OSError:
        pass


def last_folder() -> Optional[str]:
    try:
        with open(LAST_FOLDER_FILE, encoding="utf-8") as f:
            p = f.read().strip()
        return p if p and os.path.isdir(p) else None
    except OSError:
        return None


def windows_shortcuts() -> List[Tuple[str, str]]:
    """界面「快捷位置」下拉：常用 Windows 目录 + 各盘符 + 上次使用的目录。"""
    out: List[Tuple[str, str]] = []

    def add(label: str, path: str) -> None:
        if path and os.path.isdir(path) and all(p != path for _, p in out):
            out.append((label, path))

    add("项目目录 M2-3", PROJECT_DIR)
    # 图片库放在最前面：标注 99% 的时间就在这个目录里，别让用户每次翻盘符
    n_raw = len(scan_images(DEFAULT_RAW_DIR)) if os.path.isdir(DEFAULT_RAW_DIR) else 0
    if n_raw:
        add(f"图片库 data/raw（{n_raw} 张）", DEFAULT_RAW_DIR)
    prev = last_folder()
    if prev:
        add("上次使用：" + (os.path.basename(prev) or prev), prev)
    add("项目当前目录", os.getcwd())
    if not (ON_WSL or os.name == "nt"):
        return out
    if ON_WSL:
        for p in sorted(glob.glob(WIN_MOUNT_ROOT + "/[A-Za-z]")):
            if os.path.isdir(p):
                add("盘符 " + os.path.basename(p).upper() + ":\\", p)
        users = os.path.join(WIN_MOUNT_ROOT, "c", "Users")
        if os.path.isdir(users):
            try:
                names = sorted(os.listdir(users))
            except OSError:
                names = []
            for name in names:
                home = os.path.join(users, name)
                if (name.lower() in _SKIP_USER_DIRS or name.endswith(".ini")
                        or not os.path.isdir(home)):
                    continue
                add("用户 " + name, home)
                for sub, cn in (("Desktop", "桌面"), ("Downloads", "下载"),
                                ("Pictures", "图片"), ("Documents", "文档")):
                    add(f"{cn} · {name}", os.path.join(home, sub))
    return out


def pick_windows_folder(initial: Optional[str] = None) -> Tuple[Optional[str], str]:
    """调 Windows 原生「选择文件夹」对话框，返回 (Windows 路径, 状态说明)。

    只在 WSL / Windows 本机可用；取消返回 CANCELLED，失败返回原因，调用方
    据此决定是回退到 Tk 对话框还是什么都不做（绝不连弹两个框）。
    """
    exe = _powershell_exe()
    if not exe:
        return None, "未找到 powershell.exe"
    env = os.environ.copy()
    # 只有 Windows 盘路径能作为原生对话框的 SelectedPath；WSL 本地目录会变成
    # \\wsl$\... UNC，不在「此电脑」树里，传给对话框反而可能报错。
    win_initial = (local_to_windows(initial)
                   if initial and (is_windows_drive(initial) or os.name == "nt")
                   else "")
    env["M23_INITIAL"] = win_initial
    if ON_WSL:
        env["WSLENV"] = (env.get("WSLENV", "") + ":M23_INITIAL").lstrip(":")
    try:
        r = subprocess.run(
            [exe, "-NoProfile", "-STA", "-Command", PS_FOLDER_DIALOG],
            env=env, capture_output=True, timeout=PS_DIALOG_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, f"选择超时（{PS_DIALOG_TIMEOUT} 秒）"
    except Exception as exc:
        return None, repr(exc)
    text = r.stdout.decode("utf-8", "replace").strip()
    if not text:
        err = r.stderr.decode("utf-8", "replace").strip()
        return None, err[:200] if err else CANCELLED
    return text.splitlines()[-1].strip(), "ok"


# ---------------------------------------------------------------------------
# 中文字体与界面字号
#
# WSL 默认一个中文字体都没有：Tk 用 DejaVu Sans 渲染中文时，每个字都变成
# 空白方块，看起来就像「全是空格」。所以这里做两件事：
#   1) 启动前把 Windows 自带的字体「引用」到 WSL 用户字体目录（不进项目、
#      不需要 sudo、不覆盖已有文件）；
#   2) 显式把所有 Tk 命名字体 / ttk 样式换成含中文字形的字体族，并放大字号。
# ---------------------------------------------------------------------------

CJK_FAMILY_CANDIDATES: Tuple[str, ...] = (
    # 静态字体优先：可变字体（Noto Sans SC VF）在部分 Tk/Xft 组合下
    # 可能只加载默认实例，中文有变成空白的风险，所以放后面。
    "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans SC",
    "DengXian", "SimSun", "PingFang SC", "Heiti SC", "Droid Sans Fallback",
)

# WSL 能直接读 Windows 的字体目录；这里只是运行时引用本机已有字体，不随项目分发。
WSL_FONT_SOURCES: Tuple[str, ...] = (
    "/mnt/c/Windows/Fonts/simhei.ttf",          # 静态 TTF，Tk 兼容性最好
    "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",   # OFL 授权，备选
    "/mnt/c/Windows/Fonts/Deng.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
)

DEFAULT_UI_SIZE = 14
MIN_UI_SIZE, MAX_UI_SIZE = 9, 28


def _has_cjk_font() -> bool:
    """fontconfig 里有没有「字体文件真实存在」的中文字体。

    只查 family 会被陈旧缓存骗到：字体文件已被移走时 fc-list 仍可能报旧条目，
    界面照样是空白。所以再取文件路径逐个确认，缺文件就当作没有、触发重装。
    """
    try:
        r = subprocess.run(
            ["fc-list", ":lang=zh", "file"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        lines = r.stdout.decode(errors="replace").splitlines()
        files = [ln.split(":", 1)[0].strip() for ln in lines if ln.strip()]
        return any(os.path.isfile(p) for p in files)
    except Exception:
        return False


def ensure_cjk_font(quiet: bool = False) -> str:
    """确保 WSL 里有中文字体，必须在 ``tk.Tk()`` 之前调用。

    Tk 在创建窗口时就把可用字体列表缓存了，之后再装字体这一轮不会生效。
    只写用户目录 ``~/.local/share/fonts``，不改系统目录、不需要 sudo。
    """
    if os.name == "nt":
        return "Windows 自带中文字体"
    if _has_cjk_font():
        return "已检测到中文字体"
    fontdir = os.path.expanduser("~/.local/share/fonts")
    src = next((p for p in WSL_FONT_SOURCES if os.path.isfile(p)), None)
    if src is None:
        msg = ("未找到中文字体，界面中文会显示为空白；"
               "请执行 sudo apt install fonts-noto-cjk")
        if not quiet:
            print("[字体] " + msg, file=sys.stderr)
        return msg
    try:
        os.makedirs(fontdir, exist_ok=True)
        dst = os.path.join(fontdir, os.path.basename(src))
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        subprocess.run(["fc-cache", "-f", fontdir],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=30)
    except Exception as exc:
        msg = f"装入中文字体失败：{exc!r}"
        if not quiet:
            print("[字体] " + msg, file=sys.stderr)
        return msg
    ok = _has_cjk_font()
    msg = "已从 Windows 字体目录装入中文字体" if ok else "字体已复制但 fontconfig 未识别"
    if not quiet:
        print("[字体] " + msg)
    return msg


def pick_cjk_family(root, requested: Optional[str] = None) -> str:
    """从 Tk 能枚举到的字体里挑一个真正含中文字形的族名。"""
    if requested:
        return requested
    try:
        fams = set(tkfont.families(root))
    except Exception:
        fams = set()
    for name in CJK_FAMILY_CANDIDATES:
        if name in fams:
            return name
    try:
        return tkfont.nametofont("TkDefaultFont").actual("family")
    except Exception:
        return "DejaVu Sans"


def apply_ui_fonts(root, fonts: Dict[str, object],
                   family: str, size: int) -> int:
    """统一 Tk 命名字体 / ttk 样式的字体族与字号，返回实际字号。

    改「命名字体」而不是逐个控件传 font：所有未显式指定的控件会一起更新，
    运行时按 Ctrl +/- 也能即时刷新，不必重建界面。
    """
    size = max(MIN_UI_SIZE, min(MAX_UI_SIZE, int(size)))
    if tkfont is None:
        return size
    named = ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
             "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
             "TkIconFont", "TkTooltipFont")
    for nm in named:
        try:
            tkfont.nametofont(nm).configure(family=family, size=size)
        except Exception:
            pass
    if ttk is not None:
        try:
            st = ttk.Style(root)
            st.configure(".", font=(family, size))
            for style_name in ("TButton", "TLabel", "TCombobox", "TEntry",
                               "TLabelframe.Label", "TNotebook.Tab"):
                st.configure(style_name, font=(family, size))
        except Exception:
            pass
    if not fonts:
        fonts["base"] = tkfont.Font(root=root, family=family, size=size)
        fonts["bold"] = tkfont.Font(root=root, family=family, size=size,
                                    weight="bold")
        fonts["small"] = tkfont.Font(root=root, family=family,
                                     size=max(9, size - 2))
        fonts["label"] = tkfont.Font(root=root, family=family,
                                     size=max(9, size - 1))
        fonts["mono"] = tkfont.Font(root=root, family=family, size=size)
    else:
        for key, obj in fonts.items():
            extra = {"small": -2, "label": -1}.get(key, 0)
            obj.configure(family=family, size=max(9, size + extra))
    return size


# ---------------------------------------------------------------------------
# 纯数据结构与纯函数（不依赖 tkinter，便于自检）
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """一个矩形框，坐标是原图像素，允许小数（亚像素）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0

    def copy(self) -> "Box":
        return Box(self.x1, self.y1, self.x2, self.y2, self.cls)

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, float(self.cls)]

    def valid(self, eps: float = 1e-6) -> bool:
        return self.x2 - self.x1 > eps and self.y2 - self.y1 > eps

    def normalized(self) -> "Box":
        """保证 x1<=x2、y1<=y2，拖过中线后仍得到合法框。"""
        return Box(
            min(self.x1, self.x2), min(self.y1, self.y2),
            max(self.x1, self.x2), max(self.y1, self.y2), self.cls,
        )


@dataclass
class ViewTransform:
    """图像坐标 <-> 屏幕坐标：screen = image * zoom + offset。"""

    zoom: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0

    def to_screen(self, ix: float, iy: float) -> Tuple[float, float]:
        return ix * self.zoom + self.off_x, iy * self.zoom + self.off_y

    def to_image(self, sx: float, sy: float) -> Tuple[float, float]:
        return (sx - self.off_x) / self.zoom, (sy - self.off_y) / self.zoom

    def copy(self) -> "ViewTransform":
        return ViewTransform(self.zoom, self.off_x, self.off_y)

    def fit(self, width: int, height: int, view_w: int, view_h: int) -> None:
        """自适应分辨率：按画布尺寸把整张图居中放下。"""
        if width <= 0 or height <= 0 or view_w <= 0 or view_h <= 0:
            self.zoom, self.off_x, self.off_y = 1.0, 0.0, 0.0
            return
        self.zoom = min(view_w / float(width), view_h / float(height))
        self.off_x = (view_w - width * self.zoom) / 2.0
        self.off_y = (view_h - height * self.zoom) / 2.0

    def zoom_at(self, mx: float, my: float, factor: float) -> None:
        """以鼠标位置为锚点缩放：该点下的图像坐标保持不变。"""
        ix, iy = self.to_image(mx, my)
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * factor))
        self.off_x = mx - ix * self.zoom
        self.off_y = my - iy * self.zoom


def make_center_box(width: int, height: int, ratio: float = 0.22) -> Box:
    """开局在画面中心放一个默认框。"""
    w, h = width * ratio, height * ratio
    return Box((width - w) / 2.0, (height - h) / 2.0,
               (width + w) / 2.0, (height + h) / 2.0, 0)


def clamp_box(box: Box, width: int, height: int) -> Box:
    """把框限制在图像内；不越界时原样保留小数，不损失亚像素精度。"""
    b = box.copy()
    b.x1 = min(max(b.x1, 0.0), float(width))
    b.x2 = min(max(b.x2, 0.0), float(width))
    b.y1 = min(max(b.y1, 0.0), float(height))
    b.y2 = min(max(b.y2, 0.0), float(height))
    return b


def handle_points(x1: float, y1: float, x2: float, y2: float) -> Dict[str, Tuple[float, float]]:
    """八个控制点的屏幕坐标。"""
    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return {
        "nw": (x1, y1), "n": (xm, y1), "ne": (x2, y1),
        "e": (x2, ym), "se": (x2, y2), "s": (xm, y2),
        "sw": (x1, y2), "w": (x1, ym),
    }


def hit_test_mode(
    boxes: Sequence[Box], sx: float, sy: float, vt: ViewTransform,
    handle_r: float = HANDLE_R, edge_tol: float = EDGE_TOL,
) -> Tuple[Optional[int], str]:
    """返回 (框下标, 模式)。模式：8 个方向 / move / create。"""
    for i in range(len(boxes) - 1, -1, -1):
        b = boxes[i]
        x1, y1 = vt.to_screen(b.x1, b.y1)
        x2, y2 = vt.to_screen(b.x2, b.y2)
        lx, rx = min(x1, x2), max(x1, x2)
        ty, by = min(y1, y2), max(y1, y2)

        for name, (hx, hy) in handle_points(lx, ty, rx, by).items():
            if abs(sx - hx) <= handle_r and abs(sy - hy) <= handle_r:
                return i, name

        if lx - edge_tol <= sx <= rx + edge_tol and ty - edge_tol <= sy <= by + edge_tol:
            near_w, near_e = abs(sx - lx) <= edge_tol, abs(sx - rx) <= edge_tol
            near_n, near_s = abs(sy - ty) <= edge_tol, abs(sy - by) <= edge_tol
            if near_w or near_e or near_n or near_s:
                mode = ("n" if near_n else "") + ("s" if near_s else "") \
                     + ("w" if near_w else "") + ("e" if near_e else "")
                return i, mode or "move"
            return i, "move"
    return None, "create"


def apply_drag(base: Box, mode: str, dx: float, dy: float) -> Box:
    """把「图像坐标下的位移」应用到框；dx/dy 由屏幕位移除以 zoom 得到。"""
    b = base.copy()
    if mode == "move":
        b.x1 += dx; b.x2 += dx; b.y1 += dy; b.y2 += dy
    else:
        if "w" in mode:
            b.x1 += dx
        if "e" in mode:
            b.x2 += dx
        if "n" in mode:
            b.y1 += dy
        if "s" in mode:
            b.y2 += dy
    return b.normalized()


def boxes_to_array(boxes: Sequence[Box]) -> np.ndarray:
    """转成 numpy (N,5) float64：x1,y1,x2,y2,class_id。"""
    if not boxes:
        return np.zeros((0, 5), dtype=np.float64)
    return np.asarray([b.as_list() for b in boxes], dtype=np.float64)


def array_to_boxes(arr: Sequence[Sequence[float]]) -> List[Box]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 5)
    return [Box(float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                int(round(float(r[4])))) for r in a]


def yolo_line(box: Box, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2.0 / width
    cy = (box.y1 + box.y2) / 2.0 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return f"{box.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def scan_images(folder: str) -> List[str]:
    """列出文件夹里的图片（大小写扩展名都认）。

    用 os.scandir 且只对扩展名匹配项做 is_file()：Windows 盘走 9p，
    每次 stat 都要跨系统边界，720 张图时这点开销差别很明显。
    """
    try:
        with os.scandir(folder) as it:
            names = [e.name for e in it
                     if os.path.splitext(e.name)[1].lower() in IMAGE_EXTS
                     and e.is_file()]
    except OSError:
        return []
    return [os.path.join(folder, n) for n in sorted(names)]


def parse_class_lines(lines: Sequence[str]) -> List[str]:
    """把类别配置行解析成「显示串」列表，保证 列表下标 == 类别 ID。

    每行一类，支持：
        0               → ID 0，无名字，显示 "0"
        0 书            → ID 0，名字「书」，显示 "0 书"
        0:书 / 0,书     → 同上（冒号/逗号/等号/短横线作分隔符均可）
        书              → 兼容旧格式，按行序号自动编号
    跳号的 ID 会用纯数字补齐中间空位，因此下标恒等于 ID；
    以 # 开头的行视为注释。名字可以随时改，已标注框里存的是数字 ID。
    """
    specs: Dict[int, str] = {}
    auto = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s*[:：,，=\-]?\s*(.*)$", line)
        if m:
            cid, name = int(m.group(1)), m.group(2).strip()
        else:
            cid, name = auto, line
        specs[cid] = name
        auto = max(auto, cid + 1)
    n = max(auto, len(specs), 1)
    return [f"{i} {specs[i]}".rstrip() if specs.get(i) else str(i)
            for i in range(n)]


def split_class_label(label: str) -> Tuple[int, str]:
    """显示串 → (ID, 名字)；旧 JSON 里的 "class_3" 之类会返回 (-1, 原名)。"""
    m = re.match(r"^(\d+)\s*(.*)$", str(label).strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    return -1, str(label)


def default_classes_file() -> Optional[str]:
    """没显式给 --classes 时优先用项目里的 classes.txt。

    以前这里是硬回落到 DEFAULT_CLASSES（纯数字 0~5），结果 annotations.json 的
    头部写成 classes=["0".."5"]、class_names 全空 —— 框里的数字 ID 没错，但类别名
    丢了，报告和 M2-4 的 detections JSON 都要用名字，所以改成先找 classes.txt。
    """
    path = os.path.join(PROJECT_DIR, "classes.txt")
    return path if os.path.isfile(path) else None


def load_classes(path: Optional[str]) -> List[str]:
    if not path:
        path = default_classes_file()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return parse_class_lines(f.readlines())
    return list(DEFAULT_CLASSES)


def read_image_array(path: str) -> np.ndarray:
    """读图 + EXIF 方向转正 + 转 RGB；返回 contiguous uint8 (H,W,3)。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))


def crop_array(img: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = int(max(0, min(w, round(min(box.x1, box.x2)))))
    x2 = int(max(0, min(w, round(max(box.x1, box.x2)))))
    y1 = int(max(0, min(h, round(min(box.y1, box.y2)))))
    y2 = int(max(0, min(h, round(max(box.y1, box.y2)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(img[y1:y2, x1:x2])


# ---------------------------------------------------------------------------
# JSON 存储
# ---------------------------------------------------------------------------

def _write_run_log(tag: str, text: str) -> None:
    """把启动/窗口状态/异常追加到 M2-3/label_tool_run.log。

    「终端在跑但看不到窗口」这类问题光看终端看不出来：窗口可能已经建好，
    只是没映射到屏幕（WSLg 掉线、在管理员终端里启动、DISPLAY 指错）。把
    viewable/mapped/geometry 和 DISPLAY 记下来，下次直接看日志就能定位。
    """
    try:
        with open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write("%s [%s] %s\n"
                    % (time.strftime("%Y-%m-%d %H:%M:%S"), tag, text))
    except OSError:
        pass


def _preview_font(size: int):
    """取一个能画中文的字体对象；都找不到就返回 None（预览图只画框不写字）。"""
    for p in PREVIEW_FONT_SOURCES:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)      # Pillow >= 10.1 才支持 size
    except Exception:
        return None


class AnnotationStore:
    """所有标注放在一个 JSON 文件里，按文件名索引；重标即覆盖同名条目。"""

    def __init__(self, path: str, classes: Sequence[str],
                 image_mode: str = "crop", yolo_dir: Optional[str] = None,
                 raw_dir: Optional[str] = None):
        self.path = path
        self.classes = list(classes)
        self.image_mode = image_mode
        self.yolo_dir = yolo_dir
        self.raw_dir = raw_dir
        self.last_preview: Optional[str] = None      # 最近一次生成的复核图
        self.items: Dict[str, dict] = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            for item in doc.get("items", []):
                if "file" in item and "boxes" in item:
                    self.items[item["file"]] = item

    def put(self, name: str, image_path: str, img: np.ndarray,
            boxes: Sequence[Box]) -> None:
        h, w = img.shape[:2]
        arr = boxes_to_array(boxes)
        record = {
            "file": name,
            "image_path": os.path.abspath(image_path),
            "image": {
                "path": os.path.abspath(image_path),
                "shape": [int(h), int(w), 3],
                "dtype": "uint8",
                "order": "HWC",
                "data": img.tolist() if self.image_mode == "full" else None,
            },
            "boxes": arr.tolist(),
            "boxes_shape": [int(arr.shape[0]), 5],
            "crops": [],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.image_mode == "crop":
            for b in boxes:
                c = crop_array(img, b)
                record["crops"].append(c.tolist() if c is not None else [])
        self.items[name] = record
        # 同一张图重新标定 → 覆盖同名条目；新图 → 追加一条。
        self.import_raw(name, image_path)

        if self.yolo_dir:
            os.makedirs(self.yolo_dir, exist_ok=True)
            stem = os.path.splitext(name)[0]
            with open(os.path.join(self.yolo_dir, stem + ".txt"), "w",
                      encoding="utf-8") as f:
                for b in boxes:
                    f.write(yolo_line(b, w, h) + "\n")
            # 同步出一张「画好框」的复核图，人工抽检用
            self.last_preview = self.export_preview(
                name, img, boxes, os.path.join(self.yolo_dir, "preview"))

    def export_preview(self, name: str, img: np.ndarray,
                       boxes: Sequence[Box], out_dir: str,
                       max_side: int = 1280) -> Optional[str]:
        """把标注结果画到原图上，存成 JPEG，供人工复核框贴不贴合。

        为什么单独出这张图：annotations.json 里的坐标是数字，抽检时看不出
        框偏没偏；题面 §10 也要求「标注工具直接生成的框必须人工核对」，
        有图才能一眼看出压边、漏标、类别选错。

        落在 labels/preview/，与 .txt 分开：labels/ 顶层保持「纯标注文件」，
        训练脚本扫 .txt 时不会被 .jpg 干扰。
        """
        if not boxes:
            return None
        h, w = img.shape[:2]
        pil = Image.fromarray(img)
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            # BOX 是面积平均，缩小不产生摩尔纹；预览图不需要更高质量的 BICUBIC
            pil = pil.resize((max(1, int(round(w * scale))),
                              max(1, int(round(h * scale)))), Image.BOX)
        draw = ImageDraw.Draw(pil)
        side = max(pil.size)
        lw = max(2, int(round(side / 480)))          # 线宽随图大小走，别细到看不见
        font = _preview_font(max(14, int(round(side / 42))))
        for b in boxes:
            x1, y1, x2, y2 = (min(b.x1, b.x2) * scale, min(b.y1, b.y2) * scale,
                              max(b.x1, b.x2) * scale, max(b.y1, b.y2) * scale)
            color = CLASS_COLORS[b.cls % len(CLASS_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            label = (self.classes[b.cls] if 0 <= b.cls < len(self.classes)
                     else str(b.cls))
            if font is not None:
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = len(label) * 8, 14
                ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
                draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4],
                               fill=color)
                draw.text((x1 + 3, ty + 1), label, fill="#FFFFFF", font=font)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg")
        try:
            pil.save(dst, quality=88)
        except OSError:
            return None
        return dst

    def import_raw(self, name: str, image_path: str) -> Optional[str]:
        """把原图归到 data/raw（题面 §9：raw 放原始采集图，labels 放标注文件）。

        同名 -> 覆盖（重新标定）；不同名 -> 追加。
        源本来就在 data/raw 时直接跳过，避免 copy2 抛 SameFileError。
        """
        if not self.raw_dir:
            return None
        os.makedirs(self.raw_dir, exist_ok=True)
        dst = os.path.join(self.raw_dir, name)
        src = os.path.abspath(image_path)
        if src != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return dst

    def save(self) -> int:
        doc = {
            "format": "m2-3-bbox-annotation",
            "version": 1,
            "classes": self.classes,
            "class_ids": list(range(len(self.classes))),
            "class_names": [split_class_label(s)[1] for s in self.classes],
            "class_id_note": "boxes[:,4] 是数字类别 ID，等于 classes/class_ids 的下标；"
                             "改动名字不影响已标注数据",
            "image_mode": self.image_mode,
            "numpy": {
                "boxes": {
                    "dtype": "float64", "shape": "[N, 5]",
                    "columns": ["x1", "y1", "x2", "y2", "class_id"],
                    "coords": "absolute pixels, origin top-left, float (subpixel allowed)",
                },
                "image": {
                    "dtype": "uint8", "order": "HWC", "color": "RGB",
                    "note": "image.data is null unless image_mode=full; "
                            "image_mode=crop stores per-box arrays in crops[]",
                },
            },
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(self.items),
            "items": [self.items[k] for k in sorted(self.items)],
        }
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, self.path)   # 原子替换，中断不会写坏文件
        return os.path.getsize(self.path)


# ---------------------------------------------------------------------------
# 画布：显示 + 交互
# ---------------------------------------------------------------------------

class CanvasView:
    """持有 tk.Canvas，负责渲染图片、绘制框、鼠标交互。"""

    def __init__(self, parent, classes: Sequence[str]):
        self.canvas = tk.Canvas(parent, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.classes = list(classes)
        self.img: Optional[np.ndarray] = None
        self.vt = ViewTransform()
        self.boxes: List[Box] = []
        self.selected = 0
        # pen_class：新建框使用的标签 ID，与「选中框的类别」解耦。
        # 开局没有框，用户先选标签再拖框，一张图里可连续拖出不同类别的框。
        self.pen_class = 0
        self.view_w, self.view_h = 1, 1
        self.photo = None
        self.drag: Optional[dict] = None
        self.pan_start: Optional[Tuple[int, int]] = None
        self.temp_box: Optional[Box] = None
        self.on_change = None       # 由 App 挂上，用于刷新坐标栏/状态栏
        self.label_font = None      # 由 App 挂上：随字号变化的中文字体对象
        self._bind_events()

    # -- 事件绑定 -----------------------------------------------------------
    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<ButtonPress-1>", self._on_press_left)
        c.bind("<B1-Motion>", self._on_drag_left)
        c.bind("<ButtonRelease-1>", self._on_release_left)
        c.bind("<ButtonPress-2>", self._on_press_middle)
        c.bind("<B2-Motion>", self._on_drag_middle)
        c.bind("<ButtonRelease-2>", self._on_release_middle)
        # X11（WSL / Linux）滚轮是 Button-4/5，Windows 是 MouseWheel
        c.bind("<Button-4>", lambda e: self.zoom_by(e.x, e.y, 1.15))
        c.bind("<Button-5>", lambda e: self.zoom_by(e.x, e.y, 1.0 / 1.15))
        c.bind("<MouseWheel>", lambda e: self.zoom_by(
            e.x, e.y, 1.15 if e.delta > 0 else 1.0 / 1.15))
        c.bind("<Motion>", self._on_hover)

    def _on_configure(self, event) -> None:
        w, h = max(1, event.width), max(1, event.height)
        if (w, h) != (self.view_w, self.view_h):
            self.view_w, self.view_h = w, h
            self.render_all()

    # -- 载入 / 视图 --------------------------------------------------------
    def load_image(self, img: np.ndarray, boxes: Optional[Sequence[Box]] = None) -> None:
        self.img = img
        h, w = img.shape[:2]
        # 开局不自动放框：先选标签，再在图上拖出第一个框。
        # 切回已标过的图时 boxes 可能是空列表（那张图当时就无有效目标）。
        self.boxes = [b.copy() for b in boxes] if boxes else []
        self.selected = 0
        self.temp_box = None
        self.fit()

    def fit(self) -> None:
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        self.vt.fit(w, h, self.view_w, self.view_h)
        self.render_all()

    def zoom_by(self, mx: float, my: float, factor: float) -> None:
        self.vt.zoom_at(mx, my, factor)
        self.render_all()

    def render_all(self) -> None:
        self.render_image()
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def render_image(self) -> None:
        """只渲染可见区域，放大到 128× 也不会生成巨图。"""
        c = self.canvas
        c.delete("photo")
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        z = self.vt.zoom
        ix0 = max(0.0, (0.0 - self.vt.off_x) / z)
        iy0 = max(0.0, (0.0 - self.vt.off_y) / z)
        ix1 = min(float(w), (self.view_w - self.vt.off_x) / z)
        iy1 = min(float(h), (self.view_h - self.vt.off_y) / z)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sx0, sy0 = int(math.floor(ix0)), int(math.floor(iy0))
        sx1, sy1 = min(w, int(math.ceil(ix1))), min(h, int(math.ceil(iy1)))
        if sx1 <= sx0 or sy1 <= sy0:
            return
        crop = np.ascontiguousarray(self.img[sy0:sy1, sx0:sx1])
        out_w = max(1, int(round((sx1 - sx0) * z)))
        out_h = max(1, int(round((sy1 - sy0) * z)))
        resample = Image.NEAREST if z >= 2.0 else Image.BILINEAR
        pil = Image.fromarray(crop).resize((out_w, out_h), resample=resample)
        self.photo = ImageTk.PhotoImage(pil)
        px = self.vt.off_x + sx0 * z
        py = self.vt.off_y + sy0 * z
        c.create_image(px, py, anchor="nw", image=self.photo, tags="photo")
        c.tag_lower("photo")

    def draw_boxes(self) -> None:
        c = self.canvas
        c.delete("box")
        all_boxes = list(self.boxes)
        if self.temp_box is not None:
            all_boxes.append(self.temp_box)
        for i, b in enumerate(all_boxes):
            x1, y1 = self.vt.to_screen(b.x1, b.y1)
            x2, y2 = self.vt.to_screen(b.x2, b.y2)
            lx, rx = min(x1, x2), max(x1, x2)
            ty, by = min(y1, y2), max(y1, y2)
            color = CLASS_COLORS[int(b.cls) % len(CLASS_COLORS)]
            width = 3 if i == self.selected else 2
            c.create_rectangle(lx, ty, rx, by, outline=color,
                               width=width, tags="box")
            label = self.class_label(int(b.cls))
            # 中文标签宽度必须用字体实测（按字符数估算会把中文压窄、文字溢出）。
            fnt = self.label_font or "TkDefaultFont"
            try:
                tw = fnt.measure(label) + 14
                lh = fnt.metrics("linespace") + 8
            except Exception:      # 字体对象是字符串（自检路径）时退回估算
                tw, lh = max(40, len(label) * 10 + 14), 24
            c.create_rectangle(lx, max(0, ty - lh), lx + tw, max(0, ty),
                               fill=color, outline="", tags="box")
            c.create_text(lx + 7, max(0, ty - lh / 2.0), text=label, anchor="w",
                          fill="#FFFFFF", font=fnt, tags="box")
            if i == self.selected and self.temp_box is None:
                for hx, hy in handle_points(lx, ty, rx, by).values():
                    c.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5,
                                       fill="#FFFFFF", outline=color,
                                       width=1, tags="box")

    def class_label(self, cls: int) -> str:
        """框上显示的文字：数字 ID（后面可带名字）。"""
        return self.classes[cls] if 0 <= cls < len(self.classes) else str(cls)

    # -- 鼠标交互 -----------------------------------------------------------
    def _on_press_left(self, event) -> None:
        idx, mode = hit_test_mode(self.boxes, event.x, event.y, self.vt)
        if idx is not None:
            self.selected = idx
        self.drag = {
            "start": (event.x, event.y),
            "base": self.boxes[idx].copy() if idx is not None else None,
            "mode": mode, "idx": idx,
        }
        self.temp_box = None
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def _on_drag_left(self, event) -> None:
        d = self.drag
        if d is None:
            return
        z = self.vt.zoom
        dx = (event.x - d["start"][0]) / z
        dy = (event.y - d["start"][1]) / z
        if d["mode"] == "create":
            ix0, iy0 = self.vt.to_image(*d["start"])
            ix1, iy1 = self.vt.to_image(event.x, event.y)
            self.temp_box = Box(min(ix0, ix1), min(iy0, iy1),
                                max(ix0, ix1), max(iy0, iy1),
                                self.current_class())
            self.draw_boxes()
            return
        b = apply_drag(d["base"], d["mode"], dx, dy)
        if self.img is not None:
            h, w = self.img.shape[:2]
            b = clamp_box(b, w, h)
        self.boxes[d["idx"]] = b
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def _on_release_left(self, event) -> None:
        d = self.drag
        self.drag = None
        if d is None:
            return
        if d["mode"] == "create" and self.temp_box is not None:
            b = self.temp_box.normalized()
            if self.img is not None:
                h, w = self.img.shape[:2]
                b = clamp_box(b, w, h)
            if b.valid(eps=2.0):
                self.boxes.append(b)
                self.selected = len(self.boxes) - 1
        self.temp_box = None
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def _on_press_middle(self, event) -> None:
        self.pan_start = (event.x, event.y)
        self._safe_cursor("fleur")

    def _on_drag_middle(self, event) -> None:
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.vt.off_x += dx
        self.vt.off_y += dy
        self.pan_start = (event.x, event.y)
        self.render_all()

    def _on_release_middle(self, event) -> None:
        self.pan_start = None
        self._safe_cursor("arrow")

    def _on_hover(self, event) -> None:
        if self.drag or self.pan_start:
            return
        _, mode = hit_test_mode(self.boxes, event.x, event.y, self.vt)
        cursor = {
            "move": "fleur", "create": "crosshair",
            "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
        }.get(mode, "crosshair")
        self._safe_cursor(cursor)

    def _safe_cursor(self, name: str) -> None:
        try:
            self.canvas.configure(cursor=name)
        except Exception:
            pass

    # -- 供 App 调用 --------------------------------------------------------
    def current_class(self) -> int:
        """新建框使用的标签 ID（画笔类别），与选中框类别无关。"""
        return self.pen_class

    def set_pen_class(self, cls: int) -> None:
        """只改后续新建框用的标签，绝不动任何已有框。"""
        self.pen_class = cls

    def set_class(self, cls: int) -> None:
        """改选中框的类别；当前无框时等价于设置画笔类别。"""
        if 0 <= self.selected < len(self.boxes):
            self.boxes[self.selected].cls = cls
            self.draw_boxes()
        else:
            self.pen_class = cls

    def nudge(self, dx: float, dy: float) -> None:
        """键盘微调：直接改浮点坐标，实现亚像素级移动。"""
        if not (0 <= self.selected < len(self.boxes)) or self.img is None:
            return
        h, w = self.img.shape[:2]
        b = self.boxes[self.selected].copy()
        b.x1 += dx; b.x2 += dx; b.y1 += dy; b.y2 += dy
        self.boxes[self.selected] = clamp_box(b, w, h)
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def new_box(self) -> None:
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        b = make_center_box(w, h)
        b.cls = self.pen_class
        self.boxes.append(b)
        self.selected = len(self.boxes) - 1
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def delete_box(self) -> None:
        if not self.boxes:
            return
        self.boxes.pop(self.selected)
        # 删空后不再补默认框：开局与删空都保持「无框」，由用户主动新建。
        self.selected = max(0, min(self.selected, len(self.boxes) - 1))
        self.draw_boxes()
        if self.on_change:
            self.on_change()

    def set_coords(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not (0 <= self.selected < len(self.boxes)) or self.img is None:
            return
        h, w = self.img.shape[:2]
        b = Box(x1, y1, x2, y2, self.boxes[self.selected].cls).normalized()
        self.boxes[self.selected] = clamp_box(b, w, h)
        self.draw_boxes()
        if self.on_change:
            self.on_change()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class AnnotatorApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.folder: Optional[str] = None
        self.files: List[str] = []
        self.index = -1
        self.img: Optional[np.ndarray] = None
        self.classes = load_classes(args.classes)
        self.image_mode = args.image_mode
        self.step = float(args.step)
        self.store: Optional[AnnotationStore] = None
        self.last_save = ""
        self.coord_vars: List[tk.StringVar] = []
        self.coord_entries: List[tk.Entry] = []
        # 字体必须在创建控件之前配置：指定含中文字形的字体族并统一放大字号
        self.fonts: Dict[str, object] = {}
        self.ui_family = pick_cjk_family(root, args.font_family)
        self.ui_size = apply_ui_fonts(root, self.fonts,
                                      self.ui_family, args.font_size)
        self._build_ui()
        self.view.label_font = self.fonts.get("label")
        self.view.on_change = self.on_view_change
        # 关窗口时兜底再存一次，并在终端打印完整输出路径
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Return>", self.on_return)
        root.bind("<Key>", self.on_key)
        root.bind("<Control-s>", lambda e: self.save_only())
        # Ctrl + / Ctrl - 运行时改字号，命名字体一改，全界面即时重绘
        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            root.bind(seq, lambda e: self.change_font_size(1))
        for seq in ("<Control-minus>", "<Control-underscore>",
                    "<Control-KP_Subtract>"):
            root.bind(seq, lambda e: self.change_font_size(-1))
        if args.folder:
            self.open_folder(args.folder)
        elif os.path.isdir(DEFAULT_RAW_DIR) and scan_images(DEFAULT_RAW_DIR):
            # 默认图片库就在项目里，启动直接载入：不弹框、不用找路径
            print(f"[载入] 默认图片库：{DEFAULT_RAW_DIR}")
            self.open_folder(DEFAULT_RAW_DIR)
        else:
            root.after(200, self.choose_folder)
        root.after(120, self._bring_to_front)

    # -- 让窗口真的显示出来 --------------------------------------------------
    def _bring_to_front(self) -> None:
        """顶到最前、抢一次焦点，并把窗口状态写进日志。

        WSLg 下窗口偶尔开在别的窗口后面（尤其从终端启动、或开在第二个显示器
        的负坐标区域），表现为「终端在跑但看不到窗口」。这里显式 deiconify +
        lift + 短暂 topmost；0.6 秒后撤掉 topmost，不长时间挡别人的窗口。
        """
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(600, self._drop_topmost)
            self.root.focus_force()
        except Exception:
            pass
        try:
            _write_run_log("启动", "DISPLAY=%s WAYLAND_DISPLAY=%s"
                                   % (os.environ.get("DISPLAY", ""),
                                      os.environ.get("WAYLAND_DISPLAY", "")))
            _write_run_log("窗口", "viewable=%s mapped=%s geometry=%s"
                                   % (self.root.winfo_viewable(),
                                      self.root.winfo_ismapped(),
                                      self.root.winfo_geometry()))
            self.root.after(1500, self._log_window_state)
        except Exception:
            pass

    def _drop_topmost(self) -> None:
        try:
            self.root.attributes("-topmost", False)
        except Exception:
            pass

    def _log_window_state(self) -> None:
        """窗口映射 1.5 秒后再记一次：这时 viewable 才是准的。"""
        try:
            _write_run_log("窗口",
                           "1.5s: viewable=%s mapped=%s geometry=%s focus=%s"
                           % (self.root.winfo_viewable(),
                              self.root.winfo_ismapped(),
                              self.root.winfo_geometry(),
                              self.root.focus_displayof()))
        except Exception:
            pass

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title("M2-3 矩形框标注工具")
        # 按屏幕大小自适应，字号放大后底部两行控件不会挤在一起
        sw = max(900, self.root.winfo_screenwidth())
        sh = max(650, self.root.winfo_screenheight())
        w = min(1720, int(sw * 0.94))
        h = min(1040, int(sh * 0.92))
        self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")

        fn = self.fonts["base"]      # 常规
        fb = self.fonts["bold"]      # 按钮
        fs = self.fonts["small"]     # 次要说明
        fm = self.fonts["mono"]      # 数字输入

        # 第一行：文件夹选择。Windows 原生对话框 / WSL 目录 / 直接粘贴 C:\ 路径
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(8, 4))
        tk.Button(top, text="Windows 文件夹…", font=fb, padx=10, pady=3,
                  command=self.choose_folder).pack(side="left")
        tk.Button(top, text="本机目录…", font=fn, padx=8, pady=3,
                  command=self.choose_folder_wsl).pack(side="left", padx=4)
        self.shortcuts = windows_shortcuts()
        self.shortcut_var = tk.StringVar()
        self.shortcut_combo = ttk.Combobox(
            top, textvariable=self.shortcut_var, state="readonly", width=18,
            font=fn,
            values=[n for n, _ in self.shortcuts] or ["（无快捷位置）"])
        self.shortcut_combo.set("快捷位置…")
        self.shortcut_combo.pack(side="left", padx=(10, 4))
        self.shortcut_combo.bind("<<ComboboxSelected>>", self.on_shortcut)
        self.folder_var = tk.StringVar(
            value="（未选文件夹；可直接粘贴 C:\\Users\\...\\照片 后回车）")
        self.folder_entry = tk.Entry(top, textvariable=self.folder_var, font=fn)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(top, text="打开", font=fb, padx=10, pady=2,
                  command=self.open_folder_from_entry).pack(side="left", padx=2)

        # 第二行：类别与图像数组设置
        cfg = tk.Frame(self.root)
        cfg.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(cfg, text="新框标签", font=fn).pack(side="left")
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(
            cfg, textvariable=self.class_var, state="readonly", width=18,
            font=fn, values=list(self.classes))
        self.class_combo.current(0)
        self.class_combo.pack(side="left", padx=5)
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_selected)
        tk.Button(cfg, text="标签…", font=fb, padx=8, pady=2,
                  command=self.edit_classes).pack(side="left", padx=2)
        tk.Label(cfg, text="图像数组", font=fn).pack(side="left", padx=(14, 0))
        self.mode_var = tk.StringVar(value=self.image_mode)
        mode_combo = ttk.Combobox(cfg, textvariable=self.mode_var,
                                  state="readonly", width=14, font=fn,
                                  values=["crop", "none", "full"])
        mode_combo.pack(side="left", padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_selected)
        tk.Label(cfg, text="（照片在 Windows 盘上时建议选 none，JSON 更小、保存更快）",
                 font=fs).pack(side="left", padx=10)

        # 单独一行醒目显示输出文件完整路径：JSON 固定写在 M2-3 目录下，
        # 标定完的原图与 YOLO txt 分别落在 data/raw、data/labels（题目 §9）。
        out_bar = tk.Frame(self.root, relief="groove", bd=1)
        out_bar.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(out_bar, text="输出 JSON：", font=fb).pack(side="left", padx=6)
        self.output_var = tk.StringVar(value=self.display_path(DEFAULT_OUTPUT))
        tk.Label(out_bar, textvariable=self.output_var, font=fn,
                 fg="#185FA5", anchor="w").pack(
                     side="left", fill="x", expand=True, padx=4)
        tk.Button(out_bar, text="复制路径 / 打开文件夹", font=fn,
                  padx=8, pady=2,
                  command=self.open_output_dir).pack(side="right", padx=6, pady=2)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = tk.Frame(main, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text="图片列表（[x] 已标注）", font=fb).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=28, activestyle="dotbox", font=fn)
        scroll = tk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True, pady=4)
        scroll.pack(side="right", fill="y", pady=4)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        right = tk.Frame(main, relief="sunken", bd=1)
        right.pack(side="right", fill="both", expand=True)
        self.view = CanvasView(right, self.classes)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=6)
        row1 = tk.Frame(bottom)
        row1.pack(fill="x")
        tk.Button(row1, text="← 上一张", font=fb, padx=8, pady=2,
                  command=lambda: self.goto(self.index - 1)).pack(side="left")
        tk.Button(row1, text="下一张 →", font=fb, padx=8, pady=2,
                  command=lambda: self.goto(self.index + 1)).pack(side="left", padx=5)
        tk.Button(row1, text="下一张未标注", font=fb, padx=8, pady=2,
                  command=lambda: self.goto_next_unlabeled(self.index)).pack(side="left")
        tk.Button(row1, text="适应窗口", font=fb, padx=8, pady=2,
                  command=self.view.fit).pack(side="left", padx=10)
        tk.Label(row1, text="微调步长(px)", font=fn).pack(side="left", padx=(14, 0))
        self.step_var = tk.StringVar(value=str(self.step))
        tk.Entry(row1, textvariable=self.step_var, width=7,
                 font=fm).pack(side="left", padx=5)
        tk.Label(row1, text="选新框标签再拖框（不改已有框）· 数字键改选中框 · "
                 "方向键微调 · Ctrl +/- 调字号",
                 font=fs).pack(side="left", padx=6)

        row2 = tk.Frame(bottom)
        row2.pack(fill="x", pady=6)
        tk.Label(row2, text="选中框 x1 y1 x2 y2", font=fn).pack(side="left")
        for _ in range(4):
            var = tk.StringVar(value="0.000")
            ent = tk.Entry(row2, textvariable=var, width=9, font=fm)
            ent.pack(side="left", padx=3)
            ent.bind("<Return>", lambda e: (self.apply_coords(), "break")[1])
            self.coord_vars.append(var)
            self.coord_entries.append(ent)
        tk.Button(row2, text="应用坐标", font=fb, padx=8, pady=2,
                  command=self.apply_coords).pack(side="left", padx=6)
        tk.Button(row2, text="新建框(n)", font=fb, padx=8, pady=2,
                  command=self.view.new_box).pack(side="left", padx=(14, 0))
        tk.Button(row2, text="删除框(Del)", font=fb, padx=8, pady=2,
                  command=self.view.delete_box).pack(side="left", padx=5)
        self.confirm_btn = tk.Button(
            row2, text="确定 / Enter  保存并下一张", font=fb,
            command=self.confirm, bg="#185FA5", fg="#FFFFFF",
            activebackground="#2A75BD", activeforeground="#FFFFFF",
            padx=16, pady=4)
        self.confirm_btn.pack(side="right")

        self.status_var = tk.StringVar(value="请选择照片文件夹")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 font=fn, relief="groove", bd=1, padx=8, pady=3).pack(
                     fill="x", side="bottom")

    # -- 文件夹 / 载入 -------------------------------------------------------
    def choose_folder(self) -> None:
        """默认入口：优先 Windows 原生对话框，失败才回退到 WSL 的 Tk 对话框。

        用户主动点「取消」时什么都不做——绝不在取消后再弹第二个框。
        """
        mode = getattr(self.args, "dialog", "auto")
        if mode == "wsl":
            self.choose_folder_wsl()
            return
        # 起点若是 WSL 本机目录（/home/...），Windows 原生框只能在「此电脑」的
        # 盘符树里走、看不到 \\wsl$，弹出来也选不到目标，直接换 Tk 对话框。
        initial = self.folder or last_folder() or self.default_wsl_initial()
        if ON_WSL and mode == "auto" and is_wsl_native(initial):
            self.choose_folder_wsl(initial)
            return
        if ON_WSL or os.name == "nt":
            path, state = pick_windows_folder(initial)
            if path:
                self.open_folder(path)
                return
            if state == CANCELLED:
                return
            print(f"[Windows 对话框] {state}", file=sys.stderr)
            if mode == "windows":
                messagebox.showwarning(
                    "Windows 文件夹对话框不可用",
                    f"{state}\n\nWindows 原生框看不到 WSL 本机目录（\\\\wsl$）。"
                    "选 WSL 里的照片请点「本机目录…」；"
                    "选 Windows 盘请把 C:\\ 路径直接粘到顶部输入框后按回车。")
                return
            self.status_var.set(
                f"Windows 对话框不可用（{state}），已改用本机目录对话框")
        self.choose_folder_wsl()

    def choose_folder_wsl(self, initial: Optional[str] = None) -> None:
        """Tk 原生对话框：WSL 本机目录与 /mnt/c 都能浏览，起点默认落在 data/raw。

        与 Windows 原生框的关键差别：这个框走的是 Linux 文件系统，能进
        /home/xiaolingxiaotan/...，所以选本机图片库只能靠它。
        """
        cands = [initial, self.folder, last_folder(), self.default_wsl_initial()]
        start = next((p for p in cands if p and os.path.isdir(p)), None) or os.getcwd()
        d = filedialog.askdirectory(
            title="选择照片文件夹（WSL 本机 /home/... 与 Windows 盘 /mnt/c 都能浏览）",
            initialdir=start)
        if d:
            self.open_folder(d)

    @staticmethod
    def default_wsl_initial() -> str:
        """本机对话框的默认起点：data/raw 有图就给它，否则给 M2-3 项目目录。"""
        if os.path.isdir(DEFAULT_RAW_DIR) and scan_images(DEFAULT_RAW_DIR):
            return DEFAULT_RAW_DIR
        if os.path.isdir(PROJECT_DIR):
            return PROJECT_DIR
        return windows_default_initial()

    def open_folder_from_entry(self) -> None:
        """顶部输入框支持直接粘贴 C:\\ 路径，回车/点「打开」即载入。"""
        text = self.folder_var.get().strip()
        if not text or text.startswith("（"):
            self.choose_folder()
            return
        self.open_folder(text)

    def on_shortcut(self, event) -> None:
        i = self.shortcut_combo.current()
        if 0 <= i < len(self.shortcuts):
            self.open_folder(self.shortcuts[i][1])

    @staticmethod
    def display_path(path: str) -> str:
        """输出栏同时给出 Windows 形式，方便在资源管理器里找 JSON。"""
        if is_windows_drive(path):
            return f"{path}    （Windows: {local_to_windows(path)}）"
        return path

    def open_folder(self, folder: str) -> None:
        folder, jump = resolve_folder_arg(folder)
        if not os.path.isdir(folder):
            messagebox.showerror(
                "文件夹不存在",
                f"找不到目录：\n{folder}\n\n"
                "Windows 路径可直接粘贴，例如 C:\\Users\\...\\照片\n"
                "Windows 盘在 WSL 里写成 /mnt/c/Users/.../照片\n"
                "WSL 本机目录直接粘贴 /home/xiaolingxiaotan/... 或 ~/...")
            return
        files = scan_images(folder)
        if not files:
            messagebox.showwarning(
                "没有图片",
                f"{folder}\n\n该文件夹里没找到图片。支持格式："
                + " ".join(sorted(IMAGE_EXTS)))
            return
        self.folder = folder
        self.files = files
        self.folder_var.set(folder)      # 输入框里只放纯路径，便于再次编辑
        remember_folder(folder)
        out_arg = (resolve_folder_arg(self.args.output)[0] if self.args.output
                   else DEFAULT_OUTPUT)
        out = os.path.abspath(out_arg)
        yolo_dir = self.args.yolo_dir or DEFAULT_LABELS_DIR
        # 题目 §9 的三个数据目录，开跑即建齐
        for d in (DEFAULT_RAW_DIR, DEFAULT_LABELS_DIR, DEFAULT_SPLITS_DIR):
            os.makedirs(d, exist_ok=True)
        self.output_var.set(self.display_path(out))
        self.store = AnnotationStore(out, self.classes, self.image_mode,
                                     yolo_dir, DEFAULT_RAW_DIR)
        print(f"[输出] 标注 JSON：{out}")
        print(f"[输出] YOLO txt：{yolo_dir}")
        print(f"[输出] 带框复核图：{os.path.join(yolo_dir, 'preview')}"
              f"（每保存一张出一张）")
        print(f"[输出] 原始图片：{DEFAULT_RAW_DIR}（源已在 raw 则跳过复制）")
        if is_windows_drive(folder):
            print(f"[输出] Windows 路径：{local_to_windows(out)}")
        self.refresh_list()
        target = -1
        if jump:
            target = next((i for i, p in enumerate(files)
                           if os.path.basename(p) == jump), -1)
        if target >= 0:
            self.load_index(target)
        else:
            self.goto_next_unlabeled(-1)

    def refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for p in self.files:
            name = os.path.basename(p)
            mark = "[x]" if (self.store and name in self.store.items) else "[ ]"
            self.listbox.insert("end", f"{mark} {name}")

    def load_index(self, i: int) -> None:
        if not (0 <= i < len(self.files)) or self.store is None:
            return
        path = self.files[i]
        name = os.path.basename(path)
        try:
            img = read_image_array(path)
        except Exception as exc:
            messagebox.showerror("读取失败", f"{path}\n{exc}")
            return
        self.index = i
        self.img = img
        rec = self.store.items.get(name)
        boxes = array_to_boxes(rec["boxes"]) if rec else None
        self.view.load_image(img, boxes)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self.sync_coords()
        self.update_status()

    def goto(self, i: int) -> None:
        if 0 <= i < len(self.files):
            self.load_index(i)

    def goto_next_unlabeled(self, from_index: int) -> None:
        if not self.files or self.store is None:
            return
        n = len(self.files)
        for k in range(1, n + 1):
            i = (from_index + k) % n
            if os.path.basename(self.files[i]) not in self.store.items:
                self.load_index(i)
                return
        messagebox.showinfo("完成", "所有图片都已标注。点左侧列表可回去重新标定。")

    def on_list_select(self, event) -> None:
        sel = self.listbox.curselection()
        # 与当前图相同就不重载：避免 selection_set 反过来触发本事件造成递归
        if sel and int(sel[0]) != self.index:
            self.load_index(int(sel[0]))

    # -- 保存 ---------------------------------------------------------------
    def confirm(self) -> None:
        if self.img is None or self.store is None or self.index < 0:
            return
        h, w = self.img.shape[:2]
        boxes = [clamp_box(b.normalized(), w, h) for b in self.view.boxes]
        boxes = [b for b in boxes if b.valid()]
        if not boxes:
            messagebox.showwarning(
                "还没有框",
                "开局不会有默认框。请先选好标签，在图上空白处按住左键拖出一个框，\n"
                "或按 n 在中心新建一个；一张图可以拖多个不同标签的框。")
            return
        name = os.path.basename(self.files[self.index])
        self.store.put(name, self.files[self.index], self.img, boxes)
        size = self.store.save()
        print(f"[保存] {name} -> {os.path.abspath(self.store.path)} "
              f"({len(boxes)} 框，{size / 1024:.1f} KB)")
        if self.store.raw_dir:
            print(f"[导入] 原图 -> {os.path.join(self.store.raw_dir, name)} "
                  f"（同名覆盖 / 新图追加）")
        if self.store.last_preview:
            print(f"[复核图] {self.store.last_preview}（带框，一眼看框贴不贴边）")
        self.refresh_list()
        self.last_save = (f"刚保存 {name}（{len(boxes)} 框，"
                          f"{size / 1024:.1f} KB）")
        self.goto_next_unlabeled(self.index)

    def save_only(self) -> None:
        if self.store is None:
            return
        size = self.store.save()
        self.status_var.set(f"已保存 {self.store.path} ({size / 1024:.1f} KB)")

    def open_output_dir(self) -> None:
        """把 JSON 路径（能直接粘进资源管理器的 Windows 形式）放进剪贴板并定位。"""
        if self.store is None:
            messagebox.showinfo("输出路径", "请先选择照片文件夹。")
            return
        path = os.path.abspath(self.store.path)
        d = os.path.dirname(path)
        win_path = local_to_windows(path)
        clip = win_path if (ON_WSL or os.name == "nt") else path
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(clip)
        except Exception:
            pass
        opened = False
        explorer = ("/mnt/c/Windows/explorer.exe" if ON_WSL
                    else (shutil.which("explorer.exe") if os.name == "nt" else None))
        if explorer and os.path.isfile(explorer) and os.path.isdir(d):
            try:
                subprocess.Popen([explorer, "/select," + win_path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                opened = True
            except Exception:
                pass
        if not opened:
            opener = shutil.which("xdg-open")
            if opener and os.path.isdir(d):
                try:
                    subprocess.Popen([opener, d], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    opened = True
                except Exception:
                    pass
        if opened:
            self.status_var.set(f"路径已复制，并在资源管理器中定位：{clip}")
        else:
            messagebox.showinfo(
                "输出 JSON 路径（已复制到剪贴板）",
                f"{clip}\n\nJSON 固定写在 M2-3 目录下；标定完的原图在 "
                f"data/raw，YOLO txt 在 data/labels。")
            self.status_var.set(f"路径已复制到剪贴板：{clip}")

    def on_close(self) -> None:
        """关闭窗口前兜底保存，并在终端打印完整路径。"""
        if self.store is not None and self.store.items:
            try:
                size = self.store.save()
                print(f"[保存] {os.path.abspath(self.store.path)} "
                      f"({len(self.store.items)} 条，{size / 1024:.1f} KB)")
            except Exception as exc:
                print(f"[保存失败] {exc!r}", file=sys.stderr)
        self.root.destroy()

    # -- 输入 / 状态 ---------------------------------------------------------
    def on_return(self, event):
        w = self.root.focus_get()
        if w is self.folder_entry:
            self.open_folder_from_entry()
        elif w in self.coord_entries:
            self.apply_coords()
        else:
            self.confirm()
        return "break"

    def on_key(self, event) -> None:
        w = self.root.focus_get()
        if isinstance(w, tk.Entry) or (ttk and isinstance(w, ttk.Combobox)):
            return
        key = event.keysym
        if key in ("Left", "Right", "Up", "Down"):
            try:
                step = float(self.step_var.get())
            except ValueError:
                step = self.step
            if event.state & 0x0001:      # Shift
                step *= 10
            if event.state & 0x0004:      # Ctrl
                step /= 10
            dx = -step if key == "Left" else (step if key == "Right" else 0.0)
            dy = -step if key == "Up" else (step if key == "Down" else 0.0)
            self.view.nudge(dx, dy)
            self.sync_coords()
            return "break"
        if key in ("Delete", "BackSpace"):
            self.view.delete_box(); return "break"
        if key == "n":
            self.view.new_box(); return "break"
        if key == "f":
            self.view.fit(); return "break"
        if key in ("Prior", "Next"):
            self.goto(self.index - 1 if key == "Prior" else self.index + 1)
            return "break"
        if key.isdigit() and 0 <= int(key) < len(self.classes):
            cls = int(key)
            if self.view.boxes:
                self.view.set_class(cls)       # 有框：只改选中框
            else:
                self.view.set_pen_class(cls)   # 无框：预设新框标签
                self.class_combo.current(cls)
            self.update_status()
            return "break"
        return None

    def on_class_selected(self, event) -> None:
        # 下拉 = 新框标签（画笔）。只影响之后新建的框，不改已有框，
        # 否则每换一次标签就会把上一个刚画好的框改掉。
        cls = self.class_combo.current()
        if cls >= 0:
            self.view.set_pen_class(cls)
            self.update_status()

    def edit_classes(self) -> None:
        """标签设置窗：数字 ID 固定不可删，只编辑 ID 后面的名字（可留空）。"""
        win = tk.Toplevel(self.root)
        win.title("标签设置（数字 ID + 可选名字）")
        win.transient(self.root)
        fn, fb, fs = (self.fonts["base"], self.fonts["bold"],
                      self.fonts["small"])
        tk.Label(win, text="ID 是永久数字，框里存的就是它；改名不影响已标注数据。",
                 font=fs, anchor="w", justify="left").pack(
                     fill="x", padx=12, pady=(10, 4))
        tk.Label(win, text="M2-4 的 detections JSON 里 classes 顺序必须与这里一致。",
                 font=fs, anchor="w", justify="left", fg="#A32D2D").pack(
                     fill="x", padx=12, pady=(0, 8))
        grid = tk.Frame(win)
        grid.pack(fill="both", expand=True, padx=12)
        entries: List[tk.Entry] = []

        def add_row(cid: int, name: str) -> None:
            row = tk.Frame(grid)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=str(cid), font=fb, width=4,
                     anchor="w").pack(side="left")
            ent = tk.Entry(row, font=fn, width=26)
            ent.insert(0, name)
            ent.pack(side="left", padx=6)
            entries.append(ent)

        for label in self.classes:
            cid, name = split_class_label(label)
            add_row(cid if cid >= 0 else len(entries), name)

        def add_class() -> None:
            add_row(len(entries), "")

        def on_save() -> None:
            names = [e.get().strip() for e in entries]
            labels = [f"{i} {n}".rstrip() if n else str(i)
                      for i, n in enumerate(names)]
            self.apply_class_specs(labels)
            win.destroy()

        bar = tk.Frame(win)
        bar.pack(fill="x", padx=12, pady=10)
        tk.Button(bar, text="+ 添加一类", font=fb, padx=8, pady=3,
                  command=add_class).pack(side="left")
        tk.Label(bar, text="不提供删除：删 ID 会让已标注框类别失效",
                 font=fs, fg="#5F5E5A").pack(side="left", padx=10)
        tk.Button(bar, text="保存", font=fb, padx=16, pady=3,
                  command=on_save, bg="#185FA5", fg="#FFFFFF").pack(side="right")
        win.grab_set()

    def apply_class_specs(self, labels: List[str]) -> None:
        """应用新标签：刷新下拉、画布、JSON 元信息，并写回 classes.txt。"""
        labels = list(labels) or ["0"]
        self.classes = labels
        self.view.classes = labels
        self.class_combo.configure(values=labels)
        pen = max(0, min(self.view.pen_class, len(labels) - 1))
        self.view.pen_class = pen
        self.class_combo.current(pen)
        if self.store is not None:
            self.store.classes = labels
            self.store.save()
        path = self.args.classes or (
            os.path.join(
                os.path.dirname(os.path.abspath(self.store.path)),
                "classes.txt") if self.store else None)
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)),
                            exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for label in labels:
                        cid, name = split_class_label(label)
                        f.write(f"{cid} {name}".rstrip() + "\n")
                print(f"[标签] 已写入 {os.path.abspath(path)}")
            except OSError as exc:
                print(f"[标签] 写入失败：{exc!r}", file=sys.stderr)
        self.view.draw_boxes()
        self.update_status()

    def on_mode_selected(self, event) -> None:
        self.image_mode = self.mode_var.get()
        if self.store:
            self.store.image_mode = self.image_mode
        if self.image_mode == "full":
            messagebox.showwarning(
                "文件会很大",
                "整张图像素数组写进 JSON：1920×1080 约 20 MB/张，"
                "720 张会到十几 GB。一般请用 crop。")
        self.update_status()

    def apply_coords(self) -> None:
        try:
            vals = [float(v.get()) for v in self.coord_vars]
        except ValueError:
            messagebox.showwarning("输入错误", "四个坐标都要是数字，可带小数。")
            return
        self.view.set_coords(*vals)
        self.sync_coords()

    def sync_coords(self) -> None:
        boxes = self.view.boxes
        if 0 <= self.view.selected < len(boxes):
            b = boxes[self.view.selected]
            for var, val in zip(self.coord_vars, (b.x1, b.y1, b.x2, b.y2)):
                var.set(f"{val:.3f}")
        else:
            for var in self.coord_vars:
                var.set("")
        # 下拉恒为「新建框用的标签」，点已有框不覆盖它；
        # 选中框的类别看框上的数字标签和状态栏 cls=。

    def on_view_change(self) -> None:
        self.sync_coords()
        self.update_status()

    def update_status(self) -> None:
        if self.img is None or self.store is None:
            return
        h, w = self.img.shape[:2]
        boxes = self.view.boxes
        sel = ""
        if 0 <= self.view.selected < len(boxes):
            b = boxes[self.view.selected]
            sel = (f" | 选中 cls={b.cls} "
                   f"[{b.x1:.3f}, {b.y1:.3f}, {b.x2:.3f}, {b.y2:.3f}]")
        done = sum(1 for p in self.files
                   if os.path.basename(p) in self.store.items)
        self.status_var.set(
            f"第 {self.index + 1}/{len(self.files)} 张 | {w}×{h} | "
            f"zoom {self.vt_zoom():.2f}× | 框 {len(boxes)} | "
            f"新框标签 {self.view.pen_class}{sel} | "
            f"已标 {done}/{len(self.files)} | 图像数组={self.image_mode} | "
            f"字体 {self.ui_family} {self.ui_size}pt | "
            f"JSON: {self.store.path}"
            + (f" | {self.last_save}" if self.last_save else ""))

    def change_font_size(self, delta: int) -> str:
        """Ctrl +/-：即时缩放全界面字号，不影响任何标注坐标。"""
        self.ui_size = apply_ui_fonts(self.root, self.fonts,
                                      self.ui_family, self.ui_size + delta)
        self.view.label_font = self.fonts.get("label")
        self.view.draw_boxes()
        self.update_status()
        return "break"

    def vt_zoom(self) -> float:
        return self.view.vt.zoom


# ---------------------------------------------------------------------------
# 自检（不依赖界面，可在无 DISPLAY 环境运行）
# ---------------------------------------------------------------------------

def self_test() -> bool:
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (("  " + extra) if extra else ""))
        if not cond:
            ok = False

    # 1. 坐标变换
    vt = ViewTransform()
    vt.fit(1920, 1080, 960, 540)
    check("fit 缩放比", abs(vt.zoom - 0.5) < 1e-9, f"zoom={vt.zoom}")
    sx, sy = vt.to_screen(240.25, 300.75)
    ix, iy = vt.to_image(sx, sy)
    check("坐标往返（含小数）", abs(ix - 240.25) < 1e-9 and abs(iy - 300.75) < 1e-9)
    anchor = vt.to_image(400.0, 200.0)
    vt.zoom_at(400.0, 200.0, 2.0)
    anchor2 = vt.to_image(400.0, 200.0)
    check("缩放锚点不变", abs(anchor[0] - anchor2[0]) < 1e-9
          and abs(anchor[1] - anchor2[1]) < 1e-9)

    # 2. 默认框 / clamp / 拖动
    cb = make_center_box(1920, 1080)
    check("中心框居中", abs((cb.x1 + cb.x2) / 2 - 960) < 1e-9
          and abs((cb.y1 + cb.y2) / 2 - 540) < 1e-9)
    cb2 = clamp_box(Box(-5.5, -2.0, 2000.0, 1200.0), 1920, 1080)
    check("越界裁剪", cb2.x1 == 0.0 and cb2.x2 == 1920.0 and cb2.y2 == 1080.0)
    moved = apply_drag(Box(10, 20, 30, 40, 2), "e", 0.125, 0.0)
    check("亚像素拖动", abs(moved.x2 - 30.125) < 1e-9, f"x2={moved.x2}")
    moved2 = apply_drag(Box(10, 20, 30, 40, 0), "move", 0.1, -0.1)
    check("整体移动", abs(moved2.x1 - 10.1) < 1e-9 and abs(moved2.y1 - 19.9) < 1e-9)

    # 3. 命中测试
    vt3 = ViewTransform(1.0, 0.0, 0.0)
    b3 = Box(100, 100, 300, 300, 0)
    check("命中内部=move", hit_test_mode([b3], 200, 200, vt3)[1] == "move")
    check("命中左边=w", hit_test_mode([b3], 100, 200, vt3)[1] == "w")
    check("命中左上角=nw", hit_test_mode([b3], 100, 100, vt3)[1] == "nw")
    check("命中空白=create", hit_test_mode([b3], 900, 900, vt3)[1] == "create")

    # 4. numpy 往返
    boxes = [Box(240.125, 300.5, 680.75, 760.25, 0),
             Box(1100.0, 420.0, 1290.0, 610.0, 2)]
    arr = boxes_to_array(boxes)
    back = array_to_boxes(arr.tolist())
    check("numpy 形状", arr.shape == (2, 5), str(arr.shape))
    check("numpy 往返保精度", abs(back[0].x1 - 240.125) < 1e-9
          and back[1].cls == 2)
    check("空框数组形状", boxes_to_array([]).shape == (0, 5))
    yl = yolo_line(boxes[0], 1920, 1080)
    yp = yl.split()
    check("YOLO 归一化",
          yp[0] == "0"
          and abs(float(yp[1]) - 460.4375 / 1920) < 1e-6
          and abs(float(yp[3]) - 440.625 / 1920) < 1e-6, yl)

    # 4b. 类别：数字 ID + 可选名字
    check("默认标签是纯数字",
          DEFAULT_CLASSES == ["0", "1", "2", "3", "4", "5"])
    specs = parse_class_lines([
        "# 注释", "0 书", "1", "2:笔", "3，杯子", "4=保温杯", "5 草稿本"])
    check("类别写法解析", specs[0] == "0 书" and specs[1] == "1"
          and specs[2] == "2 笔" and specs[3] == "3 杯子"
          and specs[4] == "4 保温杯" and specs[5] == "5 草稿本", str(specs))
    check("旧格式自动编号",
          parse_class_lines(["书", "笔", "杯子"]) == ["0 书", "1 笔", "2 杯子"])
    sparse = parse_class_lines(["0 书", "3 笔"])
    check("跳号补齐下标=ID", len(sparse) == 4 and sparse[1] == "1"
          and sparse[3] == "3 笔", str(sparse))
    check("显示串拆分", split_class_label("3 杯子") == (3, "杯子")
          and split_class_label("4") == (4, ""))
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "classes.txt")
        with open(cp, "w", encoding="utf-8") as f:
            f.write("0 书\n1 笔\n2 杯子\n3 保温杯\n4 吧唧\n5 草稿本\n")
        check("classes.txt 读入", load_classes(cp) == [
            "0 书", "1 笔", "2 杯子", "3 保温杯", "4 吧唧", "5 草稿本"])
    bx = Box(1, 2, 3, 4, 4)
    check("改名不影响框的 ID", int(bx.cls) == 4
          and split_class_label(parse_class_lines(
              ["0", "1", "2", "3", "4 新名字", "5"])[4])[1] == "新名字")

    # 4.5 Windows / WSL 跨系统路径与图片扫描（不弹任何对话框）
    if ON_WSL:
        check("Windows 盘符→WSL 路径",
              windows_to_local(r"C:\Users\me\照片") == "/mnt/c/Users/me/照片",
              windows_to_local(r"C:\Users\me\照片"))
        check("正斜杠 / 大小写盘符",
              windows_to_local("G:/data/IMG_001.JPG") == "/mnt/g/data/IMG_001.JPG",
              windows_to_local("G:/data/IMG_001.JPG"))
        check("带引号的复制地址",
              windows_to_local('"C:\\Users\\me\\照片"') == "/mnt/c/Users/me/照片",
              windows_to_local('"C:\\Users\\me\\照片"'))
        check("Win32 长路径前缀",
              windows_to_local(r"\\?\C:\Users\me\照片")
              == "/mnt/c/Users/me/照片")
        check("UNC wsl$ → 本机路径",
              windows_to_local(r"\\wsl$\Ubuntu2204\home\me\photos")
              == "/home/me/photos")
        check("反向换算成 C:\\ 形式",
              local_to_windows("/mnt/c/Users/me/照片")
              == "C:\\Users\\me\\照片")
        check("识别 Windows 盘路径",
              is_windows_drive("/mnt/f/x") and not is_windows_drive("/home/me/x"))
        check("能定位 powershell.exe", bool(_powershell_exe()),
              str(_powershell_exe()))
    else:
        print("SKIP  Windows 盘符换算（当前不在 WSL）")
    check("WSL 本地路径原样保留",
          windows_to_local("~/photos")
          == os.path.abspath(os.path.expanduser("~/photos")))
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "子目录"))
        os.makedirs(os.path.join(td, "fake.jpg"))     # 同名目录不能混进图片列表
        for name in ("a.JPG", "b.png", "c.txt", "d.jpeg"):
            with open(os.path.join(td, name), "wb") as f:
                f.write(b"x")
        got = [os.path.basename(p) for p in scan_images(td)]
        check("扫描图片：大小写扩展名 + 忽略非图片/目录",
              got == ["a.JPG", "b.png", "d.jpeg"], str(got))
        check("传图片路径→目录 + 跳转文件名",
              resolve_folder_arg(os.path.join(td, "a.JPG")) == (td, "a.JPG"),
              str(resolve_folder_arg(os.path.join(td, "a.JPG"))))
    sc = windows_shortcuts()
    check("快捷位置都是真实目录", all(os.path.isdir(p) for _, p in sc),
          str([n for n, _ in sc])[:120])

    # 5. JSON 存储 + 图像数组
    with tempfile.TemporaryDirectory() as td:
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        img[10:40, 20:60] = (200, 30, 30)
        p = os.path.join(td, "sample.png")
        Image.fromarray(img).save(p)
        arr_read = read_image_array(p)
        check("读图形状/dtype", arr_read.shape == (60, 80, 3)
              and arr_read.dtype == np.uint8)
        out = os.path.join(td, "annotations.json")
        ydir = os.path.join(td, "labels")
        store = AnnotationStore(out, DEFAULT_CLASSES, "crop", ydir)
        store.put("sample.png", p, arr_read,
                  [Box(20.5, 10.25, 60.0, 40.0, 0)])
        size = store.save()
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        item = doc["items"][0]
        bb = np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 5)
        crop = np.asarray(item["crops"][0], dtype=np.uint8)
        check("JSON 可写可读", size > 0 and doc["count"] == 1)
        check("JSON 数字类别 ID 与名字",
              doc["class_ids"] == list(range(len(DEFAULT_CLASSES)))
              and doc["class_names"][0] == "")
        check("JSON 框含小数", abs(bb[0, 0] - 20.5) < 1e-9)
        check("JSON 框内图像数组", crop.shape[0] > 0 and crop.shape[2] == 3,
              str(crop.shape))
        check("图像元信息", item["image"]["shape"] == [60, 80, 3]
              and item["image"]["data"] is None)
        check("YOLO txt 同步写出", os.path.isfile(os.path.join(ydir, "sample.txt")))
        store2 = AnnotationStore(out, DEFAULT_CLASSES, "crop", ydir)
        check("重开可恢复记录", "sample.png" in store2.items)

        # 5b. 题目 §9 落盘：原图导入 data/raw（同名覆盖 / 新图追加）
        rawdir = os.path.join(td, "raw")
        p2 = os.path.join(td, "second.png")
        Image.fromarray(img).save(p2)
        st = AnnotationStore(os.path.join(td, "b.json"), DEFAULT_CLASSES,
                             "none", None, rawdir)
        st.put("sample.png", p, arr_read, [Box(1, 1, 20, 20, 0)])
        check("标定完图片导入 data/raw",
              os.path.isfile(os.path.join(rawdir, "sample.png")))
        st.put("second.png", p2, arr_read, [Box(2, 2, 30, 30, 1)])
        check("不同图片追加", len(os.listdir(rawdir)) == 2)
        st.put("sample.png", p, arr_read, [Box(3, 3, 40, 40, 2)])
        check("同名重标覆盖不新增文件", len(os.listdir(rawdir)) == 2)
        try:
            st.put("sample.png", os.path.join(rawdir, "sample.png"),
                   arr_read, [Box(4, 4, 50, 50, 3)])
            check("源已在 data/raw 不报 SameFileError", True)
        except Exception as exc:
            check("源已在 data/raw 不报 SameFileError", False, str(exc))
        check("产物默认固定在 M2-3 目录",
              DEFAULT_OUTPUT == os.path.join(PROJECT_DIR, "annotations.json")
              and DEFAULT_RAW_DIR == os.path.join(PROJECT_DIR, "data", "raw")
              and DEFAULT_LABELS_DIR == os.path.join(PROJECT_DIR, "data", "labels"))
        store2.put("sample.png", p, arr_read,
                   [Box(21.0, 11.0, 59.0, 39.0, 1)])
        store2.save()
        check("重标覆盖而非追加", len(store2.items) == 1)

    # 6. GUI 渲染链路（有显示环境才跑）
    if tk is not None and (os.environ.get("DISPLAY") or os.name == "nt"):
        try:
            root = tk.Tk()
            root.withdraw()
            fonts: Dict[str, object] = {}
            fam = pick_cjk_family(root)
            apply_ui_fonts(root, fonts, fam, DEFAULT_UI_SIZE)
            frame = tk.Frame(root)
            frame.pack()
            view = CanvasView(frame, DEFAULT_CLASSES)
            view.label_font = fonts.get("label")
            view.view_w, view.view_h = 640, 480
            test_img = np.zeros((120, 160, 3), dtype=np.uint8)
            view.load_image(test_img, [Box(10.5, 10.5, 60.25, 50.75, 0)])
            view.zoom_by(100, 100, 4.0)
            view.render_all()
            root.update_idletasks()
            check("GUI 渲染不报错", True)
            check("界面中文字体可配置", bool(fonts) and
                  fonts["base"].actual("family") == fam, fam)
            view.load_image(test_img, None)
            check("开局无默认框", len(view.boxes) == 0)
            view.set_pen_class(3)
            view.new_box()
            view.set_pen_class(5)
            check("换画笔不改已有框", int(view.boxes[0].cls) == 3)
            view.new_box()
            check("一张图可标多种标签", len(view.boxes) == 2
                  and [b.cls for b in view.boxes] == [3, 5])
            check("框标签以数字 ID 开头", view.class_label(3).startswith("3")
                  and view.class_label(5).startswith("5"))
            view.delete_box()
            view.delete_box()
            check("删空后不补默认框", len(view.boxes) == 0)
            view.render_all()
            print(f"[字体] 当前界面字体：{fam} {DEFAULT_UI_SIZE}pt")
            root.destroy()
        except Exception as exc:
            check("GUI 渲染不报错", False, repr(exc))
    else:
        print("SKIP  GUI 渲染（无显示环境）")

    print("\n=== " + ("ALL PASS" if ok else "SOME TESTS FAILED") + " ===")
    return ok


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="M2-3 矩形框标注工具（亚像素、滚轮缩放、中键平移、JSON/numpy 输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder",
                        help="照片所在文件夹（支持 C:\\Users\\...\\照片、/mnt/c/...、"
                             "~/...）；不填则启动后弹 Windows 原生选择框")
    parser.add_argument("--output",
                        help="JSON 输出路径（同样支持 Windows 路径）；"
                             "默认 M2-3/annotations.json（固定，不随照片文件夹变）")
    parser.add_argument("--dialog", choices=["auto", "windows", "wsl"],
                        default="auto",
                        help="文件夹选择框：auto=按起始路径自动选（WSL 本机目录用 Tk，"
                             "Windows 盘用原生框，失败回退）；windows=只用 Windows 原生；"
                             "wsl=只用 Tk（能浏览 /home 与 /mnt/c）")
    parser.add_argument("--classes", default=default_classes_file(),
                        help="标签文件：每行一类，可写 '0' / '0 书' / '0:书'；"
                             "下标即类别 ID，默认用项目里的 classes.txt")
    parser.add_argument("--image-mode", choices=["crop", "none", "full"],
                        default="crop", help="图像数组写入方式，默认 crop（只存框内区域）")
    parser.add_argument("--yolo-dir",
                        help="同步写 YOLO txt 的目录；默认 M2-3/data/labels/")
    parser.add_argument("--step", type=float, default=0.1,
                        help="方向键微调步长（px），默认 0.1")
    parser.add_argument("--font-size", type=int, default=DEFAULT_UI_SIZE,
                        help=f"界面字号 pt，默认 {DEFAULT_UI_SIZE}；"
                             f"运行时可用 Ctrl +/- 调整")
    parser.add_argument("--font-family", default=None,
                        help="手动指定字体族（中文仍是空白时用，"
                             "例如 SimHei / Noto Sans SC / Microsoft YaHei）")
    parser.add_argument("--self-test", action="store_true",
                        help="不弹界面，自检核心逻辑")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    # 必须在创建 Tk 窗口之前：Tk 启动时就缓存了可用字体列表
    ensure_cjk_font()
    if args.self_test:
        return 0 if self_test() else 1
    if tk is None:
        print("错误：缺少 tkinter。WSL 上执行 sudo apt install python3-tk", file=sys.stderr)
        return 2
    if os.name == "nt":
        try:  # Windows 高分屏，避免界面模糊
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    _write_run_log("启动", "ON_WSL=%s DISPLAY=%s WAYLAND_DISPLAY=%s python=%s"
                           % (ON_WSL, os.environ.get("DISPLAY", ""),
                              os.environ.get("WAYLAND_DISPLAY", ""),
                              sys.version.split()[0]))
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        msg = ("打不开图形界面（Tk 连不上显示服务）：%s\n\n"
               "WSL 里按顺序排查：\n"
               "  1) 用普通权限的终端启动，别用「以管理员身份运行」——WSLg 在\n"
               "     管理员会话里常常不出窗口；\n"
               "  2) Windows 11 自带 WSLg 就够用，不需要另装 VcXsrv；在 Ubuntu 里\n"
               "     执行 xeyes 能弹出小眼睛说明显示正常；\n"
               "  3) 如果 DISPLAY 被人为设成 :0，试着 unset DISPLAY 再跑一次。\n"
               "本次环境已记录到：%s" % (exc, RUN_LOG))
        print(msg, file=sys.stderr)
        _write_run_log("失败", msg)
        return 3
    try:
        AnnotatorApp(root, args)
    except Exception:
        tb = traceback.format_exc()
        print("界面初始化失败：\n" + tb, file=sys.stderr)
        _write_run_log("崩溃", tb)
        try:
            messagebox.showerror("界面初始化失败", tb[-1500:])
        except Exception:
            pass
        return 4
    root.mainloop()
    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
