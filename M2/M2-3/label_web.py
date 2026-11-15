#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2-3 浏览器标注器（label_web.py）
=================================

为什么会有这个文件
------------------
这台机器上 WSLg 退化了：msrdc 只给一个「整块桌面镜像」窗口
（标题 WARN: COPY MODE），tkinter 窗口显示不出来，也点不开。
本工具把界面搬到浏览器：Python 只起一个 HTTP 服务，界面是网页，
不依赖 WSLg / GPU / 显卡驱动，只要有浏览器就能标。

与 label_tool.py 完全共用数据和输出（可以混着用）：
    M2-3/annotations.json           同一个文件、同一个格式
    M2-3/data/raw/<文件名>          原图（本工具只读，不复制）
    M2-3/data/labels/<stem>.txt     YOLO 标注，每次保存重写
    M2-3/data/labels/preview/*.jpg  带框复核图

用法
----
    python3 label_web.py                  # 默认 8765 端口，只监听本机
    python3 label_web.py --port 9000
    python3 label_web.py --host 0.0.0.0    # 允许同一 WiFi 下的手机访问
    python3 label_web.py --folder /path    # 换一个图片目录

启动后在 Windows 浏览器打开终端里打印的地址（默认 http://localhost:8765）。

快捷键
------
    拖拽            画框（用当前选中的类别）
    点击框          选中；选中后按 Delete 删除
    0~5             有选中框则改它的类别，否则切换「新框类别」
    ← / →           上一张 / 下一张（自动保存）
    Enter           保存并跳到下一张
    S               只保存
    U               跳到下一张未标注的图
"""

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---- 路径常量 --------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "annotations.json")
DEFAULT_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
DEFAULT_LABELS_DIR = os.path.join(PROJECT_DIR, "data", "labels")
DEFAULT_PREVIEW_DIR = os.path.join(DEFAULT_LABELS_DIR, "preview")
DEFAULT_CLASSES_FILE = os.path.join(PROJECT_DIR, "classes.txt")

# 与 label_tool.py 一致的类别配色，两个工具看起来是同一套
CLASS_COLORS = ["#A32D2D", "#185FA5", "#0F6E56", "#854F0B", "#534AB7", "#993C1D"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THUMB_MAX = 160          # 左侧列表缩略图最长边
PREVIEW_MAX = 1280       # 复核图最长边（与 label_tool 一致）

_lock = threading.Lock()
_thumb_cache = {}


# ---- 基础工具 --------------------------------------------------------------
def split_class_label(text: str):
    """把 classes.txt 的一行拆成 (id, 名字)，认 `0 书` / `0: 书` / `0、书`。"""
    s = text.strip()
    m = re.match(r"^(\d+)\s*[:：.、,，]?\s*(.*)$", s)
    if not m:
        return None, s
    return m.group(1), m.group(2).strip()


def load_classes(path: str):
    """读 classes.txt：跳过 # 注释和空行，返回 [(id, name), ...] 按 id 排序。

    单行格式坏了只跳过这一行，不牵连整张表（之前踩过这个坑）。
    """
    table = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                cid, name = split_class_label(line)
                if cid is None or not cid.isdigit():
                    continue
                table[int(cid)] = name or str(cid)
    if not table:
        table = {i: str(i) for i in range(6)}
    return sorted(table.items())


def class_display(pairs):
    """界面上显示用的 `0 书` 形式。"""
    return [{"id": cid, "name": name, "label": ("%d %s" % (cid, name)).strip(),
             "color": CLASS_COLORS[cid % len(CLASS_COLORS)]} for cid, name in pairs]


def scan_images(folder: str):
    """列出目录里的图片，按文件名自然排序（IMG_2 排在 IMG_10 前面）。"""
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder)
             if n.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(folder, n))]

    def key(n):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", n)]

    return sorted(names, key=key)


def open_rgb(path: str):
    """按 EXIF 方向转正后读成 RGB —— 与标注工具、train.py 的口径一致。"""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _preview_font(size: int):
    """复核图上的中文字体，找不到就用默认（会变成方块，但不崩）。"""
    for p in ("/mnt/c/Windows/Fonts/msyh.ttc",
              "/mnt/c/Windows/Fonts/simhei.ttf",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ---- 标注存储 --------------------------------------------------------------
class Store:
    """annotations.json 的读写。格式与 label_tool.py 完全一致。"""

    def __init__(self, path: str, classes_pairs):
        self.path = path
        self.classes = ["%d %s" % (c, n) for c, n in classes_pairs]
        self.items = {}
        self._load()

    def _load(self):
        if not os.path.isfile(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            doc = json.load(f)
        for item in doc.get("items", []):
            if "file" in item and "boxes" in item:
                self.items[item["file"]] = item

    def boxes_of(self, name: str):
        item = self.items.get(name)
        if not item:
            return []
        out = []
        for b in item.get("boxes", []):
            if isinstance(b, (list, tuple)) and len(b) == 5:
                out.append([round(float(b[0]), 1), round(float(b[1]), 1),
                            round(float(b[2]), 1), round(float(b[3]), 1),
                            int(b[4])])
        return out

    def save_all(self):
        doc = {
            "format": "m2-3-bbox-annotation",
            "version": 1,
            "classes": self.classes,
            "class_ids": list(range(len(self.classes))),
            "class_names": [split_class_label(s)[1] for s in self.classes],
            "count": len(self.items),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": list(self.items.values()),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(tmp, self.path)          # 原子替换，断电不会写坏原文件

    def put(self, name: str, image_path: str, size_hw, boxes, labels_dir,
            preview_dir):
        """写一条：更新内存 -> YOLO txt -> 带框复核图 -> 整个 json 落盘。

        返回 (yolo_path, preview_path)。
        """
        h, w = int(size_hw[0]), int(size_hw[1])
        arr = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(b[4])]
               for b in boxes]
        self.items[name] = {
            "file": name,
            "image_path": os.path.abspath(image_path),
            "image": {
                "path": os.path.abspath(image_path),
                "shape": [h, w, 3],
                "dtype": "uint8",
                "order": "HWC",
                "data": None,          # 不内嵌像素，json 才不至于几百 MB
            },
            "boxes": arr,
            "boxes_shape": [len(arr), 5],
            "crops": [],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(labels_dir, exist_ok=True)
        stem = os.path.splitext(name)[0]
        yolo = os.path.join(labels_dir, stem + ".txt")
        with open(yolo, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2, cid in arr:
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = abs(x2 - x1) / float(w)
                bh = abs(y2 - y1) / float(h)
                f.write("%d %.6f %.6f %.6f %.6f\n" % (int(cid), cx, cy, bw, bh))

        prev = None
        if arr:
            prev = self._preview(name, image_path, arr, preview_dir, h, w)
        return yolo, prev

    def _preview(self, name, image_path, arr, out_dir, h, w):
        """画框存 JPEG（最长边 1280），给人工抽检用。"""
        try:
            pil = open_rgb(image_path)
            scale = min(1.0, float(PREVIEW_MAX) / float(max(h, w)))
            if scale < 1.0:
                pil = pil.resize((max(1, int(round(w * scale))),
                                  max(1, int(round(h * scale)))), Image.BOX)
            draw = ImageDraw.Draw(pil)
            side = max(pil.size)
            lw = max(2, int(round(side / 480.0)))
            font = _preview_font(max(14, int(round(side / 42.0))))
            for x1, y1, x2, y2, cid in arr:
                box = [min(x1, x2) * scale, min(y1, y2) * scale,
                       max(x1, x2) * scale, max(y1, y2) * scale]
                color = CLASS_COLORS[int(cid) % len(CLASS_COLORS)]
                draw.rectangle(box, outline=color, width=lw)
                label = split_class_label(self.classes[int(cid)])[1] \
                    if 0 <= int(cid) < len(self.classes) else str(cid)
                if font is not None and label:
                    try:
                        tb = draw.textbbox((0, 0), label, font=font)
                        tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    except Exception:
                        tw, th = len(label) * 8, 14
                    ty = box[1] - th - 4 if box[1] - th - 4 >= 0 else box[1] + 2
                    draw.rectangle([box[0], ty, box[0] + tw + 6, ty + th + 4],
                                   fill=color)
                    draw.text((box[0] + 3, ty + 1), label, fill="#FFFFFF",
                              font=font)
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg")
            pil.save(dst, quality=88)
            return dst
        except Exception as exc:      # 预览图失败不该让保存失败
            print("[预览] %s 生成失败：%s" % (name, exc), file=sys.stderr)
            return None


# ---- HTTP 服务 -------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>M2-3 标注器</title>
<style>
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:#14161a; color:#e6e8ec;
  font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
  overflow:hidden; user-select:none; }
#app { display:flex; height:100vh; }
/* 左侧列表 */
#side { width:190px; flex:none; display:flex; flex-direction:column;
  background:#1b1e24; border-right:1px solid #2a2f38; }
#side header { padding:8px 10px; font-size:12px; color:#8b93a1;
  border-bottom:1px solid #2a2f38; line-height:1.7; }
#side header b { color:#e6e8ec; font-size:13px; }
#filter { display:flex; gap:6px; padding:6px 8px; }
#filter button { flex:1; font-size:11px; padding:4px 0; cursor:pointer;
  background:#232830; color:#aeb6c2; border:1px solid #333a45; border-radius:4px; }
#filter button.on { background:#2d6cdf; color:#fff; border-color:#2d6cdf; }
#list { flex:1; overflow-y:auto; }
.row { display:flex; align-items:center; gap:6px; padding:4px 8px;
  cursor:pointer; border-left:3px solid transparent; }
.row:hover { background:#232830; }
.row.cur { background:#2a3548; border-left-color:#4a8cff; }
.row img { width:44px; height:33px; object-fit:cover; border-radius:3px;
  background:#000; flex:none; }
.row .meta { font-size:11px; color:#9aa3b0; overflow:hidden; line-height:1.35; }
.row .nm { color:#dde2ea; font-size:11px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; max-width:110px; }
.tag { display:inline-block; font-size:10px; padding:0 4px; border-radius:3px;
  margin-right:3px; }
.tag.done { background:#1f5c3a; color:#8fe3b5; }
.tag.none { background:#4a3a1c; color:#e3c98f; }
/* 右侧主区 */
#main { flex:1; display:flex; flex-direction:column; min-width:0; }
#bar { flex:none; display:flex; align-items:center; gap:10px; padding:7px 12px;
  background:#1b1e24; border-bottom:1px solid #2a2f38; font-size:13px; }
#fname { font-weight:600; color:#fff; }
#pos { color:#8b93a1; font-size:12px; }
#bar .sp { flex:1; }
#bar button { font-size:12px; padding:4px 10px; cursor:pointer; background:#232830;
  color:#cfd6e0; border:1px solid #333a45; border-radius:4px; }
#bar button:hover { background:#2d3542; }
#stage { flex:1; position:relative; overflow:hidden; background:#0d0f12; }
#cv { position:absolute; left:0; top:0; cursor:crosshair; }
#hint { position:absolute; left:12px; bottom:10px; font-size:12px;
  color:#8b93a1; background:#000a; padding:4px 8px; border-radius:4px; }
/* 类别条 */
#cls { flex:none; display:flex; gap:8px; padding:8px 12px; background:#1b1e24;
  border-top:1px solid #2a2f38; flex-wrap:wrap; align-items:center; }
.cbtn { display:flex; align-items:center; gap:6px; padding:5px 11px;
  border-radius:5px; font-size:13px; cursor:pointer; background:#232830;
  border:1px solid #333a45; color:#cfd6e0; }
.cbtn .dot { width:10px; height:10px; border-radius:2px; }
.cbtn.on { background:#2d6cdf; color:#fff; border-color:#2d6cdf; }
.cbtn small { opacity:.6; }
#clsinfo { font-size:12px; color:#8b93a1; margin-left:6px; }
#toast { position:fixed; right:16px; bottom:64px; background:#2d6cdf; color:#fff;
  padding:8px 14px; border-radius:5px; font-size:13px; opacity:0;
  transition:opacity .2s; pointer-events:none; }
#toast.show { opacity:1; }
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <header><b>图片库</b><br><span id="stat">加载中…</span></header>
    <div id="filter">
      <button class="on" data-f="all">全部</button>
      <button data-f="todo">未标</button>
      <button data-f="done">已标</button>
    </div>
    <div id="list"></div>
  </div>
  <div id="main">
    <div id="bar">
      <span id="fname">-</span><span id="pos"></span>
      <span id="zoomv">100%</span>
      <span class="sp"></span>
      <button id="bprev">← 上一张</button>
      <button id="bnext">下一张 →</button>
      <button id="breset">适应窗口</button>
      <button id="bundo">撤销框</button>
      <button id="bundoall">清空本图</button>
    </div>
    <div id="stage">
      <canvas id="cv"></canvas>
      <div id="hint">滚轮缩放 · 中键拖动整图 · 拖拽画框 · 拖框移动 / 拖边角改大小 · 方向键微调（Shift 亚像素）· ←/→ 换图</div>
    </div>
    <div id="cls"></div>
  </div>
</div>
<div id="toast"></div>
<script>
let S = { classes: [], files: [], done: {}, cur: -1, filter: 'all',
          boxes: [], sel: -1, cls: 0, img: null,
          fitScale: 1, zoom: 1, scale: 1,        // scale = fitScale * zoom
          baseOX: 0, baseOY: 0, panX: 0, panY: 0, // 居中基线 + 用户平移
          ox: 0, oy: 0,
          mode: null, panning: false,
          start: null, snap: null, handle: null };
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const stage = document.getElementById('stage');
const MIN_ZOOM = 0.2, MAX_ZOOM = 16, HANDLE_PX = 7;
const CURSORS = { n:'ns-resize', s:'ns-resize', e:'ew-resize', w:'ew-resize',
                  nw:'nwse-resize', se:'nwse-resize',
                  ne:'nesw-resize', sw:'nesw-resize' };
let dragFrom = [0, 0];

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 1400);
}

async function boot() {
  const r = await fetch('/api/state');
  const d = await r.json();
  S.classes = d.classes; S.files = d.files; S.done = d.done;
  S.cls = d.classes.length ? d.classes[0].id : 0;
  buildCls();
  renderList();
  // 优先跳到第一张未标注的
  let start = S.files.findIndex(f => !S.done[f]);
  if (start < 0) start = 0;
  if (S.files.length) load(start);
  window.addEventListener('resize', () => draw());
  // 三种时机对账：切回页面、页面重新可见、以及每 20 秒兜底一次
  window.addEventListener('focus', () => syncState(true));
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) syncState(true);
  });
  setInterval(() => syncState(true), 20000);
}

function buildCls() {
  const box = document.getElementById('cls');
  box.innerHTML = '';
  S.classes.forEach(c => {
    const b = document.createElement('div');
    b.className = 'cbtn' + (c.id === S.cls ? ' on' : '');
    b.dataset.id = c.id;
    b.innerHTML = '<span class="dot" style="background:' + c.color + '"></span>'
      + c.label + ' <small>' + c.id + '</small>';
    b.onclick = () => pickCls(c.id);
    box.appendChild(b);
  });
  const info = document.createElement('span');
  info.id = 'clsinfo';
  info.textContent = '当前新框类别：' + curClsLabel();
  box.appendChild(info);
}

function curClsLabel() {
  const c = S.classes.find(x => x.id === S.cls);
  return c ? c.label : String(S.cls);
}

function pickCls(id) {
  S.cls = id;
  document.querySelectorAll('.cbtn').forEach(e => {
    e.classList.toggle('on', Number(e.dataset.id) === id);
  });
  const info = document.getElementById('clsinfo');
  if (info) info.textContent = '当前新框类别：' + curClsLabel();
}

function renderList() {
  const list = document.getElementById('list');
  const items = S.files.filter(f => {
    if (S.filter === 'todo') return !S.done[f];
    if (S.filter === 'done') return !!S.done[f];
    return true;
  });
  updateStat();
  list.innerHTML = '';
  const frag = document.createDocumentFragment();
  items.forEach(f => {
    const i = S.files.indexOf(f);
    const row = document.createElement('div');
    row.className = 'row' + (i === S.cur ? ' cur' : '');
    row.dataset.i = i;
    const n = S.done[f] || 0;
    row.innerHTML =
      '<img loading="lazy" src="/thumb?name=' + encodeURIComponent(f) + '">' +
      '<div class="meta"><div class="nm">' + f + '</div>' +
      (n ? '<span class="tag done">' + n + ' 框</span>'
         : '<span class="tag none">未标</span>') + '</div>';
    row.onclick = () => load(i);
    frag.appendChild(row);
  });
  list.appendChild(frag);
}

function updateStat() {
  const doneCount = S.files.filter(f => S.done[f]).length;
  document.getElementById('stat').textContent =
    '已标 ' + doneCount + ' / 共 ' + S.files.length;
}

// 只更新一行右侧的「N 框 / 未标」标签，不重建整张列表。
// 以前每次保存都调 renderList()，等于把 378 行 DOM 连同 378 个缩略图请求
// 全部重来一遍，标得快的时候就会明显卡顿。
function refreshRow(i) {
  const row = document.querySelector('.row[data-i="' + i + '"]');
  if (!row) return;
  const meta = row.querySelector('.meta');
  if (!meta) return;
  const n = S.done[S.files[i]] || 0;
  const html = n ? '<span class="tag done">' + n + ' 框</span>'
                 : '<span class="tag none">未标</span>';
  const tag = meta.querySelector('.tag');
  if (tag) tag.outerHTML = html;
  else meta.insertAdjacentHTML('beforeend', html);
}

// 和服务端对账：页面开着不动、或者同时开了多个标签页时，
// 本地 S.done 会越来越旧（显示的「已标」就停住不动）。这里定期拉一次
// /api/state 把计数对齐，有差异才重绘。
async function syncState(silent) {
  if (document.hidden) return;
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const d = await r.json();
    if (!d || !d.files) return;
    let diff = 0;
    for (const f of d.files) {
      if ((d.done[f] || 0) !== (S.done[f] || 0)) diff++;
    }
    if (!diff) { S.done = d.done; updateStat(); return; }
    S.done = d.done;
    renderList();
    if (!silent) toast('已与服务端同步（' + diff + ' 张有变化）');
  } catch (e) { /* 同步失败不打扰用户，下一轮再试 */ }
}

function markRow() {
  document.querySelectorAll('.row').forEach(e => {
    e.classList.toggle('cur', Number(e.dataset.i) === S.cur);
  });
  const cur = document.querySelector('.row.cur');
  if (cur) cur.scrollIntoView({ block: 'nearest' });
}

async function load(i) {
  if (i < 0 || i >= S.files.length) return;
  if (S.cur >= 0 && S.cur !== i) await saveQuiet();
  S.cur = i;
  S.boxes = []; S.sel = -1;
  const name = S.files[i];
  document.getElementById('fname').textContent = name;
  document.getElementById('pos').textContent = (i + 1) + ' / ' + S.files.length;
  const r = await fetch('/api/boxes?name=' + encodeURIComponent(name));
  const d = await r.json();
  S.boxes = d.boxes || [];
  await loadImg(name);
  markRow();
  draw();
}

function loadImg(name) {
  return new Promise(res => {
    const im = new Image();
    im.onload = () => { S.img = im; resetView(); res(); };
    im.onerror = () => { S.img = null; res(); };
    im.src = '/img?name=' + encodeURIComponent(name) + '&t=' + Date.now();
  });
}

// 只算「适应窗口」这一层基准，不动用户的 zoom / pan
function computeFit() {
  const W = stage.clientWidth, H = stage.clientHeight;
  if (cv.width !== W || cv.height !== H || cv.style.width !== W + 'px') {
    cv.width = W; cv.height = H;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
  }
  if (!S.img) { S.fitScale = 1; S.baseOX = 0; S.baseOY = 0; return; }
  S.fitScale = Math.min(W / S.img.width, H / S.img.height);
  S.baseOX = (W - S.img.width * S.fitScale) / 2;
  S.baseOY = (H - S.img.height * S.fitScale) / 2;
}

function applyView() {
  S.scale = S.fitScale * S.zoom;
  S.ox = S.baseOX + S.panX;
  S.oy = S.baseOY + S.panY;
  const z = document.getElementById('zoomv');
  if (z) z.textContent = Math.round(S.zoom * 100) + '%';
}

function resetView() {
  S.zoom = 1; S.panX = 0; S.panY = 0;
}

// 以画布上的 (mx,my) 为锚点缩放：该点下的图像坐标保持不动
function zoomAt(mx, my, factor) {
  if (!S.img) return;
  const before = S.zoom;
  const z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, before * factor));
  if (Math.abs(z - before) < 1e-6) return;
  const ix = (mx - S.ox) / S.scale;
  const iy = (my - S.oy) / S.scale;
  S.zoom = z;
  S.scale = S.fitScale * z;
  S.panX = mx - S.baseOX - ix * S.scale;
  S.panY = my - S.baseOY - iy * S.scale;
  draw();
}

function draw() {
  computeFit();
  applyView();
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.img) {
    ctx.fillStyle = '#8b93a1'; ctx.font = '14px sans-serif';
    ctx.fillText('图片加载失败', 20, 30);
    return;
  }
  ctx.drawImage(S.img, S.ox, S.oy, S.img.width * S.scale, S.img.height * S.scale);
  S.boxes.forEach((b, k) => {
    const x = S.ox + b[0] * S.scale, y = S.oy + b[1] * S.scale;
    const w = (b[2] - b[0]) * S.scale, h = (b[3] - b[1]) * S.scale;
    const c = S.classes.find(z => z.id === b[4]);
    const col = c ? c.color : '#ff0';
    ctx.lineWidth = (k === S.sel ? 3 : 2);
    ctx.strokeStyle = col;
    ctx.strokeRect(x, y, w, h);
    if (k === S.sel) {
      ctx.setLineDash([6, 4]); ctx.strokeStyle = '#fff';
      ctx.strokeRect(x - 2, y - 2, w + 4, h + 4); ctx.setLineDash([]);
    }
    const label = c ? c.label : String(b[4]);
    ctx.font = 'bold 13px "Microsoft YaHei",sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = col;
    ctx.fillRect(x, Math.max(0, y - 19), tw + 10, 19);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 5, Math.max(13, y - 5));
  });
}

function toImg(e) {
  const r = cv.getBoundingClientRect();
  return [ (e.clientX - r.left - S.ox) / S.scale,
           (e.clientY - r.top - S.oy) / S.scale ];
}

function hit(p) {
  for (let k = S.boxes.length - 1; k >= 0; k--) {
    const b = S.boxes[k];
    if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]) return k;
  }
  return -1;
}

// 命中选中框的边/角：阈值按屏幕像素折算成图像像素，缩放多少手感都一样
function hitHandle(p) {
  if (S.sel < 0 || !S.boxes[S.sel]) return null;
  const b = S.boxes[S.sel];
  const t = HANDLE_PX / S.scale;
  if (p[0] < b[0] - t || p[0] > b[2] + t) return null;
  if (p[1] < b[1] - t || p[1] > b[3] + t) return null;
  const nearL = Math.abs(p[0] - b[0]) <= t, nearR = Math.abs(p[0] - b[2]) <= t;
  const nearT = Math.abs(p[1] - b[1]) <= t, nearB = Math.abs(p[1] - b[3]) <= t;
  if (nearL && nearT) return 'nw';
  if (nearR && nearT) return 'ne';
  if (nearL && nearB) return 'sw';
  if (nearR && nearB) return 'se';
  if (nearL) return 'w';
  if (nearR) return 'e';
  if (nearT) return 'n';
  if (nearB) return 's';
  return null;
}

function updateCursor(p) {
  let c = 'crosshair';
  const h = hitHandle(p);
  if (h) c = CURSORS[h];
  else if (hit(p) >= 0) c = 'move';
  cv.style.cursor = c;
}

cv.addEventListener('auxclick', e => { if (e.button === 1) e.preventDefault(); });

cv.addEventListener('mousedown', e => {
  if (!S.img) return;

  if (e.button === 1) {                  // 中键：拖动整张图
    e.preventDefault();
    S.panning = true;
    S.start = [e.clientX, e.clientY, S.panX, S.panY];
    cv.style.cursor = 'grabbing';
    return;
  }
  if (e.button !== 0) return;
  const p = toImg(e);

  const h = hitHandle(p);                // 1) 抓住选中框的边/角 -> 改大小
  if (h) {
    S.mode = 'resize'; S.handle = h;
    S.start = p; S.snap = S.boxes.map(b => b.slice());
    return;
  }

  const k = hit(p);                      // 2) 框内部 -> 选中，准备移动
  if (k >= 0 && !e.shiftKey) {
    S.sel = k; pickCls(S.boxes[k][4]);
    S.mode = 'move'; S.start = p; S.snap = S.boxes.map(b => b.slice());
    draw(); return;
  }

  S.mode = 'draw'; S.sel = -1;           // 3) 空白 -> 画新框
  dragFrom = [p[0], p[1]];
  draw();
});

cv.addEventListener('mousemove', e => {
  if (!S.img) return;

  if (S.panning) {
    S.panX = S.start[2] + (e.clientX - S.start[0]);
    S.panY = S.start[3] + (e.clientY - S.start[1]);
    draw(); return;
  }

  const p = toImg(e);

  if (S.mode === 'draw') {
    draw();
    const b = [Math.min(dragFrom[0], p[0]), Math.min(dragFrom[1], p[1]),
               Math.max(dragFrom[0], p[0]), Math.max(dragFrom[1], p[1])];
    ctx.save();
    ctx.strokeStyle = '#4a8cff'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.strokeRect(S.ox + b[0] * S.scale, S.oy + b[1] * S.scale,
                   (b[2] - b[0]) * S.scale, (b[3] - b[1]) * S.scale);
    ctx.restore();
    return;
  }

  if (S.mode === 'move') {               // 平移整个框，不做四舍五入 -> 亚像素
    const dx = p[0] - S.start[0], dy = p[1] - S.start[1];
    const s0 = S.snap[S.sel];
    S.boxes[S.sel] = [s0[0] + dx, s0[1] + dy, s0[2] + dx, s0[3] + dy, s0[4]];
    draw(); return;
  }

  if (S.mode === 'resize') {
    const s0 = S.snap[S.sel], b = s0.slice();
    const dx = p[0] - S.start[0], dy = p[1] - S.start[1];
    if (S.handle.indexOf('w') >= 0) b[0] = s0[0] + dx;
    if (S.handle.indexOf('e') >= 0) b[2] = s0[2] + dx;
    if (S.handle.indexOf('n') >= 0) b[1] = s0[1] + dy;
    if (S.handle.indexOf('s') >= 0) b[3] = s0[3] + dy;
    S.boxes[S.sel] = b;
    draw(); return;
  }

  updateCursor(p);
});

window.addEventListener('mouseup', e => {
  if (S.panning) {
    S.panning = false;
    cv.style.cursor = 'crosshair';
    draw();
  }
  if (!S.img || !S.mode) return;

  const round2 = v => Math.round(v * 100) / 100;   // 只做一次规整，不作为累计

  if (S.mode === 'draw') {
    let x1 = Math.min(dragFrom[0], toImg(e)[0]);
    let y1 = Math.min(dragFrom[1], toImg(e)[1]);
    let x2 = Math.max(dragFrom[0], toImg(e)[0]);
    let y2 = Math.max(dragFrom[1], toImg(e)[1]);
    x1 = Math.max(0, x1); y1 = Math.max(0, y1);
    x2 = Math.min(S.img.width, x2); y2 = Math.min(S.img.height, y2);
    if (x2 - x1 >= 2 && y2 - y1 >= 2) {   // 太小的当误点
      S.boxes.push([round2(x1), round2(y1), round2(x2), round2(y2), S.cls]);
      S.sel = S.boxes.length - 1;
    }
  } else if (S.mode === 'move' || S.mode === 'resize') {
    const b = S.boxes[S.sel];
    if (b[2] < b[0]) { const t = b[0]; b[0] = b[2]; b[2] = t; }  // 反向拖也认
    if (b[3] < b[1]) { const t = b[1]; b[1] = b[3]; b[3] = t; }
    S.boxes[S.sel] = [round2(b[0]), round2(b[1]), round2(b[2]), round2(b[3]), b[4]];
  }
  S.mode = null; S.handle = null;
  updateCursor(toImg(e));
  draw();
});

cv.addEventListener('wheel', e => {
  if (!S.img) return;
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top,
         e.deltaY < 0 ? 1.12 : 1 / 1.12);
}, { passive: false });

cv.addEventListener('dblclick', () => { resetView(); draw(); });

async function saveQuiet() {
  if (S.cur < 0) return false;
  const name = S.files[S.cur];
  const body = JSON.stringify({ name: name, boxes: S.boxes });
  let d = { ok: false, error: '网络异常' };
  // 页面卡顿/切后台时 fetch 可能直接被掐断，重试两次再说失败，
  // 免得明明写进磁盘了、界面上却当成没保存。
  for (let k = 0; k < 3; k++) {
    try {
      const r = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body
      });
      d = await r.json();
      break;
    } catch (err) {
      d = { ok: false, error: '网络异常' };
      await new Promise(res => setTimeout(res, 150 * (k + 1)));
    }
  }
  if (d.ok) {
    S.done[name] = S.boxes.length;
    updateStat();
    if (S.filter === 'all') { refreshRow(S.cur); markRow(); }
    else renderList();                  // 筛选视图下这一行可能该消失
  } else {
    toast('保存失败：' + (d.error || '未知') + '（按 S 重试）');
  }
  return !!d.ok;
}

async function manualSave() {
  await saveQuiet();
  toast('已保存 ' + S.boxes.length + ' 个框');
}

async function gotoNext() {
  if (S.cur < 0) return;
  await saveQuiet();
  if (S.cur < S.files.length - 1) load(S.cur + 1);
  else toast('已经是最后一张');
}

function gotoPrev() {
  if (S.cur > 0) load(S.cur - 1);
}

async function gotoNextTodo() {
  await saveQuiet();
  for (let k = S.cur + 1; k < S.files.length; k++) {
    if (!S.done[S.files[k]]) { load(k); return; }
  }
  for (let k = 0; k < S.files.length; k++) {
    if (!S.done[S.files[k]]) { load(k); return; }
  }
  toast('全部标完了');
}

document.getElementById('bnext').onclick = gotoNext;
document.getElementById('bprev').onclick = gotoPrev;
document.getElementById('breset').onclick = () => { resetView(); draw(); };
document.getElementById('bundo').onclick = () => {
  if (S.sel >= 0) { S.boxes.splice(S.sel, 1); S.sel = -1; draw(); }
};
document.getElementById('bundoall').onclick = () => {
  if (S.boxes.length && confirm('清空本图所有框？')) { S.boxes = []; S.sel = -1; draw(); }
};
document.querySelectorAll('#filter button').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('#filter button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    S.filter = b.dataset.f; renderList(); markRow();
  };
});

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key;

  if (k === 'Escape') { S.sel = -1; S.mode = null; draw(); return; }
  if (k === 'Home') { e.preventDefault(); resetView(); draw(); return; }

  // 方向键：有选中框 -> 微调它；没有 -> 左右换图
  if (k.indexOf('Arrow') === 0) {
    e.preventDefault();
    if (S.sel >= 0) {
      const st = e.shiftKey ? 0.1 : (e.ctrlKey ? 10 : 1);   // Shift = 亚像素
      const b = S.boxes[S.sel].slice();
      if (k === 'ArrowLeft')  { b[0] -= st; b[2] -= st; }
      if (k === 'ArrowRight') { b[0] += st; b[2] += st; }
      if (k === 'ArrowUp')    { b[1] -= st; b[3] -= st; }
      if (k === 'ArrowDown')  { b[1] += st; b[3] += st; }
      // 不在这里取整：连续微调时各自四舍五入会让宽高在 ±0.01 抖动，
      // 统一留到保存时按两位小数规整
      S.boxes[S.sel] = b;
      draw();
    } else if (k === 'ArrowLeft') gotoPrev();
    else if (k === 'ArrowRight') gotoNext();
    return;
  }

  if (k === 'Enter') { e.preventDefault(); gotoNext(); }
  else if (k === 'Delete' || k === 'Backspace') {
    if (S.sel >= 0) { e.preventDefault(); S.boxes.splice(S.sel, 1); S.sel = -1; draw(); }
  } else if (k === 's' || k === 'S') { manualSave(); }
  else if (k === 'u' || k === 'U') { gotoNextTodo(); }
  else if (k === 'a' || k === 'A') { gotoPrev(); }
  else if (k === 'd' || k === 'D') { gotoNext(); }
  else if (k === '+' || k === '=') { zoomAt(cv.width / 2, cv.height / 2, 1.25); }
  else if (k === '-' || k === '_') { zoomAt(cv.width / 2, cv.height / 2, 1 / 1.25); }
  else if (/^[0-9]$/.test(k)) {
    const id = Number(k);
    if (!S.classes.find(c => c.id === id)) return;
    if (S.sel >= 0) { S.boxes[S.sel][4] = id; pickCls(id); draw(); }
    else pickCls(id);
  }
});

window.addEventListener('beforeunload', () => {
  if (S.cur >= 0) {
    navigator.sendBeacon('/api/save',
      new Blob([JSON.stringify({ name: S.files[S.cur], boxes: S.boxes })],
               { type: 'application/json' }));
  }
});

boot();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store = None
    raw_dir = DEFAULT_RAW_DIR
    labels_dir = DEFAULT_LABELS_DIR
    preview_dir = DEFAULT_PREVIEW_DIR
    files = []
    classes = []
    server_version = "M23Label/1.0"

    def log_message(self, fmt, *args):
        # 每张缩略图都打一行会把终端刷爆，只留错误
        if args and str(args[0]).startswith(("4", "5")):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---- 响应工具 ----
    def _send(self, code, body, ctype="application/json; charset=utf-8",
              extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _file(self, path, ctype, cache=False):
        if not os.path.isfile(path):
            self._json({"error": "not found"}, 404)
            return
        with open(path, "rb") as f:
            data = f.read()
        extra = {}
        if cache:
            st = os.stat(path)
            extra["Cache-Control"] = "max-age=86400"
            extra["ETag"] = '"%d-%d"' % (int(st.st_mtime), st.st_size)
        self._send(200, data, ctype, extra)

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path

        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return

        if path == "/api/state":
            done = {}
            for n in self.files:
                b = self.store.boxes_of(n)
                if n in self.store.items:
                    done[n] = len(b)
            self._json({"classes": class_display(self.classes),
                        "files": self.files, "done": done})
            return

        if path == "/api/boxes":
            name = (q.get("name") or [""])[0]
            if name not in self.files:
                self._json({"error": "bad name"}, 400)
                return
            self._json({"boxes": self.store.boxes_of(name)})
            return

        if path == "/img":
            name = (q.get("name") or [""])[0]
            if name not in self.files:
                self._json({"error": "bad name"}, 400)
                return
            src = os.path.join(self.raw_dir, name)
            ext = os.path.splitext(name)[1].lower()
            ctype = {".png": "image/png", ".bmp": "image/bmp",
                     ".webp": "image/webp"}.get(ext, "image/jpeg")
            self._file(src, ctype, cache=True)
            return

        if path == "/thumb":
            name = (q.get("name") or [""])[0]
            if name not in self.files:
                self._json({"error": "bad name"}, 400)
                return
            key = name
            if key in _thumb_cache:
                self._send(200, _thumb_cache[key], "image/jpeg")
                return
            try:
                im = open_rgb(os.path.join(self.raw_dir, name))
                im.thumbnail((THUMB_MAX, THUMB_MAX), Image.BOX)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=72)
                data = buf.getvalue()
                _thumb_cache[key] = data      # 内存缓存，翻页不再重算
                self._send(200, data, "image/jpeg")
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return

        self._json({"error": "not found"}, 404)

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/save":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            doc = json.loads(self.rfile.read(n).decode("utf-8"))
            name = doc["name"]
            if name not in self.files:
                raise ValueError("图片名不在当前列表：%s" % name)
            boxes = []
            for b in doc.get("boxes", []):
                if len(b) != 5:
                    raise ValueError("框格式不对：%r" % (b,))
                x1, y1, x2, y2, cid = (float(b[0]), float(b[1]), float(b[2]),
                                       float(b[3]), int(b[4]))
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                if cid not in [c for c, _ in self.classes]:
                    raise ValueError("类别 %d 不在 classes.txt 里" % cid)
                # 统一保留两位小数：亚像素精度足够用，json/txt 也不会出现
                # 100.37000000000001 这种浮点尾巴
                boxes.append([round(x1, 2), round(y1, 2),
                              round(x2, 2), round(y2, 2), cid])

            src = os.path.join(self.raw_dir, name)
            with Image.open(src) as im:
                w, h = ImageOps.exif_transpose(im).size
            with _lock:
                yolo, prev = self.store.put(name, src, (h, w), boxes,
                                            self.labels_dir, self.preview_dir)
                self.store.save_all()
            self._json({"ok": True, "count": len(boxes), "yolo": yolo,
                        "preview": prev})
        except Exception as exc:
            self._json({"ok": False, "error": "%s: %s"
                        % (type(exc).__name__, exc)}, 400)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="M2-3 浏览器标注器：起一个本地网页，用浏览器画框标图")
    ap.add_argument("--port", type=int, default=8765, help="端口，默认 8765")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址，默认只本机；手机访问用 0.0.0.0")
    ap.add_argument("--folder", default=DEFAULT_RAW_DIR, help="图片目录")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="annotations.json")
    ap.add_argument("--classes", default=DEFAULT_CLASSES_FILE,
                    help="classes.txt")
    ap.add_argument("--labels", default=DEFAULT_LABELS_DIR, help="YOLO txt 目录")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.folder):
        print("图片目录不存在：%s" % args.folder, file=sys.stderr)
        return 2
    files = scan_images(args.folder)
    if not files:
        print("目录里没有图片：%s" % args.folder, file=sys.stderr)
        return 2

    classes = load_classes(args.classes)
    Handler.store = Store(args.output, classes)
    Handler.raw_dir = args.folder
    Handler.labels_dir = args.labels
    Handler.preview_dir = os.path.join(args.labels, "preview")
    Handler.files = files
    Handler.classes = classes

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print("=" * 58)
    print(" M2-3 标注器已启动")
    print("   图片    : %s（%d 张）" % (args.folder, len(files)))
    print("   类别    : %s" % "  ".join("%d %s" % (c, n) for c, n in classes))
    print("   标注    : %s" % args.output)
    print("   YOLO    : %s" % args.labels)
    print()
    print("   在浏览器打开： http://%s:%d" % (host, args.port))
    print("   停止服务：Ctrl+C")
    print("=" * 58)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
