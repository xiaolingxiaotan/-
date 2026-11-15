# -*- coding: utf-8 -*-
"""M2-3 无线收件服务（电脑端，纯标准库、零下载、零第三方软件）。

一个进程同时提供两个入口，照片都落到同一个「收件箱」inbox/：

  1) 网页批量上传（推荐，手机不用装任何 App）：
     拍摄时完全不用打开网页 —— 用系统相机一直拍，照片留在手机相册。
     拍完回到网页点一次「选择照片」多选全部，照片先压进浏览器
     IndexedDB 缓存，再自动逐张 POST 到电脑；失败的留在缓存里可补传。
     页面另有「逐张连拍」备选：每拍一张都要在相机里点确认，比较麻烦。
     注：<input capture> 每张都必须确认返回网页，这是浏览器规定；
     网页内自己开摄像头连拍需要 HTTPS 安全上下文，本机 http 环境不适用。
  2) WebDAV（手机装 FolderSync / PhotoSync 后可后台自动同步）：
     远程地址填 http://<电脑IP>:8000/ ，本地目录 DCIM/Camera，
     同步方向「仅上传到远程」，定时或即时同步。服务实现
     OPTIONS / PROPFIND / MKCOL / PUT / HEAD / GET / DELETE / MOVE。

落进 inbox/ 的照片由 wireless_import.py 自动改名入库到 data/raw，
本服务不碰项目目录，也不做像素处理之外的任何事。

网络说明：手机和电脑必须处于同一局域网（连同一个 WiFi；若校园网有
AP 隔离，则手机开个人热点、电脑连手机热点）。服务绑定 0.0.0.0，
启动时自动列出本机所有可用 IP，**以窗口打印的地址为准**，IP 会变，
旧地址作废；浏览器必须手输 http://（别被自动补成 https://）。
注意：部分无线网卡不支持 Windows 承载网络，电脑开不了「移动热点」，
此时不要假设 192.168.137.1 存在。
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import posixpath
import re
import shutil
import socket
import sys
import threading
import time
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
DEFAULT_INBOX = HERE / "inbox"
MAX_BODY = 32 * 1024 * 1024
DATA_URL_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=\s]+)$")
MAGIC = {
    "jpeg": lambda b: b.startswith(b"\xff\xd8"),
    "png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    "webp": lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP",
}
SUFFIX = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}
DAV_ALLOW = "OPTIONS, GET, HEAD, PUT, DELETE, MKCOL, PROPFIND, MOVE, COPY"
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>拍照收件箱</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing:border-box; }
  body { margin:0; padding:16px; font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
         background:#f5f6f8; color:#1b1f24; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { font-size:12px; color:#5b6572; margin-bottom:14px; line-height:1.7; }
  .card { background:#fff; border:1px solid #e3e6ea; border-radius:12px; padding:14px; margin-bottom:12px; }
  .big { font-size:22px; font-weight:700; }
  button { width:100%; min-height:64px; font-size:19px; font-weight:600; color:#fff;
           background:#2f6fed; border:0; border-radius:12px; margin-top:8px; }
  button.ghost { background:#e9eef8; color:#24509c; min-height:48px; font-size:16px; }
  button.stop { background:#d9534f; min-height:48px; font-size:16px; }
  button:active { opacity:.8; }
  button:disabled { background:#a9b6cc; }
  #file,#batch { display:none; }
  #msg { margin-top:10px; font-size:14px; min-height:22px; }
  #msg.ok { color:#177a45; } #msg.err { color:#c02c2c; }
  #thumb { display:block; max-width:100%; max-height:180px; margin:10px auto 0; border-radius:8px; }
  code { background:#eef1f5; padding:1px 5px; border-radius:4px; word-break:break-all; }
  .tip { font-size:12px; color:#5b6572; margin-top:10px; line-height:1.6; }
  .tag { display:inline-block; font-size:12px; font-weight:600; background:#2f6fed; color:#fff;
         padding:2px 8px; border-radius:6px; vertical-align:middle; }
  .tag.alt { background:#8a94a6; }
  .chk { display:block; margin:10px 0; font-size:15px; }
  .chk input { width:19px; height:19px; vertical-align:middle; }
</style>
</head>
<body>
<h1>拍照收件箱</h1>
<div class="sub">
  照片直传电脑，不经过微信；不分类，自动连号入库。<br>
  __URLS__
</div>

<div class="big" id="count" style="margin-bottom:4px">已上传 0 张</div>
<div id="msg" style="margin-bottom:12px"></div>

<div class="card">
  <div class="big"><span class="tag">推荐</span>批量拍摄</div>
  <div class="tip" style="margin-top:6px">
    <b>拍照时不用打开这个网页</b>：直接用系统相机一直拍（长按快门可连拍），
    拍完回到这里点一次「选择照片」，把刚拍的全部选中，就自动传到电脑。
  </div>
  <div class="big" id="cacheInfo" style="margin-top:10px">已缓存 0 张</div>
  <button id="choose">选择照片（可多选 / 可分几次选）</button>
  <label class="chk"><input type="checkbox" id="autoUp" checked> 选完自动上传（推荐，不用再点别的）</label>
  <button id="go" class="ghost" disabled>一起上传</button>
  <button id="clear" class="ghost" disabled>清空缓存</button>
  <input type="file" id="batch" accept="image/*" multiple>
  <div class="tip">
    照片先存在<b>这台手机浏览器的缓存</b>里（IndexedDB），关掉页面或锁屏也不会丢；
    上传成功的会自动从缓存删掉，失败的留在缓存里，再点一次「一起上传」只补传失败的。<br>
    若相册选择器只允许单选，分几次选即可，缓存会累加。
  </div>
</div>

<div class="card">
  <div class="big"><span class="tag alt">备选</span>逐张连拍</div>
  <img id="thumb" alt="">
  <button id="start">开始连拍</button>
  <button id="one" class="ghost">只拍一张</button>
  <button id="stop" class="stop" disabled>停止连拍</button>
  <input type="file" id="file" accept="image/*" capture="environment">
  <div class="tip">
    每拍一张都要在相机里点确认，回到网页后才再次唤起相机 —— 比批量模式麻烦，
    只在没法多选照片时用。<b>取消拍照</b>会自动停下。
  </div>
  <button id="resume" class="ghost" style="display:none">继续拍</button>
</div>

<div class="card">
  <div class="tip" style="margin:0">
    想让手机<b>后台自动上传</b>（连选择都不用点）：装 FolderSync，新增 WebDAV 账号，
    服务器填 <code id="dav">__DAV__</code>（用户名密码留空），
    本地目录 <code>DCIM/Camera</code> → 远程 <code>/</code>，方向「仅上传」。
  </div>
</div>

<script>
const MAX_SIDE = __MAX_SIDE__, QUALITY = __QUALITY__;
const $ = (id) => document.getElementById(id);
let auto = false, busy = false, count = 0, uploading = false;
let db = null, mem = [];

function msg(t, k) { $('msg').textContent = t; $('msg').className = k || ''; }
function bump(n) { count += n; $('count').textContent = '已上传 ' + count + ' 张'; }
function setState() {
  $('start').disabled = auto; $('one').disabled = auto || busy;
  $('stop').disabled = !auto; $('resume').style.display = 'none';
}
function pick() { $('file').click(); }

function compress(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => {
      const s = Math.min(1, MAX_SIDE / Math.max(im.naturalWidth, im.naturalHeight));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(im.naturalWidth * s));
      c.height = Math.max(1, Math.round(im.naturalHeight * s));
      c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      resolve({ url: c.toDataURL('image/jpeg', QUALITY), w: c.width, h: c.height });
    };
    im.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片解码失败')); };
    im.src = url;
  });
}

// ---- 手机端缓存：优先 IndexedDB，不可用时退到内存 ----
function openCache() {
  return new Promise((resolve) => {
    if (!window.indexedDB) { resolve(); return; }
    let r;
    try { r = indexedDB.open('m23_photo_cache', 1); } catch (e) { resolve(); return; }
    r.onupgradeneeded = () => {
      const d = r.result;
      if (!d.objectStoreNames.contains('photos')) {
        d.createObjectStore('photos', { keyPath:'id', autoIncrement:true });
      }
    };
    r.onsuccess = () => { db = r.result; resolve(); };
    r.onerror = () => resolve();
    r.onblocked = () => resolve();
  });
}
function cacheAdd(rec) {
  if (!db) { mem.push(Object.assign({ id: 'm' + Date.now() + Math.random() }, rec)); return Promise.resolve(); }
  return new Promise((res, rej) => {
    const t = db.transaction('photos', 'readwrite');
    t.objectStore('photos').add(rec);
    t.oncomplete = () => res(); t.onerror = () => rej(t.error);
  });
}
function cacheAll() {
  if (!db) return Promise.resolve(mem.slice());
  return new Promise((res, rej) => {
    const q = db.transaction('photos', 'readonly').objectStore('photos').getAll();
    q.onsuccess = () => res(q.result || []); q.onerror = () => rej(q.error);
  });
}
function cacheDel(id) {
  if (!db) { mem = mem.filter(x => x.id !== id); return Promise.resolve(); }
  return new Promise((res, rej) => {
    const t = db.transaction('photos', 'readwrite');
    t.objectStore('photos').delete(id);
    t.oncomplete = () => res(); t.onerror = () => rej(t.error);
  });
}
function cacheClear() {
  if (!db) { mem = []; return Promise.resolve(); }
  return new Promise((res, rej) => {
    const t = db.transaction('photos', 'readwrite');
    t.objectStore('photos').clear();
    t.oncomplete = () => res(); t.onerror = () => rej(t.error);
  });
}
async function refreshCache() {
  const a = await cacheAll();
  $('cacheInfo').textContent = '已缓存 ' + a.length + ' 张';
  $('go').disabled = uploading || a.length === 0;
  $('clear').disabled = a.length === 0;
  return a.length;
}

// ---- 批量：一次选多张，先缓存，再统一上传 ----
async function addFiles(files) {
  let n = 0;
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    if (f.type && f.type.indexOf('image/') !== 0) continue;
    $('cacheInfo').textContent = '正在缓存 ' + (i + 1) + ' / ' + files.length + ' 张…';
    try {
      const c = await compress(f);
      await cacheAdd({ url: c.url, w: c.w, h: c.h, name: f.name || '', added: Date.now() });
      n++;
    } catch (e) { console.warn('跳过一张：', f.name, e); }
  }
  await refreshCache();
  return n;
}
async function uploadAll() {
  if (uploading) return;
  const items = await cacheAll();
  if (!items.length) { msg('缓存里没有照片'); return; }
  uploading = true; $('go').disabled = true;
  let done = 0, fail = 0;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    try {
      const r = await fetch('/upload', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ img: it.url, w: it.w, h: it.h }) });
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
      await cacheDel(it.id);
      done++; bump(1);
    } catch (e) { fail++; }
    $('cacheInfo').textContent = '上传中 ' + (done + fail) + ' / ' + items.length
      + '（成功 ' + done + '，失败 ' + fail + '）';
  }
  uploading = false;
  await refreshCache();
  if (fail) msg('传完 ' + done + ' 张，' + fail + ' 张没成功，仍留在缓存里，再点一次「一起上传」可补传', 'err');
  else msg('已全部上传 ' + done + ' 张到电脑', 'ok');
}
$('choose').onclick = () => $('batch').click();
$('batch').onchange = async (e) => {
  const fs = Array.from(e.target.files || []);
  e.target.value = '';
  if (!fs.length) return;
  msg('正在把照片存进手机缓存…');
  const n = await addFiles(fs);
  if (!n) { msg('没有读到可用的图片', 'err'); return; }
  if ($('autoUp').checked) await uploadAll();
  else msg('已缓存 ' + n + ' 张；点「一起上传」再传到电脑', 'ok');
};
$('go').onclick = uploadAll;
$('clear').onclick = async () => {
  if (!confirm('清空手机缓存里还没上传的照片？')) return;
  await cacheClear(); await refreshCache(); msg('缓存已清空');
};

// ---- 逐张连拍（备选：每拍一张都要确认一次）----
$('start').onclick = () => { auto = true; setState(); msg('连拍中…'); pick(); };
$('one').onclick = () => { auto = false; setState(); pick(); };
$('stop').onclick = () => { auto = false; setState(); msg('已停止连拍', 'ok'); };
$('resume').onclick = () => { if (auto) { $('resume').style.display='none'; pick(); } };

$('file').onchange = async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) { auto = false; setState(); msg('已取消拍照'); return; }
  busy = true; setState(); msg('正在上传…');
  try {
    const c = await compress(f);
    const r = await fetch('/upload', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ img:c.url, w:c.w, h:c.h }) });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    bump(1);
    $('thumb').src = c.url;
    msg('已保存 ' + j.file, 'ok');
  } catch (err) { msg('上传失败：' + err.message, 'err'); }
  finally {
    e.target.value = ''; busy = false; setState();
    if (auto) setTimeout(() => {
      // 部分浏览器要求用户手势才肯唤起相机；被拦时给一个「继续拍」按钮
      $('resume').style.display = 'block';
      pick();
    }, 350);
  }
};
setState();
openCache().then(refreshCache);
</script>
</body>
</html>
"""


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("223.5.5.5", 80))
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def render_page(cfg: dict) -> str:
    urls = cfg["urls"] or ["http://127.0.0.1:" + str(cfg["port"])]
    main = urls[0]
    hotspot = [u for u in urls if "192.168.137." in u or "192.168.43." in u]
    if hotspot:
        main = hotspot[0]
    links = "服务地址 " + " / ".join("<code>" + u + "</code>" for u in urls)
    return (PAGE.replace("__URLS__", links)
                .replace("__DAV__", main + "/")
                .replace("__MAX_SIDE__", str(cfg["max_side"]))
                .replace("__QUALITY__", str(cfg["quality"])))


# ---------------------------------------------------------------------------
# WebDAV / HTTP
# ---------------------------------------------------------------------------

class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, cfg):
        super().__init__(addr, handler)
        self.cfg = cfg


class Handler(BaseHTTPRequestHandler):
    server_version = "M23Inbox/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---------- 基础 ----------
    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        extra = extra or {}
        self.send_response(code)
        if ctype:
            self.send_header("Content-Type", ctype)
        # HEAD 要声明实体长度但不发 body，避免重复 Content-Length 头
        self.send_header("Content-Length", extra.get("Content-Length", str(len(body))))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            if k.lower() != "content-length":
                self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _read_body(self, n: int) -> bytes:
        """FolderSync 等 WebDAV 客户端的 PUT 常用 Transfer-Encoding: chunked。"""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            buf = bytearray()
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    while True:  # 吃掉 trailer
                        t = self.rfile.readline()
                        if t in (b"\r\n", b"\n", b""):
                            break
                    break
                if len(buf) + size > MAX_BODY:
                    raise ValueError("too large")
                buf += self.rfile.read(size)
                self.rfile.read(2)
            return bytes(buf)
        return self.rfile.read(n) if n > 0 else b""

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _dav_headers(self):
        return {"DAV": "1,2", "Allow": DAV_ALLOW, "MS-Author-Via": "DAV"}

    def _local(self, raw_path: str) -> Path | None:
        """URL 路径 -> inbox 下的真实路径，挡住路径穿越。"""
        try:
            p = unquote(urlsplit(raw_path).path)
            p = posixpath.normpath(p).lstrip("/")
            if p.startswith("..") or "\x00" in p:
                return None
            root: Path = self.server.cfg["inbox"]
            target = (root / p).resolve() if p else root.resolve()
            if target != root.resolve() and root.resolve() not in target.parents:
                return None
            return target
        except Exception:
            return None

    def _href(self, p: Path) -> str:
        root: Path = self.server.cfg["inbox"]
        try:
            rel = p.relative_to(root).as_posix()
            rel = "/" if rel in ("", ".") else "/" + rel
        except ValueError:
            rel = "/"
        if p.is_dir() and not rel.endswith("/"):
            rel += "/"
        return quote(rel, safe="/")

    def _propfind(self, path: Path, depth: str) -> bytes:
        items = [path]
        if depth != "0" and path.is_dir():
            try:
                items += sorted(path.iterdir(), key=lambda x: x.name)
            except OSError:
                pass
        parts = ['<?xml version="1.0" encoding="utf-8"?>',
                 '<D:multistatus xmlns:D="DAV:">']
        for p in items:
            try:
                st = p.stat()
            except OSError:
                continue
            parts.append("<D:response><D:href>" + escape(self._href(p)) + "</D:href><D:propstat><D:prop>")
            if p.is_dir():
                parts.append("<D:resourcetype><D:collection/></D:resourcetype>")
            else:
                parts.append("<D:resourcetype/><D:getcontentlength>" + str(st.st_size) +
                             "</D:getcontentlength><D:getcontenttype>" +
                             escape(mimetypes.guess_type(p.name)[0] or "application/octet-stream") +
                             "</D:getcontenttype>")
            parts.append("<D:displayname>" + escape(p.name or "/") + "</D:displayname>")
            parts.append("<D:getlastmodified>" + formatdate(st.st_mtime, usegmt=True) +
                         "</D:getlastmodified>")
            parts.append("</D:prop><D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>")
        parts.append("</D:multistatus>")
        return "".join(parts).encode("utf-8")

    # ---------- GET / HEAD ----------
    def do_GET(self):
        cfg = self.server.cfg
        p = urlsplit(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, render_page(cfg).encode("utf-8"), "text/html; charset=utf-8")
            return
        if p == "/status":
            n = sum(1 for x in cfg["inbox"].rglob("*")
                    if x.is_file() and not x.name.startswith(".") and "_done" not in x.parts)
            return self._json(200, {"ok": True, "inbox": str(cfg["inbox"]), "files": n,
                                   "urls": cfg["urls"]})
        if p == "/favicon.ico":
            return self._send(204)
        local = self._local(self.path)
        if local is None or not local.is_file():
            return self._send(404, b"not found")
        try:
            data = local.read_bytes()
        except OSError:
            return self._send(404, b"not found")
        self._send(200, data, mimetypes.guess_type(local.name)[0] or "application/octet-stream")

    def do_HEAD(self):
        local = self._local(self.path)
        if local is None or not local.is_file():
            return self._send(404)
        self._send(200, b"", mimetypes.guess_type(local.name)[0] or "application/octet-stream",
                   {"Content-Length": str(local.stat().st_size)})

    # ---------- 网页上传 ----------
    def do_POST(self):
        if urlsplit(self.path).path != "/upload":
            return self._json(404, {"ok": False, "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad length"})
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"ok": False, "error": "请求体过大或为空"})
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return self._json(400, {"ok": False, "error": "JSON 解析失败"})
        m = DATA_URL_RE.match(str(data.get("img", "")))
        if not m:
            return self._json(400, {"ok": False, "error": "图片不是合法 dataURL"})
        kind, b64 = m.group(1), m.group(2)
        try:
            blob = base64.b64decode(re.sub(r"\s+", "", b64), validate=True)
        except Exception:
            return self._json(400, {"ok": False, "error": "base64 解码失败"})
        if not blob or len(blob) > MAX_BODY or not MAGIC[kind](blob):
            return self._json(400, {"ok": False, "error": "图片内容不合法"})
        inbox: Path = self.server.cfg["inbox"]
        with _LOCK:
            inbox.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            i = 1
            while True:
                name = f"WEB_{stamp}_{i:03d}{SUFFIX[kind]}"
                f = inbox / name
                if not f.exists():
                    break
                i += 1
            f.write_bytes(blob)
        print(f"[网页上传] {name}  {len(blob)//1024} KB", flush=True)
        self._json(200, {"ok": True, "file": name})

    # ---------- WebDAV ----------
    def do_OPTIONS(self):
        self._send(200, b"", "text/plain; charset=utf-8", self._dav_headers())

    def do_PROPFIND(self):
        local = self._local(self.path)
        if local is None or not local.exists():
            return self._send(404, b"not found", extra=self._dav_headers())
        body = self._propfind(local, self.headers.get("Depth", "1"))
        self._send(207, body, "application/xml; charset=utf-8", self._dav_headers())

    def do_MKCOL(self):
        local = self._local(self.path)
        if local is None:
            return self._send(403)
        if local.exists():
            return self._send(405)
        try:
            local.mkdir(parents=True)
            return self._send(201)
        except OSError:
            return self._send(409)

    def do_PUT(self):
        local = self._local(self.path)
        if local is None:
            return self._send(403)
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send(400)
        if n < 0 or n > MAX_BODY:
            return self._send(413)
        try:
            buf = self._read_body(n)
        except (OSError, ValueError):
            return self._send(413)
        existed = local.exists()
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(buf)
        except OSError as e:
            return self._send(500, str(e).encode("utf-8"))
        print(f"[WebDAV] {'覆盖' if existed else '新增'} {local.name}  {len(buf)//1024} KB", flush=True)
        self._send(201 if not existed else 204, extra=self._dav_headers())

    def do_DELETE(self):
        local = self._local(self.path)
        if local is None:
            return self._send(403)
        if not local.exists():
            return self._send(404)
        try:
            if local.is_dir():
                shutil.rmtree(local)
            else:
                local.unlink()
        except OSError:
            return self._send(423)
        self._send(204)

    def do_MOVE(self):
        src = self._local(self.path)
        dst = self._local(self.headers.get("Destination", ""))
        if src is None or dst is None or not src.exists():
            return self._send(403)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError:
            return self._send(500)
        self._send(201, extra=self._dav_headers())

    def do_COPY(self):
        self._send(501)


# ---------------------------------------------------------------------------
# 自检与启动
# ---------------------------------------------------------------------------

def self_test() -> None:
    import http.client
    import tempfile

    with tempfile.TemporaryDirectory(prefix="m23_dav_") as td:
        root = Path(td)
        cfg = {"inbox": root / "inbox", "port": 0, "urls": ["http://127.0.0.1:8000"],
               "max_side": 1706, "quality": 0.9}
        cfg["inbox"].mkdir(parents=True, exist_ok=True)
        srv = Server(("127.0.0.1", 0), Handler, cfg)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        checks = 0

        def check(cond, label):
            nonlocal checks
            assert cond, "FAIL: " + label
            checks += 1
            print("  [OK] " + label)

        def req(method, path, body=None, headers=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=6)
            c.request(method, path, body=body, headers=headers or {})
            r = c.getresponse()
            data = r.read()
            code, hdrs = r.status, dict(r.getheaders())
            c.close()
            return code, hdrs, data

        try:
            code, _, html = req("GET", "/")
            check(code == 200 and "开始连拍".encode("utf-8") in html and b"FolderSync" in html,
                  "首页含连拍按钮与 WebDAV 配置提示")
            check(b"m23_photo_cache" in html and b"multiple" in html and b"uploadAll" in html,
                  "首页含批量选择 + 浏览器缓存 + 统一上传逻辑")
            check(b"choose" in html and b"autoUp" in html and b"cacheDel" in html,
                  "首页含自动上传开关与失败保留缓存逻辑")
            code, h, _ = req("OPTIONS", "/")
            check(code == 200 and "1,2" in h.get("DAV", ""), "OPTIONS 声明 DAV 1,2")
            code, _, xml = req("PROPFIND", "/", headers={"Depth": "1"})
            check(code == 207 and b"multistatus" in xml and b"collection" in xml,
                  "PROPFIND 返回 207 多状态 XML")
            code, _, _ = req("MKCOL", "/Camera")
            check(code == 201 and (cfg["inbox"] / "Camera").is_dir(), "MKCOL 建目录")
            png = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR" + b"\x00" * 17)
            code, _, _ = req("PUT", "/Camera/IMG_001.png", png)
            check(code == 201 and (cfg["inbox"] / "Camera" / "IMG_001.png").read_bytes() == png,
                  "PUT 上传并保留手机原文件名与子目录")
            code, h, _ = req("HEAD", "/Camera/IMG_001.png")
            check(code == 200, "HEAD 已存在文件")
            code, _, xml = req("PROPFIND", "/Camera", headers={"Depth": "1"})
            check(code == 207 and b"IMG_001.png" in xml and b"getcontentlength" in xml,
                  "列目录含文件名与大小")
            code, _, data = req("GET", "/Camera/IMG_001.png")
            check(code == 200 and data == png, "GET 下载字节一致")
            code, _, _ = req("PUT", "/../escape.png", b"x")
            check(not (root / "escape.png").exists(), "路径穿越不会写到 inbox 之外")
            # 模拟 FolderSync 的 chunked PUT（无 Content-Length）
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=6)
            c.putrequest("PUT", "/Camera/chunked.png")
            c.putheader("Transfer-Encoding", "chunked")
            c.endheaders()
            c.send(b"5\r\nhello\r\n0\r\n\r\n")
            r = c.getresponse()
            r.read()
            chunk_code = r.status
            c.close()
            check(chunk_code in (201, 204) and
                  (cfg["inbox"] / "Camera" / "chunked.png").read_bytes() == b"hello",
                  "chunked PUT（WebDAV 客户端常见）")
            dataurl = "data:image/png;base64," + base64.b64encode(png).decode()
            code, _, body = req("POST", "/upload",
                                json.dumps({"img": dataurl, "w": 1, "h": 1}).encode(),
                                {"Content-Type": "application/json"})
            check(code == 200 and json.loads(body).get("ok"), "网页 POST 上传成功")
            check(len(list(cfg["inbox"].glob("WEB_*.png"))) == 1, "网页上传落到 inbox 且前缀 WEB_")
            code, _, _ = req("MOVE", "/Camera/IMG_001.png",
                             headers={"Destination": "http://127.0.0.1:%d/Camera/moved.png" % port})
            check(code in (201, 204) and (cfg["inbox"] / "Camera" / "moved.png").exists(),
                  "MOVE 重命名")
            code, _, _ = req("DELETE", "/Camera/moved.png")
            check(code == 204 and not (cfg["inbox"] / "Camera" / "moved.png").exists(),
                  "DELETE 删除")
            print(f"SELF-TEST ALL PASS ({checks} 项)")
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(timeout=3)


def main() -> None:
    ap = argparse.ArgumentParser(description="M2-3 无线收件服务（WebDAV + 网页连拍）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--inbox", default=str(DEFAULT_INBOX), help="收件箱，默认脚本同目录 inbox/")
    ap.add_argument("--max-side", type=int, default=1706)
    ap.add_argument("--quality", type=float, default=0.9)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    inbox = Path(args.inbox)
    try:
        inbox.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[错误] 收件箱不可用：{e}")
        sys.exit(1)

    ips = lan_ips()
    cfg = {"inbox": inbox, "port": args.port,
           "urls": [f"http://{ip}:{args.port}" for ip in ips],
           "max_side": args.max_side, "quality": args.quality}
    # 只列真实检测到的网卡地址，绝不注入不存在的「热点固定 IP」
    if not cfg["urls"]:
        cfg["urls"] = [f"http://127.0.0.1:{args.port}"]

    try:
        srv = Server(("0.0.0.0", args.port), Handler, cfg)
    except OSError as e:
        print(f"[错误] 端口 {args.port} 启动失败：{e}，可换 --port 8080")
        sys.exit(1)

    print("=" * 68)
    print("M2-3 无线收件服务 · WebDAV + 网页批量上传（零下载、零安装）")
    print("=" * 68)
    print(f"收件箱    {inbox}")
    print(f"最终入库  交给 wireless_import.py（另开一个黑窗口）")
    print("-" * 68)
    print("【手机要打开的地址】手机与电脑须连同一个 WiFi；校园网有 AP 隔离时，")
    print("  改成手机开个人热点、电脑连手机热点。浏览器手输下面任一个，")
    print("  必须是 http:// 开头，别被自动补成 https://：")
    for u in cfg["urls"]:
        print("    " + u)
    print("  IP 会变，每次重启服务都以这里打印的为准；打不开先看本窗口有无新地址。")
    print("-" * 68)
    print("手机有两种用法，二选一，照片都进同一个收件箱：")
    print("  A 不装 App（推荐）：拍照时不用开网页，用系统相机一直拍；")
    print("     拍完打开上面地址 -> 「选择照片」多选全部 -> 自动批量上传")
    print("     照片先存手机浏览器缓存，没传成功的留着可补传；另有逐张连拍备选")
    print("  B 装 FolderSync：WebDAV 账号，服务器填 " + cfg["urls"][0] + "/")
    print("     用户名密码留空；本地 DCIM/Camera -> 远程 /，方向「仅上传」")
    print("-" * 68)
    print("首次启动若弹「Windows 安全中心警报」→ 勾「专用网络」点「允许访问」；")
    print("若仍连不上，右键管理员运行 allow_firewall.bat 放行 8000。")
    print("Ctrl+C 停止。")
    print("=" * 68, flush=True)

    def watch():
        seen = set(ips)
        while True:
            time.sleep(5)
            new = set(lan_ips()) - seen
            if new:
                seen |= new
                for ip in sorted(new):
                    print(f"[网络变化] http://{ip}:{args.port}", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
