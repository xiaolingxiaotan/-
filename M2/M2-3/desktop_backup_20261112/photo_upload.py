# -*- coding: utf-8 -*-
"""M2-3 拍照直传（只跑在电脑上，手机端零安装、零文件）。

用法（在电脑上）：
    双击 start_upload.bat，或 python photo_upload.py
手机用浏览器打开启动时打印的 http://<电脑IP>:8000
点「拍照上传」-> 系统相机拍完确认 -> 照片直接落到电脑。

不分类：所有照片平铺在一个目录里，按 IMG_0001.jpg 连号命名，不覆盖。

产物（全在电脑端）：
    照片  <RAW_DIR>/IMG_0001.jpg      默认 RAW_DIR 是 WSL 项目的 M2-3/data/raw
    台账  <脚本同目录>/photo_log.csv   时间 / 文件名 / 尺寸 / 字节 / 来源

实现约定：
    * 只用 Python 标准库，兼容 Python 3.14（3.14 已移除 cgi，所以不用 multipart）。
    * 前端用 canvas 缩放后转 dataURL，POST JSON；服务端 base64 解码落盘。
      canvas 重编码会丢掉 EXIF，因此「标注坐标基准 = 存下来的像素」，训练侧读图不会再有转正歧义。
    * 最长边默认压到 1706，与现有 166 张素材的 1279x1706 对齐。
    * 绑定 0.0.0.0，启动时列出所有网卡 IP；运行中检测到新网卡（例如刚开的热点）会自动打印新地址。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent

# 默认照片落盘位置：WSL 项目里的 M2-3/data/raw（Windows 可直接写 UNC）。
# 想让照片也落在桌面这个文件夹，改下面这一行，或启动时加 --raw-dir。
DEFAULT_RAW = (
    r"\\wsl$\Ubuntu2204\home\xiaolingxiaotan\智能车一轮考核"
    r"\智能车一轮考核\M2\M2-3\data\raw"
)
DEFAULT_LOG = HERE / "photo_log.csv"
PREFIX = "IMG"

MAX_BODY = 24 * 1024 * 1024       # 单次请求体上限
MAX_IMAGE = 16 * 1024 * 1024      # 单张图片上限
SUFFIX = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}
MAGIC = {
    "jpeg": lambda b: b.startswith(b"\xff\xd8"),
    "png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    "webp": lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP",
}
DATA_URL_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=\s]+)$")
LOG_FIELDS = ["time", "file", "width", "height", "bytes", "source", "raw_dir"]
NAME_RE = re.compile(r"^" + PREFIX + r"_(\d+)\.(jpg|jpeg|png|webp)$", re.I)

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------

def photo_files(raw: Path) -> list[Path]:
    if not raw.is_dir():
        return []
    return [p for p in raw.iterdir() if p.is_file() and NAME_RE.match(p.name)]


def next_index(raw: Path) -> int:
    mx = 0
    for p in photo_files(raw):
        m = NAME_RE.match(p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def append_log(log_path: Path, row: dict) -> None:
    """追加一行台账。手写 BOM，避免 utf-8-sig 追加时把 BOM 写进文件中间。"""
    fresh = not log_path.exists() or log_path.stat().st_size == 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        if fresh:
            f.write("\ufeff")  # Excel 打开中文不乱码
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("223.5.5.5", 80))  # UDP connect 不实际发包，只取本机出口 IP
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


def is_hotspot(ip: str) -> bool:
    """Windows 移动热点固定 192.168.137.1；Android 手机热点常见 192.168.43.x。"""
    return ip.startswith("192.168.137.") or ip.startswith("192.168.43.")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>拍照直传</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin:0; padding:16px; font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
         background:#f5f6f8; color:#1b1f24; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { font-size:12px; color:#5b6572; margin-bottom:14px; line-height:1.7; }
  .card { background:#fff; border:1px solid #e3e6ea; border-radius:12px; padding:14px; margin-bottom:12px; }
  .count { font-size:17px; font-weight:600; }
  .path { font-size:11px; color:#7a8491; word-break:break-all; margin-top:8px; line-height:1.5; }
  button { width:100%; min-height:78px; font-size:21px; font-weight:600; color:#fff;
           background:#2f6fed; border:0; border-radius:12px; }
  button:active { background:#2459c8; }
  button:disabled { background:#a9b6cc; }
  #file { display:none; }
  #msg { margin-top:10px; font-size:14px; min-height:22px; }
  #msg.ok { color:#177a45; } #msg.err { color:#c02c2c; }
  #thumb { display:block; max-width:100%; max-height:230px; margin:10px auto 0; border-radius:8px; }
  code { background:#eef1f5; padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<h1>拍照直传</h1>
<div class="sub">
  照片拍完直接存进<b>电脑</b>，手机不留文件、不装 App。<br>
  不分类，统一按 IMG_0001 连号命名。
</div>

<div class="card">
  <div class="count" id="count">已入库 - 张</div>
  <div class="path" id="raw">落盘目录：读取中…</div>
  <div class="path" id="urls">__URLS__</div>
</div>

<div class="card">
  <button id="shoot">拍照上传下一张</button>
  <input type="file" id="file" accept="image/*" capture="environment">
  <div id="msg"></div>
  <img id="thumb" alt="">
</div>

<script>
const MAX_SIDE = __MAX_SIDE__;
const QUALITY = __QUALITY__;
const $ = (id) => document.getElementById(id);
let busy = false;

function msg(t, kind) {
  $('msg').textContent = t;
  $('msg').className = kind || '';
}

async function refresh() {
  try {
    const r = await fetch('/status?t=' + Date.now());
    const j = await r.json();
    $('count').textContent = '已入库 ' + j.count + ' 张';
    $('raw').textContent = '落盘目录：' + j.raw;
  } catch (e) { msg('状态读取失败：' + e.message, 'err'); }
}

function compress(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => {
      const scale = Math.min(1, MAX_SIDE / Math.max(im.naturalWidth, im.naturalHeight));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(im.naturalWidth * scale));
      c.height = Math.max(1, Math.round(im.naturalHeight * scale));
      c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      // canvas 重编码会丢 EXIF：存下来的像素就是标注坐标的基准
      resolve({ url: c.toDataURL('image/jpeg', QUALITY), w: c.width, h: c.height });
    };
    im.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片解码失败')); };
    im.src = url;
  });
}

$('shoot').onclick = () => { if (!busy) $('file').click(); };
$('file').onchange = async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  busy = true; $('shoot').disabled = true; msg('正在处理并上传…');
  try {
    const c = await compress(f);
    const body = JSON.stringify({ img: c.url, w: c.w, h: c.h });
    const r = await fetch('/upload', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    $('thumb').src = c.url;
    msg('已保存 ' + j.file + '（第 ' + j.count + ' 张）', 'ok');
    await refresh();
  } catch (err) {
    msg('上传失败：' + err.message, 'err');
  } finally {
    e.target.value = '';   // 允许连续拍同名文件
    busy = false; $('shoot').disabled = false;
  }
};
refresh();
</script>
</body>
</html>
"""


def render_page(cfg: dict) -> str:
    urls = cfg["urls"] or ["http://<电脑IP>:" + str(cfg["port"])]
    links = "<br>".join(
        "服务地址 <code>" + u + "</code>" + ("（热点）" if is_hotspot(u.split("//")[1].split(":")[0]) else "")
        for u in urls
    )
    return (PAGE
            .replace("__URLS__", links)
            .replace("__MAX_SIDE__", str(cfg["max_side"]))
            .replace("__QUALITY__", str(cfg["quality"])))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, cfg):
        super().__init__(addr, handler)
        self.cfg = cfg


class Handler(BaseHTTPRequestHandler):
    server_version = "M23PhotoUpload/2.0"

    def log_message(self, fmt, *args):
        pass

    def reply(self, code, obj=None, text="", ctype="text/plain; charset=utf-8"):
        if obj is not None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        cfg = self.server.cfg
        p = urlsplit(self.path).path
        if p in ("/", "/index.html"):
            body = render_page(cfg).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif p == "/status":
            self.reply(200, {"ok": True, "count": len(photo_files(cfg["raw"])),
                            "raw": str(cfg["raw"]), "urls": cfg["urls"]})
        elif p == "/favicon.ico":
            self.reply(204)
        else:
            self.reply(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        cfg = self.server.cfg
        if urlsplit(self.path).path != "/upload":
            return self.reply(404, {"ok": False, "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"ok": False, "error": "bad length"})
        if n <= 0 or n > MAX_BODY:
            return self.reply(413, {"ok": False, "error": "请求体过大或为空"})
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return self.reply(400, {"ok": False, "error": "JSON 解析失败"})

        m = DATA_URL_RE.match(str(data.get("img", "")))
        if not m:
            return self.reply(400, {"ok": False, "error": "图片不是合法的 dataURL"})
        kind, b64 = m.group(1), m.group(2)
        try:
            blob = base64.b64decode(re.sub(r"\s+", "", b64), validate=True)
        except Exception:
            return self.reply(400, {"ok": False, "error": "base64 解码失败"})
        if not blob or len(blob) > MAX_IMAGE or not MAGIC[kind](blob):
            return self.reply(400, {"ok": False, "error": "图片内容不合法或过大"})
        try:
            w, h = int(data.get("w", 0)), int(data.get("h", 0))
        except (TypeError, ValueError):
            w = h = 0

        with _LOCK:
            raw = cfg["raw"]
            raw.mkdir(parents=True, exist_ok=True)
            name = f"{PREFIX}_{next_index(raw):04d}{SUFFIX[kind]}"
            path = raw / name
            path.write_bytes(blob)
            append_log(cfg["log"], {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": name, "width": w, "height": h,
                "bytes": len(blob), "source": "wifi", "raw_dir": str(raw),
            })
            count = len(photo_files(raw))
        print(f"[上传] {name}  {w}x{h}  {len(blob)//1024} KB  -> {path}", flush=True)
        self.reply(200, {"ok": True, "file": name, "count": count})


# ---------------------------------------------------------------------------
# 自检与启动
# ---------------------------------------------------------------------------

def self_test() -> None:
    """不碰项目数据：临时目录 + 随机端口，端到端跑一遍 GET/POST/落盘/台账。"""
    import tempfile
    import urllib.error
    import urllib.request

    with tempfile.TemporaryDirectory(prefix="m23_upload_") as td:
        root = Path(td)
        cfg = {"raw": root / "raw", "log": root / "photo_log.csv",
               "max_side": 1706, "quality": 0.9,
               "port": 0, "urls": ["http://127.0.0.1:0"]}
        srv = Server(("127.0.0.1", 0), Handler, cfg)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{port}"
        checks = 0

        def check(cond, label):
            nonlocal checks
            assert cond, "FAIL: " + label
            checks += 1
            print(f"  [OK] {label}")

        def post(payload):
            req = urllib.request.Request(
                base + "/upload", data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode("utf-8"))

        try:
            with urllib.request.urlopen(base + "/", timeout=5) as r:
                html = r.read().decode("utf-8")
            check(r.status == 200 and "拍照上传" in html and "不分类" in html,
                  "首页可访问且含拍照按钮、无批次分类")
            with urllib.request.urlopen(base + "/status", timeout=5) as r:
                check(json.loads(r.read())["count"] == 0, "初始计数为 0")

            png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDw"
                   "AEhQGAhKmMIQAAAABJRU5ErkJggg==")
            good = {"img": "data:image/png;base64," + png, "w": 1, "h": 1}
            code, j = post(good)
            check(code == 200 and j.get("ok") and j.get("count") == 1, "上传成功，计数变 1")
            check(j.get("file") == "IMG_0001.png", "平铺命名 IMG_0001.png（无分类子目录）")
            f1 = cfg["raw"] / "IMG_0001.png"
            check(f1.is_file() and f1.read_bytes().startswith(b"\x89PNG"), "文件真实落盘且字节正确")
            txt = cfg["log"].read_text(encoding="utf-8-sig")
            check("IMG_0001.png" in txt and "time,file,width" in txt, "台账含表头与文件名")

            code, j = post(good)
            check(code == 200 and j.get("file") == "IMG_0002.png" and j.get("count") == 2,
                  "连续上传序号自增、不覆盖")
            code, _ = post(dict(good, img="hello"))
            check(code == 400, "非法图片被拒绝")
            code, _ = post(dict(good, img="data:image/png;base64,AAAA"))
            check(code == 400, "magic bytes 不符被拒绝")
            with urllib.request.urlopen(base + "/status", timeout=5) as r:
                check(json.loads(r.read())["count"] == 2, "状态接口计数正确")
            print(f"SELF-TEST ALL PASS ({checks} 项)")
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(timeout=3)


def print_banner(cfg: dict, ips: list[str]) -> None:
    print("=" * 66)
    print("M2-3 拍照直传 · 服务已启动（不分类，照片平铺连号）")
    print("=" * 66)
    print(f"脚本位置  {Path(__file__).resolve()}")
    print(f"照片落盘  {cfg['raw']}\\IMG_0001.jpg")
    print(f"台账      {cfg['log']}")
    print("手机浏览器打开下面任一地址（手机要和电脑在【同一个网】）：")
    for ip in ips:
        tag = "   <== 热点地址，手机连电脑热点时用这个" if is_hotspot(ip) else ""
        print(f"    http://{ip}:{cfg['port']}{tag}")
    print("-" * 66)
    print("打不开时按顺序检查：")
    print("  1) 这个黑窗口不能关，关了服务就停（看到本行说明服务在跑）")
    print("  2) 电脑和手机必须在同一个网：电脑开移动热点 -> 手机连它；")
    print("     或手机开热点 -> 电脑的 WiFi 也切过去（不是只开热点不连）")
    print("  3) 首次启动若弹「Windows 安全中心警报」，勾【专用网络】点【允许访问】")
    print("  4) 上面没有 192.168.137.x / 192.168.43.x 说明热点没建起来，")
    print("     建立热点后本窗口会自动打印新地址，用新地址，别用旧地址")
    print("  5) 地址要手输 http:// 开头，别让浏览器自动补成 https")
    print("  6) 实在连不上：右键管理员运行 allow_firewall.bat 放行端口")
    print("Ctrl+C 停止服务。")
    print("=" * 66, flush=True)


def watch_ips(cfg: dict, stop: threading.Event) -> None:
    """运行中检测到新网卡（例如刚开的热点）就打印新地址并更新首页。"""
    seen = set(cfg["ips"])
    while not stop.wait(4):
        cur = set(lan_ips())
        new = cur - seen
        if new:
            seen |= new
            cfg["urls"] = [f"http://{ip}:{cfg['port']}" for ip in sorted(seen)]
            for ip in sorted(new):
                tag = "   <== 新出现的热点地址，用这个！" if is_hotspot(ip) else ""
                print(f"[网络变化] http://{ip}:{cfg['port']}{tag}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="M2-3 拍照直传（电脑端，不分类）")
    ap.add_argument("--port", type=int, default=8000, help="端口，默认 8000")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW, help="照片目录，默认 WSL 项目的 M2-3/data/raw")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="台账 CSV，默认脚本同目录 photo_log.csv")
    ap.add_argument("--max-side", type=int, default=1706, help="前端压缩最长边，默认 1706")
    ap.add_argument("--quality", type=float, default=0.9, help="JPEG 质量，默认 0.9")
    ap.add_argument("--self-test", action="store_true", help="端到端自检，不碰项目数据")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    raw = Path(args.raw_dir)
    log = Path(args.log)
    try:
        raw.mkdir(parents=True, exist_ok=True)
        if not raw.is_dir():
            raise OSError("不是目录")
        log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[错误] 目标目录不可用：{raw}\n  {e}")
        print("  确认 WSL 已启动、UNC 路径可写；或改用 --raw-dir 指向电脑本地文件夹。")
        sys.exit(1)

    ips = lan_ips()
    cfg = {"raw": raw, "log": log, "max_side": args.max_side,
           "quality": args.quality, "port": args.port,
           "ips": ips, "urls": [f"http://{ip}:{args.port}" for ip in ips]}
    try:
        srv = Server(("0.0.0.0", args.port), Handler, cfg)
    except OSError as e:
        print(f"[错误] 端口 {args.port} 启动失败：{e}")
        print("  可能端口被占用，换一个：python photo_upload.py --port 8080")
        sys.exit(1)

    print_banner(cfg, ips)
    stop = threading.Event()
    threading.Thread(target=watch_ips, args=(cfg, stop), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()
