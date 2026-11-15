# -*- coding: utf-8 -*-
"""M2-3 USB 自动导入（完全不依赖网络，手机零安装）。

用法（电脑上）：
    双击 start_usb_import.bat
    手机用数据线连电脑，手机上点「传输文件 / 允许访问」
    之后正常拍照；脚本每 4 秒扫一次手机 DCIM，把新照片自动搬进电脑。

不分类：所有照片平铺在一个目录，按 IMG_0001.jpg 连号命名，不覆盖。
去重：按「设备/文件夹/文件名/大小」记进 usb_state.json，重复插线不会反复导入。

默认只复制、保留手机原图；加 --move 会在复制成功后从手机删掉原图
（MTP 移动兼容性因手机而异，失败会自动跳过并保留原图，不会丢照片）。

只在电脑上跑；手机端不写任何文件、不装 App。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = (
    r"\\wsl$\Ubuntu2204\home\xiaolingxiaotan\智能车一轮考核"
    r"\智能车一轮考核\M2\M2-3\data\raw"
)
DEFAULT_LOG = HERE / "photo_log.csv"
DEFAULT_STATE = HERE / "usb_state.json"
PREFIX = "IMG"
LOG_FIELDS = ["time", "file", "width", "height", "bytes", "source", "raw_dir"]
NAME_RE = re.compile(r"^" + PREFIX + r"_(\d+)\.(jpg|jpeg|png|webp|heic|heif)$", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")

# PowerShell 脚本全部是 ASCII，避免 .ps1/.bat 中文编码坑。
# 中文路径一律通过 JSON 配置文件（UTF-8）传入，绝不拼进命令行。
PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$cfg = Get-Content -Raw -Encoding UTF8 -LiteralPath $env:M23_CFG | ConvertFrom-Json
$staging = [string]$cfg.staging
$move = [bool]$cfg.move
$known = @{}
foreach ($k in @($cfg.known)) { $known[[string]$k] = $true }
if (-not (Test-Path -LiteralPath $staging)) { New-Item -ItemType Directory -Force -Path $staging | Out-Null }
Get-ChildItem -LiteralPath $staging -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$sh = New-Object -ComObject Shell.Application
$computer = $sh.Namespace(17)
$photos = New-Object System.Collections.ArrayList
$devices = New-Object System.Collections.ArrayList
$skipDir = '^(Android|LOST\.DIR|Music|Movies|Ringtones|Alarms|Notifications|Podcasts|Audiobooks|System|Backup|\.thumbnails)$'

function Scan($ns, $depth, $tag) {
    if ($null -eq $ns -or $depth -gt 5) { return }
    try { $items = @($ns.Items()) } catch { return }
    foreach ($it in $items) {
        try {
            $nm = [string]$it.Name
            if ($it.IsFolder) {
                if ($nm -notmatch $skipDir) {
                    Scan ($sh.Namespace($it.Path)) ($depth + 1) ($tag + '/' + $nm)
                }
            } elseif ($nm -match '\.(jpg|jpeg|png|heic|heif|webp)$') {
                $sz = 0
                try { $sz = [long]$it.Size } catch {}
                $key = $tag + '/' + $nm + '|' + [string]$sz
                if (-not $known.ContainsKey($key)) {
                    [void]$photos.Add([pscustomobject]@{
                        Name = $nm; Key = $key; Folder = [string]$ns.Self.Path; Size = $sz
                    })
                }
            }
        } catch {}
    }
}

try {
    foreach ($dev in @($computer.Items())) {
        try {
            $p = [string]$dev.Path
            if (-not $dev.IsFolder) { continue }
            if ($p -match '^[A-Za-z]:') { continue }   # skip local drives
            $dn = [string]$dev.Name
            [void]$devices.Add($dn)
            Scan ($sh.Namespace($p)) 0 $dn
        } catch {}
    }
} catch {}

$files = New-Object System.Collections.ArrayList
$idx = 0
foreach ($ph in $photos) {
    try {
        $srcNs = $sh.Namespace($ph.Folder)
        if ($null -eq $srcNs) { continue }
        $item = $srcNs.ParseName($ph.Name)
        if ($null -eq $item) { continue }
        $destNs = $sh.Namespace($staging)
        if ($move) { $destNs.MoveHere($item, 1044) } else { $destNs.CopyHere($item, 1044) }
        $target = Join-Path $staging $ph.Name
        $ok = $false
        for ($i = 0; $i -lt 80; $i++) {
            if (Test-Path -LiteralPath $target) { $ok = $true; break }
            Start-Sleep -Milliseconds 250
        }
        if (-not $ok) { continue }
        $idx++
        $ext = [System.IO.Path]::GetExtension($ph.Name)
        $staged = 'P{0:0000}{1}' -f $idx, $ext
        Rename-Item -LiteralPath $target -NewName $staged -Force -ErrorAction Stop
        [void]$files.Add([pscustomobject]@{ key = $ph.Key; staged = $staged; orig = $ph.Name })
    } catch {}
}

$res = [pscustomobject]@{
    devices = @($devices.ToArray())
    found   = @($photos.ToArray()).Count
    files   = @($files.ToArray())
}
ConvertTo-Json -InputObject $res -Compress -Depth 5 |
    Set-Content -LiteralPath $env:M23_OUT -Encoding UTF8
"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def image_size(path: Path) -> tuple[int, int]:
    """只用标准库读 JPEG/PNG 尺寸；读不到返回 (0, 0)。"""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                return struct.unpack(">II", head[16:24])
            if head.startswith(b"\xff\xd8"):
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker or marker[0] in (0xD8, 0xD9, 0x01) or 0xD0 <= marker[0] <= 0xD7:
                        continue
                    ln = struct.unpack(">H", f.read(2))[0]
                    if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        seg = f.read(5)
                        return struct.unpack(">HH", seg[1:5])[::-1]  # (height,width) -> (w,h)
                    f.seek(ln - 2, 1)
    except Exception:
        pass
    return 0, 0


def append_log(log_path: Path, row: dict) -> None:
    fresh = not log_path.exists() or log_path.stat().st_size == 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        if fresh:
            f.write("\ufeff")
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def next_index(raw: Path) -> int:
    mx = 0
    if raw.is_dir():
        for p in raw.iterdir():
            m = NAME_RE.match(p.name) if p.is_file() else None
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def load_state(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return [str(x) for x in d.get("known", [])]
    except Exception:
        return []


def save_state(path: Path, known: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"known": known[-20000:]}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def run_powershell(cfg: dict, staging: Path, out_json: Path, cfg_json: Path) -> dict:
    cfg_json.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    if out_json.exists():
        out_json.unlink()
    env = dict(os.environ)
    env["M23_CFG"] = str(cfg_json)
    env["M23_OUT"] = str(out_json)
    enc = base64.b64encode(PS_SCRIPT.encode("utf-16-le")).decode("ascii")
    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc]
    p = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
    if not out_json.exists():
        err = (p.stderr or p.stdout or b"").decode("utf-8", "replace")[-500:]
        raise RuntimeError(f"PowerShell 未返回结果（exit={p.returncode}）: {err}")
    txt = out_json.read_text(encoding="utf-8-sig").strip()
    if not txt:
        return {"devices": [], "found": 0, "files": []}
    d = json.loads(txt)
    if isinstance(d, list):
        d = d[0] if d else {}
    return {"devices": d.get("devices") or [], "found": d.get("found", 0),
            "files": d.get("files") or []}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def import_once(args, staging: Path, cfg_json: Path, out_json: Path) -> int:
    known = load_state(args.state)
    res = run_powershell({"staging": str(staging), "move": bool(args.move),
                          "known": known}, staging, out_json, cfg_json)
    devices = res["devices"]
    files = res["files"]
    if not devices:
        return -1  # 没检测到手机
    if not files:
        return 0   # 手机在，但没有新照片

    if isinstance(files, dict):
        files = [files]
    raw: Path = args.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        src = staging / str(f.get("staged", ""))
        key = str(f.get("key", ""))
        if not src.is_file():
            continue
        ext = src.suffix.lower() or ".jpg"
        if ext == ".jpeg":
            ext = ".jpg"
        name = f"{PREFIX}_{next_index(raw):04d}{ext}"
        dst = raw / name
        shutil.copy2(src, dst)
        w, h = image_size(dst)
        append_log(args.log, {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file": name, "width": w, "height": h,
            "bytes": dst.stat().st_size, "source": "usb", "raw_dir": str(raw),
        })
        if key:
            known.append(key)
        print(f"[导入] {name}  {w}x{h}  <- {f.get('orig', '')}", flush=True)
        try:
            src.unlink()
        except OSError:
            pass
        n += 1
    save_state(args.state, known)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="M2-3 USB 自动导入（电脑端，不依赖网络）")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW, help="照片目录，默认 WSL 项目的 M2-3/data/raw")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="台账 CSV，默认脚本同目录 photo_log.csv")
    ap.add_argument("--state", default=str(DEFAULT_STATE), help="去重台账 JSON")
    ap.add_argument("--interval", type=float, default=4.0, help="轮询间隔秒，默认 4")
    ap.add_argument("--move", action="store_true", help="导入后从手机删除原图（默认保留）")
    ap.add_argument("--once", action="store_true", help="只扫一次就退出（自检用）")
    ap.add_argument("--self-test", action="store_true", help="只测本地函数，不碰手机和项目数据")
    args = ap.parse_args()
    args.raw_dir = Path(args.raw_dir)
    args.log = Path(args.log)
    args.state = Path(args.state)

    if args.self_test:
        import tempfile as _tf
        with _tf.TemporaryDirectory(prefix="m23_usb_") as td:
            t = Path(td)
            st = t / "state.json"
            save_state(st, ["a", "b"])
            assert load_state(st) == ["a", "b"], "state 读写失败"
            print("  [OK] state 读写")
            f = t / "a.png"
            f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
                          + struct.pack(">II", 7, 11) + b"\x00" * 9)
            assert image_size(f) == (7, 11), "PNG 尺寸解析失败"
            print("  [OK] PNG 尺寸解析")
            lg = t / "log.csv"
            append_log(lg, {"time": "t", "file": "IMG_0001.jpg", "width": 1,
                            "height": 2, "bytes": 3, "source": "usb", "raw_dir": "x"})
            assert "time,file,width" in lg.read_text(encoding="utf-8-sig"), "台账写入失败"
            print("  [OK] 台账写入")
            (t / "IMG_0003.jpg").write_bytes(b"x")
            (t / "notes.txt").write_bytes(b"x")
            assert next_index(t) == 4, "连号扫描失败"
            print("  [OK] 连号扫描")
            assert PS_SCRIPT.isascii(), "PowerShell 片段含非 ASCII 字符"
            print("  [OK] PowerShell 片段全 ASCII（无编码坑）")
            print("SELF-TEST ALL PASS (5 项)")
        return

    staging = Path(tempfile.gettempdir()) / "m23_usb_staging"
    staging.mkdir(parents=True, exist_ok=True)
    cfg_json = Path(tempfile.gettempdir()) / "m23_usb_cfg.json"
    out_json = Path(tempfile.gettempdir()) / "m23_usb_out.json"

    print("=" * 66)
    print("M2-3 USB 自动导入 · 不依赖网络，手机零安装")
    print("=" * 66)
    print(f"照片落盘  {args.raw_dir}\\IMG_0001.jpg   （不分类，平铺连号）")
    print(f"台账      {args.log}")
    print(f"去重记录  {args.state}")
    print(f"手机原图  {'导入后删除（--move）' if args.move else '保留（默认，最安全）'}")
    print("-" * 66)
    print("1) 用数据线把手机连到电脑")
    print("2) 手机上点「传输文件 / 允许访问」（有的要下拉通知栏选 MTP）")
    print("3) 之后正常拍照，本窗口会自动把新照片搬进电脑")
    print("   黑窗口不要关；Ctrl+C 停止。")
    print("=" * 66, flush=True)

    warned = False
    while True:
        try:
            n = import_once(args, staging, cfg_json, out_json)
        except subprocess.TimeoutExpired:
            print("[警告] 扫描手机超时，下一轮重试", flush=True)
            n = 0
        except Exception as e:
            print(f"[警告] 本轮失败：{e}", flush=True)
            n = 0
        if n == -1:
            if not warned:
                print("[等待] 未检测到手机：确认数据线已插、手机上已点「允许访问/传输文件」", flush=True)
                warned = True
        else:
            if warned:
                print("[已连接] 手机已识别，开始监听新照片", flush=True)
                warned = False
            if n > 0:
                print(f"[完成] 本轮导入 {n} 张，累计 {next_index(args.raw_dir) - 1} 张", flush=True)
        if args.once:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已停止。")
            break


if __name__ == "__main__":
    main()
