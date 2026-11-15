# -*- coding: utf-8 -*-
r"""M2-3 无线同步收件箱 -> 自动入库（电脑端，与传输协议无关）。

设计要点：**传输协议和入库彻底解耦**。
手机端不管用 Syncthing / FolderSync(WebDAV) / PhotoSync(FTP) / 网页上传，
只要把照片放进本脚本的「收件箱」目录，脚本就会自动：
    递归扫描新照片 -> 等文件写完整 -> 内容哈希去重 -> 改名 IMG_0001 平铺进 data/raw
    -> 写 photo_log.csv 台账，不分类、不覆盖、可反复运行。

为什么这样分层：换手机 App、换协议、换网络，都不用改这一段。

用法：
    双击 start_wireless_import.bat
或  python wireless_import.py --once          # 只扫一次
    python wireless_import.py --self-test    # 自检，不碰项目数据

默认目录（都在电脑桌面那个文件夹里，除了最终库在 WSL 项目）：
    收件箱  <脚本目录>/inbox/                <-- 手机端同步目标指向这里
    最终库  \\wsl$\...\M2-3\data\raw\IMG_0001.jpg
    台账    <脚本目录>/photo_log.csv
    去重    <脚本目录>/wireless_state.json

安全约定：
    * 默认**保留**收件箱原文件，靠内容 SHA1 去重，绝不删手机上的照片。
    * --consume 会把已入库文件移到 inbox/_done/（仍可恢复），仅在手机端
      配了「仅发送 / 单向上传」时才可开启；双向同步下移动会被同步成删除。
    * 不改动像素、不做 EXIF 转正：标注器用 exif_transpose 读图，训练侧读图
      也必须做同样的转正，否则归一化框会整体偏掉。见 docs/data.md 记录口径。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INBOX = HERE / "inbox"
DEFAULT_RAW = (
    r"\\wsl$\Ubuntu2204\home\xiaolingxiaotan\智能车一轮考核"
    r"\智能车一轮考核\M2\M2-3\data\raw"
)
DEFAULT_LOG = HERE / "photo_log.csv"
DEFAULT_STATE = HERE / "wireless_state.json"

PREFIX = "IMG"
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
NAME_RE = re.compile(r"^" + PREFIX + r"_(\d+)\.(jpg|jpeg|png|heic|heif|webp)$", re.I)
LOG_FIELDS = ["time", "file", "width", "height", "bytes", "source", "raw_dir"]


# ---------------------------------------------------------------------------
# 基础工具
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
                        return struct.unpack(">HH", seg[1:5])[::-1]
                    f.seek(ln - 2, 1)
    except Exception:
        pass
    return 0, 0


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_log(log_path: Path, row: dict) -> None:
    fresh = not log_path.exists() or log_path.stat().st_size == 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        if fresh:
            f.write("\ufeff")  # Excel 打开中文不乱码
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


def load_state(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return set(str(x) for x in d.get("hashes", []))
    except Exception:
        return set()


def save_state(path: Path, hashes: set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"hashes": sorted(hashes)}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def iter_images(inbox: Path, done: Path):
    """递归列出收件箱里的图片，跳过 _done 归档目录和隐藏文件。"""
    if not inbox.is_dir():
        return
    for p in inbox.rglob("*"):
        try:
            if not p.is_file() or p.name.startswith("."):
                continue
            if done in p.parents:
                continue
            if p.suffix.lower() not in IMG_EXT:
                continue
            yield p
        except OSError:
            continue


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------

class Importer:
    def __init__(self, inbox: Path, raw: Path, log: Path, state: Path,
                 source: str = "sync", consume: bool = False, min_age: float = 1.0):
        self.inbox = inbox
        self.raw = raw
        self.log = log
        self.state_path = state
        self.source = source
        self.consume = consume
        self.min_age = min_age
        self.done = inbox / "_done"
        self.hashes = load_state(state)
        self.pending: dict[str, tuple[int, int]] = {}

    def scan(self) -> int:
        """扫一轮。返回本轮入库张数。文件要连续两轮签名一致且写完才处理。"""
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)
        now = time.time()
        seen: set[str] = set()
        to_archive: list[Path] = []
        n = 0
        for p in iter_images(self.inbox, self.done):
            key = str(p)
            seen.add(key)
            try:
                st = p.stat()
                sig = (st.st_size, st.st_mtime_ns)
            except OSError:
                continue
            # 连续两轮一致 -> 判定文件已写完整
            if self.pending.get(key) != sig:
                self.pending[key] = sig
                continue
            if self.min_age > 0 and now - st.st_mtime < self.min_age:
                continue
            try:
                h = sha1_of(p)
            except OSError:
                continue
            self.pending.pop(key, None)
            if h in self.hashes:
                if self.consume:
                    to_archive.append(p)   # 之前已入库的，开启 consume 后一并归档
                continue
            ext = p.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            name = f"{PREFIX}_{next_index(self.raw):04d}{ext}"
            dst = self.raw / name
            shutil.copy2(p, dst)
            w, hh = image_size(dst)
            append_log(self.log, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": name, "width": w, "height": hh,
                "bytes": dst.stat().st_size, "source": self.source,
                "raw_dir": str(self.raw),
            })
            self.hashes.add(h)
            n += 1
            print(f"[入库] {name}  {w}x{hh}  <- {p.relative_to(self.inbox)}", flush=True)
            if self.consume:
                to_archive.append(p)
        # 遍历结束后统一归档，避免 rglob 生成器遍历到一半被移动
        if to_archive:
            self.done.mkdir(parents=True, exist_ok=True)
            for p in to_archive:
                target = self.done / p.name
                k = 1
                while target.exists():
                    target = self.done / f"{p.stem}_{k}{p.suffix}"
                    k += 1
                try:
                    shutil.move(str(p), str(target))
                    self.pending.pop(str(p), None)
                except OSError as e:
                    print(f"[警告] 归档失败（原文件保留）：{e}", flush=True)
        # 清掉已消失文件的 pending
        for k in list(self.pending):
            if k not in seen:
                self.pending.pop(k, None)
        if n:
            save_state(self.state_path, self.hashes)
            print(f"[完成] 本轮入库 {n} 张，累计 {next_index(self.raw) - 1} 张", flush=True)
        return n


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="m23_wifi_") as td:
        t = Path(td)
        inbox, raw = t / "inbox", t / "raw"
        (inbox / "Camera").mkdir(parents=True)
        log, state = t / "photo_log.csv", t / "state.json"
        png = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
               + struct.pack(">II", 7, 11) + b"\x00" * 9)
        (inbox / "Camera" / "a.png").write_bytes(png)
        (inbox / "b.jpg").write_bytes(b"\xff\xd8\xff\xd9" + b"X" * 100)
        (inbox / "dup.png").write_bytes(png)          # 同内容不同名 -> 应去重
        (inbox / "notes.txt").write_bytes(b"hello")   # 非图片 -> 忽略
        imp = Importer(inbox, raw, log, state, source="selftest", min_age=0)
        imp.scan()   # 第一轮只登记签名
        n = imp.scan()
        assert n == 2, f"应入库 2 张，实际 {n}"
        png_out = [p for p in raw.iterdir() if p.suffix.lower() == ".png"]
        assert len(png_out) == 1 and image_size(png_out[0]) == (7, 11), "PNG 尺寸解析失败"
        assert any(p.suffix.lower() == ".jpg" for p in raw.iterdir()), "JPEG 未入库"
        assert {p.name for p in raw.iterdir()} == {"IMG_0001.png", "IMG_0002.jpg"} or \
               {p.name for p in raw.iterdir()} == {"IMG_0001.jpg", "IMG_0002.png"}, "连号命名异常"
        assert len(load_state(state)) == 2, "去重 state 异常"
        print("  [OK] 递归扫描 / 写完整判定 / 内容哈希去重 / 连号平铺")
        assert imp.scan() == 0, "重复扫描不应重复入库"
        print("  [OK] 重复运行不重复入库")
        lines = [x for x in log.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        assert len(lines) == 3 and "time,file,width" in lines[0], "台账异常"
        print("  [OK] 台账表头与行数")
        # consume 模式把原文件移到 _done，不删除
        (inbox / "c.png").write_bytes(png + b"unique")
        imp.consume = True
        imp.scan(); imp.scan()
        assert (inbox / "_done" / "a.png").is_file(), "consume 未归档"
        assert (inbox / "Camera" / "a.png").exists() is False, "consume 未移走源文件"
        print("  [OK] --consume 归档到 _done 而非删除")
        print("SELF-TEST ALL PASS (5 项)")


def main() -> None:
    ap = argparse.ArgumentParser(description="M2-3 无线收件箱自动入库（与传输协议无关）")
    ap.add_argument("--inbox", default=str(DEFAULT_INBOX), help="收件箱目录，手机端同步目标指向这里")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW, help="最终库，默认 WSL 项目的 M2-3/data/raw")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="台账 CSV")
    ap.add_argument("--state", default=str(DEFAULT_STATE), help="去重记录 JSON")
    ap.add_argument("--source", default="sync", help="台账里的来源标记")
    ap.add_argument("--interval", type=float, default=3.0, help="扫描间隔秒，默认 3")
    ap.add_argument("--min-age", type=float, default=1.0, help="文件写完多久才处理，默认 1 秒")
    ap.add_argument("--consume", action="store_true",
                    help="已入库原文件移到 inbox/_done/（仅当手机端为单向上传时开启）")
    ap.add_argument("--once", action="store_true", help="只扫一轮就退出")
    ap.add_argument("--self-test", action="store_true", help="自检，不碰项目数据")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    inbox = Path(args.inbox)
    raw = Path(args.raw_dir)
    log = Path(args.log)
    state = Path(args.state)
    try:
        inbox.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        if not raw.is_dir():
            raise OSError("最终库不是目录")
    except OSError as e:
        print(f"[错误] 目录不可用：{e}\n  确认 WSL 已启动、UNC 路径可写；或用 --raw-dir 指向本地目录。")
        sys.exit(1)

    imp = Importer(inbox, raw, log, state, source=args.source,
                   consume=args.consume, min_age=args.min_age)
    print("=" * 68)
    print("M2-3 无线收件箱 · 自动入库（传输协议无关）")
    print("=" * 68)
    print(f"收件箱    {inbox}")
    print("           ^^^ 手机端 Syncthing / FolderSync / PhotoSync 的同步目标指这里")
    print(f"最终库    {raw}\\IMG_0001.jpg   （不分类，平铺连号）")
    print(f"台账      {log}")
    print(f"去重      {state}  （已记录 {len(imp.hashes)} 张）")
    print(f"原文件    {'入库后移到 inbox/_done/' if args.consume else '保留在收件箱（默认，最安全）'}")
    print("-" * 68)
    print("手机端只要把新照片送进收件箱，这边自动改名入库，无需任何操作。")
    print("换协议、换 App、换网络都不用改本脚本。Ctrl+C 停止。")
    print("=" * 68, flush=True)

    while True:
        try:
            imp.scan()
            if args.once:
                # 单次模式也要连扫两轮：第一轮登记签名，第二轮才判定写完并入库
                imp.min_age = 0
                imp.scan()
        except Exception as e:
            print(f"[警告] 本轮失败：{e}", flush=True)
        if args.once:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已停止。")
            break


if __name__ == "__main__":
    main()
