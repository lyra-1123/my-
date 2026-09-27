# -*- coding: utf-8 -*-
"""
从 Dukascopy 原始数据源（datafeed.dukascopy.com，与网页版 Historical Data Export 同源）批量下载 M1 K 线，
按年输出与 data/ 现有文件相同的格式：无表头、分号分隔  YYYYMMDD HHMMSS;open;high;low;close;volume（UTC）。

只用 Python 标准库。需要能访问 datafeed.dukascopy.com 的网络（在本地电脑运行）。

用法（在仓库根目录）：
  1) 自检：下载几天 BID，与 data/ 里已有的 BID 文件逐分钟比对（验证字段顺序、价格单位、时区）
       python scripts/dukascopy_download.py --check
  2) 下载 ASK（2009 至今，约 6500 个交易日，断点续传）
       python scripts/dukascopy_download.py --side ASK --start 2009-01-01 --end 2026-09-26
  3) 白银
       python scripts/dukascopy_download.py --symbol XAGUSD --side BID --start 2009-01-01 --end 2026-09-26
       python scripts/dukascopy_download.py --symbol XAGUSD --side ASK --start 2009-01-01 --end 2026-09-26

输出：data/DAT_ASCII_<品种>_<BID|ASK>_M1_<年份>.csv（名字里带 BID/ASK，不会与现有的 DAT_ASCII_XAUUSD_M1_* 混在一起）。
逐日原始文件缓存在 data/raw_dukascopy/，中断后重跑会跳过已下载的日期。

原始格式（每日一个 LZMA 压缩文件 .../<品种>/<年>/<月-1，两位>/<日>/<BID|ASK>_candles_min_1.bi5）：
  每条 24 字节，大端：int32 距当日 00:00 UTC 的秒数、int32 开、int32 收、int32 低、int32 高、float32 成交量；
  价格 = 整数 / 价格除数（XAUUSD、XAGUSD 为 1000）。字段顺序以 --check 的比对结果为准。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DIVISOR = {"XAUUSD": 1000, "XAGUSD": 1000}
REC = struct.Struct(">5if")


def fetch(sym: str, side: str, day: dt.date, raw_dir: str, retries: int = 5) -> bytes:
    """下载某日原始文件（带缓存与重试）；周末/无数据返回 b""。"""
    path = os.path.join(raw_dir, sym, side, f"{day:%Y%m%d}.bi5")
    if os.path.exists(path):
        return open(path, "rb").read()
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day, side=side)   # 月份从 0 开始
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = b""
                break
            time.sleep(2 ** k)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** k)
    else:
        raise RuntimeError(f"下载失败（已重试 {retries} 次）：{url}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        fh.write(data)
    os.replace(path + ".part", path)
    return data


def parse(data: bytes, day: dt.date, div: float) -> list[tuple]:
    """返回 [(UTC datetime, open, high, low, close, volume), ...]。"""
    if not data:
        return []
    raw = lzma.decompress(data)
    base = dt.datetime(day.year, day.month, day.day)
    out = []
    for sec, o, c, lo, hi, v in REC.iter_unpack(raw):
        out.append((base + dt.timedelta(seconds=sec), o / div, hi / div, lo / div, c / div, v))
    return out


def fmt(row) -> str:
    t, o, h, l, c, v = row
    return f"{t:%Y%m%d %H%M%S};{o:.3f};{h:.3f};{l:.3f};{c:.3f};{v:.5f}"


def days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() != 5:                       # 周六全天休市；周日晚开盘，保留
            yield d
        d += dt.timedelta(days=1)


def download(sym: str, side: str, start: dt.date, end: dt.date, out_dir: str, workers: int) -> None:
    div = DIVISOR.get(sym)
    if div is None:
        sys.exit(f"未知品种 {sym} 的价格除数，请在 DIVISOR 里补充")
    raw_dir = os.path.join(out_dir, "raw_dukascopy")
    all_days = list(days(start, end))
    by_year: dict[int, list] = {}
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for day, data in zip(all_days, ex.map(lambda d: fetch(sym, side, d, raw_dir), all_days)):
            by_year.setdefault(day.year, []).extend(parse(data, day, div))
            done += 1
            if done % 100 == 0 or done == len(all_days):
                print(f"  {sym} {side}: {done}/{len(all_days)} 天（{day}）", flush=True)
    for y, rows in sorted(by_year.items()):
        rows.sort(key=lambda r: r[0])
        path = os.path.join(out_dir, f"DAT_ASCII_{sym}_{side}_M1_{y}.csv")
        with open(path, "w", encoding="ascii") as fh:
            fh.write("\n".join(fmt(r) for r in rows) + "\n")
        nz = sum(1 for r in rows if r[5] > 0)
        print(f"写出 {path}：{len(rows)} 行（有成交量 {nz} 行）")


def check(out_dir: str) -> None:
    """下载若干天 BID，与 data/ 现有 BID 文件逐分钟比对。"""
    files = sorted(glob.glob(os.path.join(out_dir, "DAT_ASCII_XAUUSD_M1_*.csv")))
    if not files:
        sys.exit("data/ 下没有现有的 DAT_ASCII_XAUUSD_M1_*.csv，无法自检；可跳过自检，下载一个月 ASK 上传后由研究端核对")
    ref = {}
    for f in [f for f in files if "2024" in f or "2026" in f]:
        for line in open(f, encoding="utf-8"):
            p = line.strip().split(";")
            if len(p) == 6:
                ref[p[0]] = tuple(float(x) for x in p[1:5])
    test_days = [dt.date(2024, 3, 12), dt.date(2024, 11, 6), dt.date(2026, 9, 24)]
    raw_dir = os.path.join(out_dir, "raw_dukascopy")
    n = bad = 0
    worst = 0.0
    for day in test_days:
        for t, o, h, l, c, v in parse(fetch("XAUUSD", "BID", day, raw_dir), day, DIVISOR["XAUUSD"]):
            k = f"{t:%Y%m%d %H%M%S}"
            if k in ref and v > 0:
                n += 1
                diff = max(abs(a - b) for a, b in zip((o, h, l, c), ref[k]))
                worst = max(worst, diff)
                bad += diff > 0.0015
    if n == 0:
        sys.exit("自检失败：下载的数据与现有文件没有任何时间戳重合（时区或日期对不上），请把输出发给研究端")
    print(f"自检：比对 {n} 分钟，价格不一致 {bad} 分钟，最大差 {worst:.4f}")
    print("自检 OK，可以下载 ASK" if bad <= n * 0.001 else "自检 FAIL：字段顺序或价格单位不对，请把输出发给研究端，先不要下载")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dukascopy M1 批量下载")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--side", default="ASK", choices=["BID", "ASK"])
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default=str(dt.date.today() - dt.timedelta(days=1)))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="只做自检（与现有 BID 文件比对）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        check(a.out)
    else:
        download(a.symbol.upper(), a.side, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.out, a.workers)


if __name__ == "__main__":
    main()
