# -*- coding: utf-8 -*-
"""
Dukascopy 导出的 XAUUSD M1 数据加载 + 重采样。

文件格式（dukascopy_export_*.py 输出）：
    无表头，分号分隔：  YYYYMMDD HHMMSS;open;high;low;close;volume
    价格为 BID 价，volume 为 Dukascopy 成交量，时间为 UTC。
文件名：DAT_ASCII_XAUUSD_M1_<年份>[_DUKAREAL].csv（也支持 .csv.gz）
"""
from __future__ import annotations

import glob
import os

import pandas as pd

# 研究频率 -> pandas 重采样规则
RESAMPLE_RULES = {"1MIN": "1min", "3MIN": "3min", "5MIN": "5min", "15MIN": "15min", "30MIN": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}


def load_m1(data_dir: str, years: list[int] | None = None) -> pd.DataFrame:
    """读取目录下全部（或指定年份）M1 文件，拼接、去重、按时间排序。"""
    files = sorted(glob.glob(os.path.join(data_dir, "DAT_ASCII_XAUUSD_M1_*.csv*")))
    if years is not None:
        files = [f for f in files if any(f"_M1_{y}" in os.path.basename(f) for y in years)]
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 DAT_ASCII_XAUUSD_M1_*.csv")
    frames = []
    for f in files:
        d = pd.read_csv(f, sep=";", header=None,
                        names=["dt", "open", "high", "low", "close", "volume"])
        d.index = pd.to_datetime(d.pop("dt"), format="%Y%m%d %H%M%S")
        frames.append(d)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # 剔除无成交的填充 K 线（周末/休市时段可能出现 volume=0 且 OHLC 不变）
    df = df[df["volume"] > 0]
    return df.astype(float)


def resample_ohlcv(m1: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    M1 -> 目标频率。K 线以左端时间戳标记，区间左闭右开（与 MT4/MT5 一致）。
    日线按纽约 17:00 日切（外汇/贵金属通行惯例，避免周日短线 K 线）。
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if freq == "1D":
        ny = m1.tz_localize("UTC").tz_convert("America/New_York")
        ny.index = ny.index + pd.Timedelta(hours=7)          # 17:00 NY -> 次日 00:00
        out = ny.resample("1D").agg(agg)
        out.index = out.index.tz_localize(None)
    else:
        out = m1.resample(RESAMPLE_RULES[freq], label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])
