# -*- coding: utf-8 -*-
"""从 MT5 取 K 线并换算为 UTC（与研究/模拟盘相同的 DataFrame 格式：UTC 无时区索引 + OHLCV，volume=tick_volume）。"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from live import config as C
from live.mt5_api import mt5

TF = {"30MIN": "TIMEFRAME_M30", "1H": "TIMEFRAME_H1", "1D": "TIMEFRAME_D1", "1MIN": "TIMEFRAME_M1"}
DUR = {"30MIN": pd.Timedelta("30min"), "1H": pd.Timedelta("1h"), "1D": pd.Timedelta("1D"), "1MIN": pd.Timedelta("1min")}


def connect():
    ok = mt5.initialize(path=C.MT5_PATH) if C.MT5_PATH else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 初始化失败：{mt5.last_error()}")
    if not mt5.symbol_select(C.SYMBOL, True):
        raise RuntimeError(f"无法选中品种 {C.SYMBOL}：{mt5.last_error()}")


def server_to_utc(ts_server: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """服务器时间（无时区）→ UTC（无时区）。"""
    idx = pd.DatetimeIndex(ts_server)
    mode = C.SERVER_TZ
    if mode == "ny+7":
        ny = (idx - pd.Timedelta(hours=7)).tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        return ny.tz_convert("UTC").tz_localize(None)
    if mode.startswith("fixed:"):
        return idx - pd.Timedelta(hours=float(mode.split(":")[1]))
    raise ValueError(f"未知 SERVER_TZ：{mode}")


def utc_now_from_server() -> pd.Timestamp:
    """用最新报价时间（服务器时间）换算当前 UTC；休市时退回本机 UTC。"""
    tick = mt5.symbol_info_tick(C.SYMBOL)
    local = pd.Timestamp(time.time(), unit="s")
    if tick is None or tick.time <= 0:
        return local
    t = server_to_utc(pd.DatetimeIndex([pd.Timestamp(tick.time, unit="s")]))[0]
    return max(t, local - pd.Timedelta(minutes=5)) if pd.notna(t) else local


def fetch_bars(freq: str, count: int, now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """取最近 count 根 K 线，换算为 UTC，并丢弃尚未走完的最后一根。"""
    rates = mt5.copy_rates_from_pos(C.SYMBOL, getattr(mt5, TF[freq]), 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"取 {freq} K 线失败：{mt5.last_error()}")
    df = pd.DataFrame(rates)
    idx = server_to_utc(pd.to_datetime(df["time"], unit="s"))
    out = pd.DataFrame({"open": df["open"].to_numpy(), "high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                        "close": df["close"].to_numpy(), "volume": df["tick_volume"].to_numpy().astype(float)}, index=idx)
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["volume"] > 0]
    now_utc = now_utc or utc_now_from_server()
    return out[out.index + DUR[freq] <= now_utc]
