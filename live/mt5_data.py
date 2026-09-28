# -*- coding: utf-8 -*-
"""从 MT5 取 K 线并换算为 UTC（与研究/模拟盘相同的 DataFrame 格式：UTC 无时区索引 + OHLCV，volume=tick_volume）。"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from live import config as C
from live.mt5_api import mt5

TF = {"30MIN": "TIMEFRAME_M30", "1H": "TIMEFRAME_H1", "1D": "TIMEFRAME_D1", "5MIN": "TIMEFRAME_M5", "1MIN": "TIMEFRAME_M1"}
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


def fetch_bars(freq: str, count: int, now_utc: pd.Timestamp | None = None, contiguous: bool = True,
               include_partial: bool = False) -> pd.DataFrame:
    """取最近 count 根 K 线，换算为 UTC，并丢弃尚未走完的最后一根。"""
    # 请求数量超过终端"图表最大K线数"或服务器历史时，MT5 会直接返回 None；逐步减半重试
    rates, n = None, count
    while n >= 500:
        rates = mt5.copy_rates_from_pos(C.SYMBOL, getattr(mt5, TF[freq]), 0, n)
        if rates is not None and len(rates) > 0:
            break
        n //= 2
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"取 {freq} K 线失败：{mt5.last_error()}")
    df = pd.DataFrame(rates)
    idx = server_to_utc(pd.to_datetime(df["time"], unit="s"))
    out = pd.DataFrame({"open": df["open"].to_numpy(), "high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                        "close": df["close"].to_numpy(), "volume": df["tick_volume"].to_numpy().astype(float)}, index=idx)
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["volume"] > 0]
    if contiguous:
        out = latest_contiguous(out)
    if include_partial:   # 含尚未走完的最后一根（仅用于盘中预估，交易决策从不使用）
        return out
    now_utc = now_utc or utc_now_from_server()
    return out[out.index + DUR[freq] <= now_utc]


def latest_contiguous(bars: pd.DataFrame, max_gap_days: float = C.MAX_GAP_DAYS) -> pd.DataFrame:
    """只保留最后一个缺口（> max_gap_days 天）之后的连续数据。经纪商历史常有大段缺失，跨缺口计算滚动指标会失真。"""
    gaps = bars.index.to_series().diff() > pd.Timedelta(days=max_gap_days)
    if gaps.any():
        bars = bars[bars.index >= gaps[gaps].index[-1]]
    return bars


def utc_to_server(ts_utc: pd.Timestamp) -> pd.Timestamp:
    """UTC（无时区）→ 服务器时间（无时区）。"""
    mode = C.SERVER_TZ
    if mode == "ny+7":
        return pd.Timestamp(ts_utc).tz_localize("UTC").tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    if mode.startswith("fixed:"):
        return pd.Timestamp(ts_utc) + pd.Timedelta(hours=float(mode.split(":")[1]))
    raise ValueError(f"未知 SERVER_TZ：{mode}")


def _srv_arg(ts_server: pd.Timestamp):
    # MetaTrader5 包把 datetime 参数当作"秒数"使用；传带 UTC 时区的对象，避免被本机时区再换算一次
    return pd.Timestamp(ts_server).tz_localize("UTC").to_pydatetime()


def deals_between(utc_a: pd.Timestamp, utc_b: pd.Timestamp) -> pd.DataFrame:
    """[utc_a, utc_b) 内本品种的全部成交，时间换算为 UTC。查询窗口两边各放宽 2 天再精确过滤，不依赖终端对时间参数的解释。"""
    a, b = utc_to_server(utc_a) - pd.Timedelta(days=2), utc_to_server(utc_b) + pd.Timedelta(days=2)
    ds = mt5.history_deals_get(_srv_arg(a), _srv_arg(b)) or ()
    cols = ["ticket", "order", "time", "time_msc", "type", "entry", "magic", "position_id", "reason", "volume", "price",
            "commission", "swap", "profit", "fee", "symbol", "comment"]
    df = pd.DataFrame([{k: getattr(d, k, 0) for k in cols} for d in ds], columns=cols)
    df = df[df["symbol"] == C.SYMBOL].copy()
    if df.empty:
        df["utc"] = pd.Series(dtype="datetime64[ns]")
        return df
    df["utc"] = server_to_utc(pd.to_datetime(df["time_msc"], unit="ms"))
    return df[(df["utc"] >= utc_a) & (df["utc"] < utc_b)].sort_values("time_msc").reset_index(drop=True)


def bars_range(freq: str, utc_a: pd.Timestamp, utc_b: pd.Timestamp) -> pd.DataFrame:
    """[utc_a, utc_b) 内已走完的 K 线（用于计算单笔交易的最大浮盈/浮亏）。"""
    a, b = utc_to_server(utc_a) - pd.Timedelta(hours=1), utc_to_server(utc_b) + pd.Timedelta(hours=1)
    rates = mt5.copy_rates_range(C.SYMBOL, getattr(mt5, TF[freq]), _srv_arg(a), _srv_arg(b))
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rates)
    idx = server_to_utc(pd.to_datetime(df["time"], unit="s"))
    out = pd.DataFrame({"open": df["open"].to_numpy(), "high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                        "close": df["close"].to_numpy()}, index=idx)
    return out[(out.index >= utc_a) & (out.index < utc_b)]
