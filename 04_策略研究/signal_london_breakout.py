"""
signal_london_breakout.py
============================
假设8：伦敦开盘突破亚洲盘区间（00_方案/hypothesis_london_breakout.md）。时间全部为伦敦当地时间
（随夏令时），在 M15 上识别：

  亚洲盘区间  伦敦 00:00-08:00 的最高/最低（该时段不足 MIN_RANGE_BARS 根则当天不交易）
  入场        伦敦 08:00-12:00 内，第一根收盘 > 区间高点的 M15 做多 / < 区间低点做空；每天最多一笔
  止损        区间另一侧（做多止损在区间低点，做空在区间高点）
  强制平仓    当天交易日结束前最后一根（纽约 17:00 之前）收盘
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.exits import ExitRule  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from goldq.sessions import london_time, ny_time  # noqa: E402
from compute_indicators import compute_atr  # noqa: E402

TIMEFRAME = "15min"
RANGE_START_H, RANGE_END_H = 0, 8
ENTRY_END_H = 12
MIN_RANGE_BARS = 24  # 8 小时 = 32 根 M15，少于 24 根视为假期/数据缺失
EXIT_RULES = [
    ExitRule("stop_dayend", "given", "none", max_bars=None),
    ExitRule("stop_2R_dayend", "given", "fixed_r", max_bars=None, r_multiple=2.0),
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = compute_atr(df, 14, max_gap_minutes=180)  # 只给回测引擎做有效性检查用
    ldn = london_time(df["time_utc"])
    ny = ny_time(df["time_utc"])
    df["ldn_date"] = ldn.dt.tz_localize(None).dt.normalize()
    ldn_hour = ldn.dt.hour.to_numpy()
    before_ny_close = (ny.dt.hour < 17).to_numpy()

    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    n = len(df)
    signal = np.zeros(n, dtype=int)
    stop_dist = np.full(n, np.nan)
    last_idx = np.full(n, -1, dtype=np.int64)
    rng_hi = np.full(n, np.nan)
    rng_lo = np.full(n, np.nan)

    for _, idx in df.groupby("ldn_date").indices.items():
        idx = np.sort(idx)
        h = ldn_hour[idx]
        in_range = idx[(h >= RANGE_START_H) & (h < RANGE_END_H)]
        if len(in_range) < MIN_RANGE_BARS:
            continue
        hi, lo = high[in_range].max(), low[in_range].min()
        day_end = idx[before_ny_close[idx]]
        if len(day_end) == 0:
            continue
        end = day_end.max()
        for t in idx[(h >= RANGE_END_H) & (h < ENTRY_END_H)]:
            if close[t] > hi:
                signal[t], stop_dist[t] = 1, close[t] - lo
            elif close[t] < lo:
                signal[t], stop_dist[t] = -1, hi - close[t]
            else:
                continue
            last_idx[t], rng_hi[t], rng_lo[t] = end, hi, lo
            break

    df["signal"] = signal
    df["stop_dist"] = stop_dist
    df["last_idx"] = last_idx
    df["range_high"] = rng_hi
    df["range_low"] = rng_lo
    return df


def load_m15(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    return resample_ohlcv(fetch_bars(symbol=symbol, start=start, end=end), rule=TIMEFRAME)


def describe_trades(feat: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    rows = feat[feat["signal"] != 0].tail(n_examples)
    return pd.DataFrame({
        "伦敦日期": rows["ldn_date"].dt.date.to_numpy(),
        "亚洲盘高": rows["range_high"].round(2).to_numpy(),
        "亚洲盘低": rows["range_low"].round(2).to_numpy(),
        "方向": np.where(rows["signal"] > 0, "做多", "做空"),
        "入场(伦敦时间)": london_time(rows["time_utc"]).dt.strftime("%H:%M").to_numpy(),
        "入场价": rows["close"].round(2).to_numpy(),
        "强制平仓(纽约时间)": ny_time(feat["time_utc"].iloc[rows["last_idx"].to_numpy()]).dt.strftime("%H:%M").to_numpy(),
    })
