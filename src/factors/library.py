"""Candidate factors for XAUUSD regime classification.

For a martingale strategy the relevant question is not "which direction will
price go" but "is the market currently choppy/mean-reverting (safe to average
down into) or trending (a martingale ladder can run away and blow up)". Every
factor here is computed from information available at time t only (no
look-ahead); labels in `labels.py` look forward and are for evaluation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return _true_range(df).rolling(n).mean()


def realized_vol(df: pd.DataFrame, n: int) -> pd.Series:
    log_ret = np.log(df["close"]).diff()
    return log_ret.rolling(n).std()


def bollinger_width(df: pd.DataFrame, n: int, k: float = 2.0) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return (2 * k * sd) / ma


def rsi(df: pd.DataFrame, n: int) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(n).mean()


def ma_slope(df: pd.DataFrame, n: int) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    return (ma - ma.shift(n)) / (n * df["close"])


def zscore_vs_ma(df: pd.DataFrame, n: int) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return (df["close"] - ma) / sd.replace(0, np.nan)


def efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    """Kaufman's Efficiency Ratio over the trailing n bars: net move / path
    length. Near 1 = trending, near 0 = choppy. Usable both as a backward
    factor and (shifted forward) as the regime label — see labels.py."""
    net = (df["close"] - df["close"].shift(n)).abs()
    path = df["close"].diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def session_bucket(df: pd.DataFrame) -> pd.Series:
    """Coarse UTC trading-session label. NOT independently verified against a
    live UTC feed (see reports/00_progress.md) — treat session-based findings
    as provisional until that's checked."""
    hour = df["time"].dt.hour
    bins = [-1, 7, 13, 16, 21, 24]
    labels = ["asia", "london_open", "ny_overlap", "ny_only", "late"]
    return pd.cut(hour, bins=bins, labels=labels).astype(str)


def day_of_week(df: pd.DataFrame) -> pd.Series:
    return df["time"].dt.dayofweek


def build_factor_table(df: pd.DataFrame) -> pd.DataFrame:
    """df: OHLCV with a `time` column (e.g. H1 bars). Returns a DataFrame of
    factor columns aligned to df's index, all backward-looking only."""
    out = pd.DataFrame(index=df.index)
    for n in (14, 24, 48):
        out[f"atr_{n}"] = atr(df, n)
        out[f"realized_vol_{n}"] = realized_vol(df, n)
        out[f"bb_width_{n}"] = bollinger_width(df, n)
        out[f"rsi_{n}"] = rsi(df, n)
        out[f"adx_{n}"] = adx(df, n)
        out[f"ma_slope_{n}"] = ma_slope(df, n)
        out[f"zscore_vs_ma_{n}"] = zscore_vs_ma(df, n)
        out[f"efficiency_ratio_{n}"] = efficiency_ratio(df, n)
    out["hour"] = df["time"].dt.hour
    out["session"] = session_bucket(df)
    out["day_of_week"] = day_of_week(df)
    out["time"] = df["time"]
    return out
