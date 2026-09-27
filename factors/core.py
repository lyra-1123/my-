# -*- coding: utf-8 -*-
"""
因子库公共组件：频率参数、输入校验、ATR、相对成交量、滚动 MAD 去极值 + Z-Score。

约定
----
- 输入 DataFrame：DatetimeIndex（UTC，升序）+ open/high/low/close/volume 五列，禁止其他字段。
- 因子值在 t 根 K 线收盘可得，只能使用 <= t 的数据；交易在 t+1 开盘执行。
- 因子方向统一：因子值 > 0 表示"看多"（反转类因子自行取负号），评估时统一期望 IC > 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# 窗口以"根 K 线"为单位；这些是先验起点而非优化结果，禁止在样本内网格搜索后回填。
FREQ_PRESETS = {
    "1D":    dict(chan=20, atr=14, vol_base=20, thrust=5,  sq_short=5,  sq_long=60,  norm=250,  intraday=False, trend=(20, 60, 120)),
    "4H":    dict(chan=30, atr=14, vol_base=20, thrust=6,  sq_short=6,  sq_long=90,  norm=500,  intraday=True,  trend=(30, 90, 180)),
    "1H":    dict(chan=24, atr=24, vol_base=20, thrust=6,  sq_short=8,  sq_long=120, norm=750,  intraday=True,  trend=(24, 72, 240)),
    "30MIN": dict(chan=32, atr=32, vol_base=20, thrust=8,  sq_short=8,  sq_long=160, norm=1000, intraday=True,  trend=(32, 96, 288)),
    "15MIN": dict(chan=32, atr=32, vol_base=20, thrust=8,  sq_short=12, sq_long=192, norm=1500, intraday=True,  trend=(32, 96, 384)),
    "5MIN":  dict(chan=36, atr=48, vol_base=20, thrust=12, sq_short=12, sq_long=288, norm=2000, intraday=True,  trend=(36, 144, 576)),
    # 剥头皮频率（第九批加入）：窗口约为"1 小时"的倍数；norm 约 3~4 个交易日
    "3MIN":  dict(chan=20, atr=40, vol_base=20, thrust=10, sq_short=10, sq_long=200, norm=2500, intraday=True,  trend=(20, 60, 240)),
    "1MIN":  dict(chan=60, atr=60, vol_base=20, thrust=15, sq_short=30, sq_long=600, norm=5000, intraday=True,  trend=(60, 180, 720)),
}
ALL_FREQS = list(FREQ_PRESETS)


def params(freq: str, **overrides) -> dict:
    """取某频率的预设参数，可用关键字覆盖。"""
    return {**FREQ_PRESETS[freq], **overrides}


def check_input(df: pd.DataFrame) -> pd.DataFrame:
    """校验字段：只允许 OHLCV，杜绝使用数据字典之外的字段。"""
    need = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少字段: {miss}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index 必须是 DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index 必须按时间升序")
    return df[need].astype(float)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR。TR 用到前一根收盘价（t-1），t 时刻已知，无未来函数。"""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def relative_volume(df: pd.DataFrame, n_days: int, intraday: bool) -> pd.Series:
    """
    相对成交量 = 当前成交量 / 历史"同类"K 线成交量的中位数（基准不含当前 K 线）。
    日内频率按"同一时刻(time-of-day)"分组，消除亚盘/伦敦/纽约的成交量季节性；日频直接用前 n 根。
    """
    vol = df["volume"]
    minp = max(5, n_days // 4)
    if intraday:
        tod = df.index.hour * 60 + df.index.minute
        base = vol.groupby(tod).transform(lambda s: s.shift(1).rolling(n_days, min_periods=minp).median())
    else:
        base = vol.shift(1).rolling(n_days, min_periods=minp).median()
    return vol / (base + EPS)


def rolling_mad_zscore(x: pd.Series, window: int, n_mad: float = 3.0) -> pd.Series:
    """
    滚动 MAD 去极值 + 滚动 Z-Score（纯时间序列，无未来函数）。
      1) med = 滚动中位数；MAD = 滚动 |x - med| 的中位数（近似 MAD，避免 O(N*W) 内存）
      2) 截断到 med ± n_mad * 1.4826 * MAD（MAD = 0 的稀疏段不截断，避免把信号全部压成中位数）
      3) 截断后序列做滚动均值/标准差标准化
    横截面 MAD 不适用于单品种；全样本 MAD 会引入未来信息。
    """
    minp = max(30, window // 4)
    med = x.rolling(window, min_periods=minp).median()
    mad = (x - med).abs().rolling(window, min_periods=minp).median()
    scale = (1.4826 * mad).where(mad > EPS)
    clipped = x.clip(lower=med - n_mad * scale, upper=med + n_mad * scale)
    mu = clipped.rolling(window, min_periods=minp).mean()
    sd = clipped.rolling(window, min_periods=minp).std()
    z = (clipped - mu) / (sd + EPS)
    return z.where(sd > EPS, 0.0)


# ---------------------------------------------------------------------------
# 多周期（MTF）工具：只使用"已收盘"的大周期 K 线，严格无未来函数
# ---------------------------------------------------------------------------
BAR_DURATION = {"1MIN": "1min", "3MIN": "3min", "5MIN": "5min", "15MIN": "15min", "30MIN": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}

# 每个基准频率对应的两个更高周期（固定时长，便于计算"大周期 K 线何时收盘"）
HTF_MAP = {
    "5MIN": ("30min", "4h"),
    "15MIN": ("1h", "4h"),
    "30MIN": ("4h", "1D"),
    "1H": ("4h", "1D"),
    "4H": ("1D", "7D"),
    "1D": ("7D", "28D"),
}


def htf_feature(d: pd.DataFrame, freq: str, rule: str, feature_fn) -> pd.Series:
    """
    在更高周期 rule 上计算特征，并对齐回基准频率 freq。

    对齐规则（防未来函数的关键）：
      - 大周期 K 线 [s, s+rule) 在 s+rule 时刻才收盘，其特征只能在 s+rule 之后使用；
      - 基准 K 线 [t, t+bar) 在 t+bar 收盘时决策，只能取 "大周期收盘时间 <= t+bar" 的最新值；
      - 正在形成中的大周期 K 线（未收盘）永远不被使用。
    feature_fn: 输入大周期 OHLCV DataFrame，返回同索引的 Series。
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    htf = d.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    f = feature_fn(htf)
    f_avail = pd.DataFrame({"t": f.index + pd.Timedelta(rule), "v": f.to_numpy()}).dropna()
    base_end = pd.DataFrame({"t": d.index + pd.Timedelta(BAR_DURATION[freq])})
    merged = pd.merge_asof(base_end, f_avail, on="t", direction="backward")
    return pd.Series(merged["v"].to_numpy(), index=d.index)


def vol_scaled_momentum(bars: pd.DataFrame, lookbacks=(5, 20)) -> pd.Series:
    """波动率缩放动量（近似 t 统计量）的多窗口平均：log(C/C_L) / (σ·√L)。"""
    lc = np.log(bars["close"])
    L_max = max(lookbacks)
    sigma = lc.diff().rolling(L_max, min_periods=L_max // 2).std()
    parts = [(lc - lc.shift(L)) / (sigma * np.sqrt(L) + EPS) for L in lookbacks]
    return pd.concat(parts, axis=1).mean(axis=1)


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    交易日标签（纽约 17:00 为界，与换日/过夜费一致）：周日晚开盘的 K 线归入周一。
    所有"按日汇总"的地方都必须用它，不能用 UTC 日期再过滤周末（会丢掉周日晚上的盈亏）。
    """
    ny = index.tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return ny.normalize().tz_localize(None)
