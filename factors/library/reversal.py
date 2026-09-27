# -*- coding: utf-8 -*-
"""
家族：放量衰竭反转（第二批，2026-09-27）。
动机：第一批中 VolWeightedCloseThrust 在 5/15MIN 上 2009-2026 每一年 IC 都为负，
说明日内"放量强势"之后倾向回吐。这里把它显式建模为反转因子（因子 > 0 仍表示看多）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore
from ..registry import register


@register(
    name="VolumeClimaxReversal",
    cn_name="放量衰竭反转",
    family="reversal",
    hypothesis="日内极端放量 + 振幅扩张往往是止损盘/追单集中成交的'高潮'（流动性需求冲击），"
               "做市商提供流动性要求补偿，冲击过后价格部分回吐（Grossman-Miller 流动性补偿）",
    formula="MAD_Z( -(C - C.shift(k))/ATR.shift(1) * TS_MAX(max(log RelVol,0)*clip(RX-1,0,3), k) )，k=3",
    risks=["单笔期望小，点差/ATR 高的频率上被成本吞噬", "趋势日（数据公布后单边）逆势连续亏损",
           "alpha 衰减：2020 年后反转强度明显下降"],
    freqs=("5MIN", "15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_volume_climax_reversal(df: pd.DataFrame, freq: str = "15MIN", k: int = 3, **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    a_prev = atr(d, p["atr"]).shift(1)
    move = (d["close"] - d["close"].shift(k)) / (a_prev + EPS)
    rv = relative_volume(d, p["vol_base"], p["intraday"])
    rx = (d["high"] - d["low"]) / (a_prev + EPS)
    climax = np.log(rv.clip(lower=EPS)).clip(lower=0) * (rx - 1.0).clip(0, 3)
    raw = -move * climax.rolling(k, min_periods=k).max()
    return rolling_mad_zscore(raw, p["norm"])
