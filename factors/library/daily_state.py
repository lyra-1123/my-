# -*- coding: utf-8 -*-
"""
家族：日线级状态 × 日内执行（第十五批，2026-09-29）。

填补 TT30（30MIN 价格路径，约 16 小时）与 HA1H（一年区间位置）之间"数日~数月"的时间尺度。
全部只用已收盘日线（htf_feature，按收盘时间 merge_asof），在 1H 上执行统一规则（每日平仓、次日重新确认）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, check_input, params, rolling_mad_zscore
from ..registry import register
from .trend_continuation import daily_feature

FREQS = ("1H",)


def _range_pos(df: pd.DataFrame, freq: str, days: int, **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    hi = daily_feature(d, freq, lambda b: b["high"].rolling(days, min_periods=days).max())
    lo = daily_feature(d, freq, lambda b: b["low"].rolling(days, min_periods=days).min())
    pos = 2 * (d["close"] - lo) / (hi - lo + EPS) - 1
    return rolling_mad_zscore(pos.clip(-3, 3), p["norm"])


@register(
    name="DailyRangePos5",
    cn_name="周度高低位突破",
    family="daily_state",
    hypothesis="最近一周（5 个交易日）的高低点是短线交易者的参照与止损堆积处；价格处在周区间极端或突破时，止损与追单推动延续",
    formula="H5/L5=最近 5 根已收盘日线最高/最低；pos=2·(C−L5)/(H5−L5)−1，截断 ±3；MAD_Z(pos)",
    risks=["与 TT30 的日内趋势可能重叠（周区间突破常伴随日内强势）", "区间窄时 pos 放大，噪声大"],
    freqs=FREQS,
    added="2026-09-29",
)
def factor_daily_range_pos5(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    return _range_pos(df, freq, 5, **kw)


@register(
    name="DailyRangePos21",
    cn_name="月度高低位突破",
    family="daily_state",
    hypothesis="最近一个月（21 个交易日）的高低点锚定；介于周度与 HA1H 的一年锚点之间",
    formula="H21/L21=最近 21 根已收盘日线最高/最低；pos=2·(C−L21)/(H21−L21)−1，截断 ±3；MAD_Z(pos)",
    risks=["与 HA1H（250 日）同类，可能相关较高"],
    freqs=FREQS,
    added="2026-09-29",
)
def factor_daily_range_pos21(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    return _range_pos(df, freq, 21, **kw)


@register(
    name="DailyMomentum5",
    cn_name="五日动量",
    family="daily_state",
    hypothesis="数小时~数日尺度的极端走势倾向延续（lessons L22）；用 5 日波动率缩放收益衡量这一尺度的趋势强度",
    formula="M=ln(C/已收盘日线 5 日前收盘)/(20 日日收益标准差·√5)；MAD_Z(clip(M,±4))",
    risks=["5 日尺度与日内反转、周度均值回归相互抵消", "事件日跳空会放大 M"],
    freqs=FREQS,
    added="2026-09-29",
)
def factor_daily_momentum5(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    c5 = daily_feature(d, freq, lambda b: b["close"].shift(4))           # 已收盘日线中，含最新一根的第 5 根前收盘
    sig = daily_feature(d, freq, lambda b: np.log(b["close"]).diff().rolling(20, min_periods=15).std())
    m = np.log(d["close"] / c5) / (sig * np.sqrt(5) + EPS)
    return rolling_mad_zscore(m.clip(-4, 4), p["norm"])


@register(
    name="MA200Distance",
    cn_name="200 日均线距离",
    family="daily_state",
    hypothesis="长期趋势位置：价格在 200 日均线之上且距离越远，长期趋势越强（对照：预期与 HA1H 高相关）",
    formula="(C − 已收盘日线 SMA200) / 已收盘日线 ATR(20)；MAD_Z",
    risks=["与 HA1H 同为长期趋势状态，冗余风险高", "牛市中大部分时间为正，需看空头侧"],
    freqs=FREQS,
    added="2026-09-29",
)
def factor_ma200_distance(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)

    def atr_d(b):
        pc = b["close"].shift(1)
        tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / 20, adjust=False, min_periods=20).mean()
    ma = daily_feature(d, freq, lambda b: b["close"].rolling(200, min_periods=180).mean())
    a = daily_feature(d, freq, atr_d)
    return rolling_mad_zscore((d["close"] - ma) / (a + EPS), p["norm"])
