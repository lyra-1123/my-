# -*- coding: utf-8 -*-
"""
家族：结合黄金长期上涨趋势的趋势延续（第八批，2026-09-27）。

与在跑的 TT30（30MIN 极端走势延续）区分：本批以 4H/1D 为主、持仓数日，机制分别为锚定、信息连续性、需求承接、波动结构。
日线级特征统一在"已收盘日线"上计算，日内频率通过 core.htf_feature 对齐（无未来函数）。
方向判断以逐年去漂移的 ATR 边际和多空拆分为准（牛市 beta 会美化做多信号）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, htf_feature, params, relative_volume, rolling_mad_zscore, vol_scaled_momentum
from ..registry import register
from .trend_reversion import long_trend


def daily_feature(d: pd.DataFrame, freq: str, fn) -> pd.Series:
    """在已收盘日线上计算 fn(日线 DataFrame)；1D 频率直接计算。"""
    return fn(d) if freq == "1D" else htf_feature(d, freq, "1D", fn)


@register(
    name="HighAnchorMomentum",
    cn_name="52 周高点锚定动量",
    family="trend_continuation",
    hypothesis="锚定效应（George & Hwang 2004）：投资者以一年内的高/低点为参照，价格接近或突破高点时反应不足，突破后继续上行；"
               "黄金长期上涨中频繁创新高，对称地处理接近一年低点的情形",
    formula="H250/L250=已收盘日线 250 日最高/最低；pos=2·(C−L250)/(H250−L250)−1（突破时 >1 或 <−1）；MAD_Z(pos)",
    risks=["位置型信号变化慢，与长期趋势因子高度相关", "历史新高附近波动大，回撤深",
           "牛市中大部分时间 pos>0，做多占比高，需看空头侧与去漂移"],
    freqs=("1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_high_anchor_momentum(df: pd.DataFrame, freq: str = "4H", anchor_days: int = 250, **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    mp = int(anchor_days * 0.8)
    hi = daily_feature(d, freq, lambda b: b["high"].rolling(anchor_days, min_periods=mp).max())
    lo = daily_feature(d, freq, lambda b: b["low"].rolling(anchor_days, min_periods=mp).min())
    pos = 2 * (d["close"] - lo) / (hi - lo + EPS) - 1
    return rolling_mad_zscore(pos.clip(-3, 3), p["norm"])


@register(
    name="ContinuousInfoMomentum",
    cn_name="连续信息动量（温水煮青蛙）",
    family="trend_continuation",
    hypothesis="Da, Gurun & Warachka（2014）：由许多小幅同向变动累积的趋势（信息连续到达）比由少数大跳涨形成的趋势更容易被忽视，"
               "反应不足更严重、延续性更强；用'同向日数占比'衡量趋势的平滑程度",
    formula="M=VSM(已收盘日线,60) 截断 ±3；cont=60 日中与 M 同号的日数占比 − 反号日数占比；MAD_Z( M·max(cont,0) )",
    risks=["60 日窗口在趋势末端反应慢", "平滑上涨在黄金中常对应低波动期，与 QuietTrend 可能相关",
           "日线样本少，统计功效低"],
    freqs=("1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_continuous_info_momentum(df: pd.DataFrame, freq: str = "4H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)

    def fn(b):
        r = np.log(b["close"]).diff()
        M = vol_scaled_momentum(b, (60,)).clip(-3, 3)
        up = (r > 0).astype(float).rolling(60, min_periods=40).mean()
        dn = (r < 0).astype(float).rolling(60, min_periods=40).mean()
        cont = np.where(M > 0, up - dn, dn - up)
        return M * pd.Series(cont, index=b.index).clip(lower=0)
    return rolling_mad_zscore(daily_feature(d, freq, fn), p["norm"])


@register(
    name="DemandAbsorptionTrend",
    cn_name="趋势中的买盘承接",
    family="trend_continuation",
    hypothesis="上涨趋势中日内被砸出长下影、收盘回到高位，说明卖压被实物/机构买盘吸收，需求强，趋势延续；"
               "下跌趋势中对称地看长上影（反弹被卖盘压回）；放量时承接更可信",
    formula="absorb=(min(O,C)−L − (H−max(O,C)))/(H−L)，日线 5 日均值 × (1+max(ln RelVol_日,0))；"
            "只保留 sign(absorb)=sign(T) 一侧；MAD_Z( absorb·min(|T|,3) )",
    risks=["影线对日切时间敏感（Dukascopy UTC 日 vs 纽约 17:00）", "长下影也可能是下跌中继的技术反抽",
           "4H 以下频率只是日线信号的重复"],
    freqs=("4H", "1D"),
    added="2026-09-27",
)
def factor_demand_absorption_trend(df: pd.DataFrame, freq: str = "1D", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)

    def fn(b):
        rng = (b["high"] - b["low"]).where(lambda x: x > EPS)
        lower = b[["open", "close"]].min(axis=1) - b["low"]
        upper = b["high"] - b[["open", "close"]].max(axis=1)
        ab = ((lower - upper) / rng).fillna(0).rolling(5, min_periods=5).mean()
        rv = b["volume"] / (b["volume"].shift(1).rolling(20, min_periods=10).median() + EPS)
        return ab * (1 + np.log(rv.clip(lower=EPS)).clip(lower=0))
    ab = daily_feature(d, freq, fn)
    T = long_trend(d, freq)
    raw = (ab * T.abs().clip(upper=3)).where(np.sign(ab) == np.sign(T), 0.0)
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="QuietTrend",
    cn_name="低波动趋势",
    family="trend_continuation",
    hypothesis="黄金的稳健上涨常伴随波动收敛（持续的被动吸筹），伴随波动放大的趋势更多是恐慌/挤仓，易衰竭；"
               "趋势 × 波动收敛程度越高，延续概率越大",
    formula="RV_n=已收盘日线对数收益的 n 日标准差；vc=−ln(RV_20/RV_60)；MAD_Z( T · (1+clip(vc,−1,1)) )",
    risks=["本质仍是长期趋势因子，与 TSMOM 类相关", "波动收敛后常伴随突破（方向不定），与假设冲突",
           "日线级信号，统计功效低"],
    freqs=("1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_quiet_trend(df: pd.DataFrame, freq: str = "4H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    T = long_trend(d, freq)

    def fn(b):
        r = np.log(b["close"]).diff()
        return -np.log(r.rolling(20, min_periods=15).std() / (r.rolling(60, min_periods=40).std() + EPS))
    vc = daily_feature(d, freq, fn).clip(-1, 1)
    return rolling_mad_zscore(T * (1 + vc), p["norm"])
