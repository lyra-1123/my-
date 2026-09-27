# -*- coding: utf-8 -*-
"""
家族：结合黄金长期上涨趋势的均值回归（第七批，2026-09-27）。

背景：黄金 2009-2026 年长期上涨（样本外买入持有 1 盎司 ≈ +2770$），"逢跌买入"天然会被牛市 beta 美化。
因此本批因子的方向判断以"逐年去漂移后的 ATR 边际"和多空拆分为准，而不是美元净利。
长期趋势 T 统一定义为：已收盘日线上的波动率缩放动量（回看 60、120 日），截断到 ±3（core.htf_feature，无未来函数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, htf_feature, params, rolling_mad_zscore, vol_scaled_momentum
from ..registry import register

BARS_PER_DAY = {"5MIN": 276, "15MIN": 92, "30MIN": 46, "1H": 23, "4H": 6, "1D": 1}   # 黄金约 23 小时/天


def long_trend(d: pd.DataFrame, freq: str) -> pd.Series:
    """长期趋势（日线，60/120 日），已对齐到基准频率，截断 ±3。"""
    if freq == "1D":
        return vol_scaled_momentum(d, (60, 120)).clip(-3, 3)
    return htf_feature(d, freq, "1D", lambda b: vol_scaled_momentum(b, (60, 120))).clip(-3, 3)


@register(
    name="UptrendDipReversion",
    cn_name="长期趋势中的多日超跌回归",
    family="trend_reversion",
    hypothesis="黄金有大量价格敏感的实物/官方买家（央行、印度与中国的首饰和投资需求、ETF 逢低申购），"
               "长期上升趋势中的数日级深度回调多为流动性冲击（美元急涨、保证金追缴、CTA 减仓）而非基本面改变，价格回到趋势；"
               "下跌趋势中对称地卖出反弹",
    formula="T=VSM(已收盘日线,(60,120))；上涨时 depth=(TS_MAX(H,3日)−C)/ATR，下跌时 depth=(C−TS_MIN(L,3日))/ATR；"
            "MAD_Z( sign(T)·min(|T|,3)·max(depth,0) )",
    risks=["趋势拐点处的'回调'实为反转（2013 年金价崩跌）", "与牛市 beta 高度混杂，必须看去漂移与空头侧",
           "持仓可能跨夜；日内频率受换日前平仓约束"],
    freqs=("30MIN", "1H", "4H"),
    added="2026-09-27",
)
def factor_uptrend_dip_reversion(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    T = long_trend(d, freq)
    s = np.sign(T)
    k = 3 * BARS_PER_DAY[freq]
    a = atr(d, p["atr"])
    dd_up = (d["high"].rolling(k, min_periods=k).max() - d["close"]) / (a + EPS)      # 距 3 日高点的回撤
    dd_dn = (d["close"] - d["low"].rolling(k, min_periods=k).min()) / (a + EPS)       # 距 3 日低点的反弹
    depth = pd.Series(np.where(s > 0, dd_up, np.where(s < 0, dd_dn, 0.0)), index=d.index).clip(lower=0)
    raw = s * T.abs().clip(upper=3) * depth
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="RSI2PullbackInTrend",
    cn_name="趋势中的 RSI(2) 超卖/超买",
    family="trend_reversion",
    hypothesis="短周期（2 根 K 线）的极端涨跌多为过度反应；在长期趋势方向上买超卖、卖超买（Connors 类），"
               "利用'趋势 + 短期过度反应'的非对称性；逆趋势一侧不交易",
    formula="RSI2=Wilder RSI(2)；dev=(50−RSI2)/50；只保留 sign(dev)=sign(T) 的一侧；MAD_Z( dev·min(|T|,3) )",
    risks=["RSI(2) 在单边急跌中可以连续多根保持超卖，逆势接刀", "信号频繁，1D 以下成本占比高",
           "与 UptrendDipReversion 同属'顺大逆小'，可能高度相关"],
    freqs=("1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_rsi2_pullback_in_trend(df: pd.DataFrame, freq: str = "4H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    T = long_trend(d, freq)
    ch = d["close"].diff()
    up = ch.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    rsi2 = 100 - 100 / (1 + up / (dn + EPS))
    dev = (50 - rsi2) / 50                                              # >0 超卖（看多）
    aligned = np.sign(dev) == np.sign(T)
    raw = (dev * T.abs().clip(upper=3)).where(aligned, 0.0)
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="WeekendGapReversion",
    cn_name="周末跳空回补",
    family="trend_reversion",
    hypothesis="周末休市期间信息堆积，周日开盘的跳空常常过度反应（注意力驱动）；流动性恢复后套利与做市资金回补跳空",
    formula="gap=(本周首根开盘−上周末收盘)/ATR；按周序列做 MAD_Z（104 周）；本周开盘后 24 小时内因子=−gap_z，其余为 0",
    risks=["每周最多一个信号，样本少（约 900 周）", "重大周末事件（地缘冲突）的跳空可能是新信息而不回补",
           "开盘后第一小时受执行层'换日后 1 小时不开仓'约束，入场偏晚"],
    freqs=("15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_weekend_gap_reversion(df: pd.DataFrame, freq: str = "30MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    a = atr(d, p["atr"])
    gap_hours = (d.index.to_series().diff() > pd.Timedelta(hours=24))           # 周末休市后的第一根
    first = d.index[gap_hours.to_numpy()]
    prev_close = d["close"].shift(1)
    g = ((d["open"] - prev_close) / (a.shift(1) + EPS))[gap_hours.to_numpy()]
    gz = rolling_mad_zscore(g, 104)
    out = pd.Series(0.0, index=d.index)
    for t0, v in gz.items():
        if np.isnan(v):
            continue
        out.loc[t0:t0 + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)] = -v
    return out


ASIA = (23 * 60, 7 * 60)        # 亚盘 [23:00, 07:00) UTC（跨日）
LONDON_HOLD = (7 * 60, 12 * 60)  # 伦敦持有窗口 [07:00, 12:00) UTC


@register(
    name="AsiaSessionReversion",
    cn_name="亚盘偏离的伦敦回归",
    family="trend_reversion",
    hypothesis="亚盘流动性薄，价格容易被少量订单推离；伦敦开盘后主要定价资金进场，把偏离拉回；"
               "与长期趋势同向的回归（上涨趋势中亚盘下跌 → 伦敦做多）权重更高",
    formula="asia=log(C_07:00/O_23:00)/STD_60d；raw_day=−asia×(1.5 若 −asia 与 T 同向，否则 0.5)；"
            "日级 MAD_Z(250)，广播到伦敦 [07:00,12:00) UTC，其余为 0",
    risks=["固定 UTC 时段未处理夏令时", "亚盘的方向可能反映亚洲实物需求（信息）而非噪声，此时不回归",
           "每天最多一个信号"],
    freqs=("15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_asia_session_reversion(df: pd.DataFrame, freq: str = "30MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    tod = d.index.hour * 60 + d.index.minute
    # 亚盘归属到"结束日"：23:00 之后的 K 线归入次日
    sess_day = (d.index + pd.Timedelta(hours=1)).normalize()
    in_asia = (tod >= ASIA[0]) | (tod < ASIA[1])
    asia = d[in_asia].groupby(sess_day[in_asia]).agg(o=("open", "first"), c=("close", "last"))
    r = np.log(asia["c"] / asia["o"])
    z = r / (r.shift(1).rolling(60, min_periods=20).std() + EPS)
    T = long_trend(d, freq)
    T_day = T.groupby(sess_day).last().shift(1).reindex(z.index)                    # 前一日收盘时的趋势
    w = np.where(np.sign(-z) == np.sign(T_day), 1.5, 0.5)
    zd = rolling_mad_zscore(-z * w, 250)
    in_hold = (tod >= LONDON_HOLD[0]) & (tod < LONDON_HOLD[1])
    out = pd.Series(0.0, index=d.index)
    out[in_hold] = zd.reindex(sess_day[in_hold]).to_numpy()
    return out.fillna(0.0)
