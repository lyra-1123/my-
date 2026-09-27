# -*- coding: utf-8 -*-
"""
家族：日内趋势 / 震荡状态下的剥头皮（第九批，2026-09-27，频率 1MIN / 3MIN / 5MIN）。

成本前提：点差 0.2$ 固定。1MIN 的点差/ATR 在 2015 年约 0.62、2026 年约 0.09；历史美元回测必然大幅亏损，
判断以"按当前成本折算的 ATR 边际"（evaluate.atr_edge，近 3 年 vs 当前成本）为准。
交易日锚点：纽约 17:00（core.trading_day），与换日、过夜费一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore, trading_day
from ..registry import register

BPH = {"1MIN": 60, "3MIN": 20, "5MIN": 12}            # 每小时 K 线数
SCALP_FREQS = ("1MIN", "3MIN", "5MIN")


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """有方向的效率比：净位移 / 路径长度 ∈ [−1, 1]。"""
    return (close - close.shift(n)) / (close.diff().abs().rolling(n, min_periods=n).sum() + EPS)


def trend_weight(er: pd.Series) -> pd.Series:
    """|ER| ≤ 0.2 → 0（震荡），≥ 0.5 → 1（趋势），中间线性。"""
    return ((er.abs() - 0.2) / 0.3).clip(0, 1)


def anchored_vwap(d: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """当日（纽约 17:00 起）累计 VWAP、成交量加权标准差、当日已走 K 线数。"""
    td = trading_day(d.index)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    v = d["volume"]
    g = pd.Series(td, index=d.index)
    cv = v.groupby(g).cumsum()
    vwap = (tp * v).groupby(g).cumsum() / (cv + EPS)
    var = (tp * tp * v).groupby(g).cumsum() / (cv + EPS) - vwap ** 2
    n = v.groupby(g).cumcount()
    return vwap, np.sqrt(var.clip(lower=EPS)), n


@register(
    name="IntradayRegimeScalp",
    cn_name="日内状态切换剥头皮",
    family="scalp",
    hypothesis="用 1 小时效率比判断日内状态：趋势时顺势在 5 分钟回调处入场（短期反转 + 日内动量），"
               "震荡时在 1 小时区间边缘反向；短期反转只有放在正确的状态里才有意义",
    formula="ER=效率比(1h)；w=clip((|ER|−0.2)/0.3,0,1)；pull=sign(ER)·max(−sign(ER)·(C−C₋₅ₘ)/ATR,0)；"
            "rng=−(2·(C−L₁ₕ)/(H₁ₕ−L₁ₕ)−1)；MAD_Z( w·pull + (1−w)·rng )",
    risks=["单笔幅度小，点差占比高（历史上 1MIN 点差达 0.3~0.9 ATR）", "状态判断滞后：趋势刚起时被当成震荡反向做",
           "1 分钟成交价假设为下一根开盘价，实盘滑点/延迟影响大"],
    freqs=SCALP_FREQS,
    added="2026-09-27",
)
def factor_intraday_regime_scalp(df: pd.DataFrame, freq: str = "1MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    n, k = BPH[freq], max(1, BPH[freq] // 12)
    er = efficiency_ratio(d["close"], n)
    w = trend_weight(er)
    a = atr(d, p["atr"])
    s = np.sign(er)
    pull = s * (-s * (d["close"] - d["close"].shift(k)) / (a + EPS)).clip(lower=0)
    hi = d["high"].rolling(n, min_periods=n).max()
    lo = d["low"].rolling(n, min_periods=n).min()
    rng = -(2 * (d["close"] - lo) / (hi - lo + EPS) - 1)
    return rolling_mad_zscore(w * pull + (1 - w) * rng, p["norm"])


SESSIONS = (("Europe/London", 8 * 60), ("America/New_York", 8 * 60 + 20))   # 伦敦 08:00、纽约 COMEX 08:20（当地时间）
OR_MIN, VALID_MIN = 30, 180


@register(
    name="OpeningRangeBreakout",
    cn_name="开盘区间突破",
    family="scalp",
    hypothesis="伦敦与纽约开盘后 30 分钟的区间汇集了隔夜订单与新信息；放量突破该区间说明当日主导方向确立，"
               "交易员与执行算法跟随（日内动量）",
    formula="OR=开盘后 30 分钟高/低（当地时间，自动处理夏令时）；开盘后 30~180 分钟内："
            "brk=(C−ORH)/ORW 或 (C−ORL)/ORW；× (1+max(ln RelVol,0))；其余时间为 0；只在非零观测上做 MAD_Z（2000 个观测）",
    risks=["假突破（突破后迅速回到区间）", "数据公布（13:30 UTC）常在纽约开盘区间内外剧烈来回",
           "每个时段每天最多一次有效突破方向，样本受限"],
    freqs=SCALP_FREQS,
    added="2026-09-27",
)
def factor_opening_range_breakout(df: pd.DataFrame, freq: str = "5MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    rv = relative_volume(d, p["vol_base"], True)
    out = pd.Series(0.0, index=d.index)
    for tz, open_min in SESSIONS:
        loc = d.index.tz_localize("UTC").tz_convert(tz)
        m = (loc.hour * 60 + loc.minute) - open_min
        day = pd.Series(loc.tz_localize(None).normalize(), index=d.index)
        wk = np.asarray(loc.dayofweek < 5)
        in_or = (m >= 0) & (m < OR_MIN) & wk
        orh = d["high"][in_or].groupby(day[in_or]).max()
        orl = d["low"][in_or].groupby(day[in_or]).min()
        valid = (m >= OR_MIN) & (m < VALID_MIN) & wk
        H = day[valid].map(orh); L = day[valid].map(orl)
        C = d["close"][valid]
        W = (H - L).clip(lower=EPS)
        brk = np.where(C > H, (C - H) / W, np.where(C < L, (C - L) / W, 0.0))
        out[valid] = np.nan_to_num(brk * (1 + np.log(rv[valid].clip(lower=EPS)).clip(lower=0)))
    # 稀疏信号：只在非零观测上做滚动标准化（按观测个数计窗口），其余时间保持 0，避免 0 被平移成非零值
    nz = out[out != 0]
    z = pd.Series(0.0, index=d.index)
    z[nz.index] = rolling_mad_zscore(nz, 2000).to_numpy()
    return z.fillna(0.0)


@register(
    name="AnchoredVWAPBandReversion",
    cn_name="日内 VWAP 带回归",
    family="scalp",
    hypothesis="当日 VWAP 是机构执行的基准价；震荡状态下价格偏离 VWAP ±N 个标准差后，做市与执行算法把价格拉回",
    formula="VWAP、σ=当日（纽约 17:00 起）成交量加权；dev=(C−VWAP)/σ；w=趋势权重(1h ER)；"
            "当日已走不足 30 分钟时为 0；MAD_Z( −dev·(1−w) )",
    risks=["趋势日逆势连续亏损", "开盘初期 VWAP 与 σ 不稳定", "单笔幅度受 σ 大小限制，低波动时段付不起点差"],
    freqs=SCALP_FREQS,
    added="2026-09-27",
)
def factor_anchored_vwap_band_reversion(df: pd.DataFrame, freq: str = "1MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    vwap, sd, nday = anchored_vwap(d)
    dev = ((d["close"] - vwap) / sd).clip(-5, 5)
    w = trend_weight(efficiency_ratio(d["close"], BPH[freq]))
    raw = (-dev * (1 - w)).where(nday >= BPH[freq] // 2, 0.0)
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="AnchoredVWAPTrendFollow",
    cn_name="日内 VWAP 趋势跟随",
    family="scalp",
    hypothesis="趋势日中价格持续停留在当日 VWAP 一侧并越走越远，说明机构在持续单向执行；顺着偏离方向做",
    formula="dev=(C−VWAP)/σ（当日）；ER_日=(C−当日开盘)/当日路径长度；w_日=clip((|ER_日|−0.2)/0.3,0,1)；"
            "当日已走不足 30 分钟时为 0；MAD_Z( clip(dev,±3)·w_日 )",
    risks=["趋势日在尾盘反转", "与 TT30（30MIN 趋势尾部）可能相关", "开盘初期效率比噪声大"],
    freqs=SCALP_FREQS,
    added="2026-09-27",
)
def factor_anchored_vwap_trend_follow(df: pd.DataFrame, freq: str = "5MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    vwap, sd, nday = anchored_vwap(d)
    dev = ((d["close"] - vwap) / sd).clip(-3, 3)
    g = pd.Series(trading_day(d.index), index=d.index)
    first_open = d["open"].groupby(g).transform("first")
    path = d["close"].diff().abs().where(nday > 0, 0.0).groupby(g).cumsum()
    er_day = (d["close"] - first_open) / (path + EPS)
    raw = (dev * trend_weight(er_day)).where(nday >= BPH[freq] // 2, 0.0)
    return rolling_mad_zscore(raw, p["norm"])
