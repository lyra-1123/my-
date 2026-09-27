# -*- coding: utf-8 -*-
"""
家族：日内短线的新信息来源（第十三批，2026-09-27，频率 15MIN / 30MIN / 1H）。

预登记：reports/intraday_batch13_prereg.md（参数全部事先固定，不做网格搜索）。
与已有日内因子的区别：不再对价格路径做变换（L26），而是换信息来源——
  A 日历时钟（时段季节性）、B 结算时点（COMEX 结算前的日内动量）、C 收益分布的三阶矩（已实现偏度）。
交易日锚点：纽约 17:00（core.trading_day），与换日、过夜费一致；时钟用纽约当地时间（自动处理美国夏令时）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, check_input, params, rolling_mad_zscore, trading_day
from ..registry import register

BAR_MIN = {"15MIN": 15, "30MIN": 30, "1H": 60}
SEASON_DAYS = 250          # 季节性估计窗口：过去 250 个交易日（约 1 年）
SEASON_K = 4               # 预测接下来 4 根 K 线（= 评估器 15MIN/30MIN/1H 的主持有期）


def ny_clock(index: pd.DatetimeIndex) -> np.ndarray:
    """纽约当地时间的分钟数（0~1439）。"""
    ny = index.tz_localize("UTC").tz_convert("America/New_York")
    return np.asarray(ny.hour * 60 + ny.minute)


@register(
    name="IntradaySeasonality",
    cn_name="日内时段季节性",
    family="intraday_info",
    hypothesis="24 小时市场中各时段参与者不同（亚洲实物买盘、伦敦定价、纽约期货/ETF），若某些时段在一年内系统性偏涨/偏跌，"
               "过去一年同一时段的平均收益可以预测接下来几根 K 线；信息来源是日历时钟，与价格路径类策略低相关",
    formula="μ_s,σ²_s=此前 250 个交易日同一纽约时刻 s 的收益均值/方差（再减去全时段均值，去牛市漂移）；"
            "raw=Σ_{j=1..4} μ_{s(t+j)} / sqrt(Σ σ²_{s(t+j)}/n)；MAD_Z(raw)",
    risks=["季节性可能随市场结构（交易时段、ETF 流量）漂移，一年窗口反应慢",
           "伦敦夏令时与美国错位 2~3 周，伦敦开盘所在时段每年有几周错位",
           "周一 18:00 时段含周末跳空，噪声大", "使用 K 线自身 UTC 时间戳（转换为纽约时间）"],
    freqs=("15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_intraday_seasonality(df: pd.DataFrame, freq: str = "30MIN", **kw) -> pd.Series:
    """
    【逻辑】用"已结束交易日"的同时刻收益估计日内形状，预测 t+1..t+4 的累计收益（t 统计量）。
    【算子】pivot(交易日 × 纽约时刻) → 按交易日滚动 250 日均值/方差 → shift(1 日) → 按未来时刻查表求和。
    【无未来函数】未来 K 线的时刻只由日历决定；μ/σ² 在交易日 D 只用 D 之前的交易日（shift(1)）。
    【风险点】见 risks；估计窗口固定为先验值 250 日，未调参。
    """
    p = params(freq, **kw)
    d = check_input(df)
    clock = ny_clock(d.index)
    td = trading_day(d.index)
    r = np.log(d["close"]).diff()
    P = pd.DataFrame({"td": td, "clk": clock, "r": r.to_numpy()}).pivot_table(index="td", columns="clk", values="r", aggfunc="mean")
    roll = P.rolling(SEASON_DAYS, min_periods=SEASON_DAYS // 2)
    mu, var, cnt = roll.mean().shift(1), roll.var().shift(1), roll.count().shift(1)
    mu = mu.sub(mu.mean(axis=1), axis=0)                  # 去掉整体漂移（牛市 beta），只保留日内形状
    se2 = (var / cnt.where(cnt > 0))
    cols = {c: i for i, c in enumerate(P.columns)}
    row = P.index.get_indexer(td)
    mu_a, se_a = mu.to_numpy(), se2.to_numpy()
    num = np.zeros(len(d)); den = np.zeros(len(d))
    step = BAR_MIN[freq]
    for j in range(1, SEASON_K + 1):
        col = np.array([cols.get(c, -1) for c in (clock + j * step) % 1440])
        ok = col >= 0
        m = np.where(ok, mu_a[row, np.where(ok, col, 0)], np.nan)
        s = np.where(ok, se_a[row, np.where(ok, col, 0)], np.nan)
        num += np.nan_to_num(m); den += np.nan_to_num(s)
    raw = pd.Series(np.where(den > 0, num / np.sqrt(den + EPS), np.nan), index=d.index)
    return rolling_mad_zscore(raw, p["norm"])


SETTLE_START, SETTLE_END = 12 * 60 + 30, 13 * 60 + 30     # 纽约 12:30 入场、13:30 出场（COMEX 结算 13:28~13:30）


def session_momentum(d: pd.DataFrame, freq: str, start: int, end: int) -> pd.Series:
    """
    当日（纽约 18:00 开盘起）到 start 时刻的累计收益 R，逐日滚动 MAD_Z（250 个交易日）；
    只在收盘时刻 ∈ [start, end) 的 K 线上非零 → 迟滞规则下 start 开盘入场、end 开盘出场。
    也用于事件研究的对照窗口（10:30、14:30）。
    """
    bar = BAR_MIN[freq]
    clock = ny_clock(d.index)
    td = pd.Series(trading_day(d.index), index=d.index)
    wk = np.asarray(d.index.tz_localize("UTC").tz_convert("America/New_York").dayofweek < 5)
    first_open = d["open"].groupby(td).transform("first")
    at_start = (clock == start - bar) & wk                  # 收盘于 start 的 K 线
    R = pd.Series(np.log(d["close"][at_start] / first_open[at_start]).to_numpy(), index=td[at_start].to_numpy())
    R = R[~R.index.duplicated(keep="last")]
    zR = rolling_mad_zscore(R, SEASON_DAYS)
    live = (clock >= start - bar) & (clock < end - bar) & wk
    out = pd.Series(0.0, index=d.index)
    out[live] = td[live].map(zR).to_numpy()
    return out.fillna(0.0)


@register(
    name="SettlementMomentum",
    cn_name="COMEX 结算前日内动量",
    family="intraday_info",
    hypothesis="Gao-Han-Li-Zhou（2018）日内动量：不频繁再平衡的资金与做市商对冲集中在收盘/结算前，"
               "当日到 12:30 的累计收益方向在 COMEX 结算前 1 小时（12:30~13:30 NY）延续",
    formula="R=ln(C@12:30 NY / 当日第一根开盘)；z=逐日 MAD_Z(R,250)；只在收盘∈[12:30,13:30) 的 K 线上取 z，其余为 0",
    risks=["每天只有 1 小时持仓，交易次数受迟滞门槛限制（统一评估只取 |z|>1.5 的极端日），以事件研究为主",
           "美国假日/提前收盘日结算时间不同", "13:30 前后常有美国数据/美联储讲话，窗口内噪声大",
           "使用 K 线自身 UTC 时间戳（转换为纽约时间）"],
    freqs=("15MIN", "30MIN"),
    added="2026-09-27",
)
def factor_settlement_momentum(df: pd.DataFrame, freq: str = "15MIN", **kw) -> pd.Series:
    """
    【逻辑】结算相关的对冲/再平衡顺着当日方向执行，在结算前推动价格延续。
    【算子】R=ln(C_12:30/O_day)；逐日 MAD_Z；广播到 12:30~13:30 的持仓 K 线。
    【无未来函数】R 在 12:30 收盘时已知；逐日标准化只用过去 250 日。
    【风险点】见 risks。
    """
    d = check_input(df)
    return session_momentum(d, freq, SETTLE_START, SETTLE_END)


SKEW_N = {"15MIN": 96, "30MIN": 48, "1H": 24}            # 1 个交易日的 K 线数


@register(
    name="RealizedSkewReversal",
    cn_name="已实现偏度反转",
    family="intraday_info",
    hypothesis="Amaya-Christoffersen-Jacobs-Vasquez（2015）：已实现偏度高（收益由少数大涨贡献，彩票型）之后收益偏低，"
               "负偏（少数大跌、止损踩踏）之后有补偿；信息来自收益分布的三阶矩而不是收益水平",
    formula="RSkew=√N·Σr³/(Σr²)^1.5（N=1 个交易日的 K 线数）；MAD_Z(−RSkew)",
    risks=["偏度与近期收益相关，可能只是短期反转（L1）的另一种写法，须看扣除 IntradayReversalCore 后的残差 ICIR",
           "单根大 K 线（数据公布、周末跳空）主导偏度", "1H 只有 24 个观测，估计噪声大"],
    freqs=("15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_realized_skew_reversal(df: pd.DataFrame, freq: str = "30MIN", **kw) -> pd.Series:
    """
    【逻辑】正偏 = 彩票型需求推高价格 → 之后回落；负偏 = 被动卖出 → 之后补偿。取负号使 >0 看多。
    【算子】r=Δln C；RSkew=√N·TS_SUM(r³,N)/TS_SUM(r²,N)^1.5；MAD_Z(−RSkew)。
    【无未来函数】只用 ≤t 的 N 根收益。
    【风险点】见 risks。
    """
    p = params(freq, **kw)
    d = check_input(df)
    n = SKEW_N[freq]
    r = np.log(d["close"]).diff()
    s2 = (r ** 2).rolling(n, min_periods=n).sum()
    s3 = (r ** 3).rolling(n, min_periods=n).sum()
    rskew = np.sqrt(n) * s3 / (s2 ** 1.5 + EPS)
    return rolling_mad_zscore(-rskew.where(s2 > EPS), p["norm"])
