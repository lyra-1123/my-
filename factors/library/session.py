# -*- coding: utf-8 -*-
"""
家族：交易时段接力（第二批，2026-09-27）。
注意：本因子使用 K 线时间戳（UTC）划分时段。时间戳是 K 线本身的索引而非额外数据字段，
但它引入了"日历"信息，风险自查中单列说明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, check_input, params, rolling_mad_zscore
from ..registry import register

LONDON = (7 * 60, 12 * 60)     # 伦敦时段 [07:00, 12:00) UTC
NY_HOLD = (12 * 60, 16 * 60)   # 纽约上半场持有窗口 [12:00, 16:00) UTC


@register(
    name="LondonNYSessionMomentum",
    cn_name="伦敦→纽约时段接力动量",
    family="session",
    hypothesis="日内动量（Gao-Han-Li-Zhou 2018）：早段价格发现由知情/机构资金主导，"
               "放量的伦敦时段方向在纽约开盘后被美国资金延续（信息逐步扩散 + 再平衡需求）",
    formula="日级 raw = log(C_Lon_end/O_Lon_start)/STD_60d * (1+max(log(V_Lon/MED20(V_Lon)),0))；"
            "MAD_Z 在日级序列上做，再广播到 NY 持有窗口内的 K 线，窗口外为 0",
    risks=["固定 UTC 时段未处理夏令时（伦敦/纽约开盘相差 1 小时）", "每天最多一个信号，样本量有限",
           "美国数据 13:30 UTC 公布可能直接覆盖伦敦方向"],
    freqs=("5MIN", "15MIN", "30MIN", "1H", "4H"),
    added="2026-09-27",
)
def factor_london_ny_session_momentum(df: pd.DataFrame, freq: str = "15MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    if not p["intraday"]:
        return pd.Series(0.0, index=d.index)
    tod = d.index.hour * 60 + d.index.minute
    day = d.index.normalize()

    in_lon = (tod >= LONDON[0]) & (tod < LONDON[1])
    lon = d[in_lon].groupby(day[in_lon]).agg(o=("open", "first"), c=("close", "last"), v=("volume", "sum"))
    lon_ret = np.log(lon["c"] / lon["o"])
    sd = lon_ret.shift(1).rolling(60, min_periods=20).std()                  # 只用之前的日子
    vc = lon["v"] / (lon["v"].shift(1).rolling(20, min_periods=5).median() + EPS)
    raw_day = lon_ret / (sd + EPS) * (1.0 + np.log(vc.clip(lower=EPS)).clip(lower=0))
    z_day = rolling_mad_zscore(raw_day, 250)

    in_hold = (tod >= NY_HOLD[0]) & (tod < NY_HOLD[1])
    out = pd.Series(0.0, index=d.index)
    out[in_hold] = z_day.reindex(day[in_hold]).to_numpy()
    return out.fillna(0.0)
