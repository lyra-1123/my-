# -*- coding: utf-8 -*-
"""
家族：组合（第六批，2026-09-27）。

来源：research/factor_correlation.py、research/combo_weights.py（成分选择只用样本内信息）。
  - |t|≥5 的日内因子只构成 6 个信号簇；扣除其他簇后，只有 −VWAPDeviation、−VolWeightedCloseThrust
    的残差 ICIR 在样本内显著为正（5MIN：+0.71 / +0.35），其余簇的独有部分为噪声甚至反向。
  - 权重：等权。样本内估计的最大 ICIR 权重在样本外反而更差（过拟合），因此不用。
"""
from __future__ import annotations

import pandas as pd

from ..core import params, rolling_mad_zscore
from ..registry import register
from .trend import factor_vwap_deviation
from .volume_breakout import factor_vol_weighted_close_thrust


@register(
    name="IntradayReversalCore",
    cn_name="日内反转核心组合",
    family="combo",
    hypothesis="日内短期反转的两个独立来源等权组合：价格相对近期成交成本（VWAP）的偏离回归，"
               "与放量强势 K 线（收盘位置/实体/振幅）的回吐（流动性冲击补偿）；两者残差 ICIR 均显著，互为补充",
    formula="MAD_Z( 0.5*(−VWAPDeviation) + 0.5*(−VolWeightedCloseThrust) )，成分均为已标准化因子",
    risks=["信息主要在分布中部，尾部交易的单笔幅度小（见 lessons L8），统一迟滞规则下未必盈利",
           "两成分相关约 0.5，组合提升有限（样本外 ICIR 1.04 vs 最好单因子 1.00）",
           "成分选择依赖样本内残差 ICIR 的判断，存在选择偏差"],
    freqs=("5MIN", "15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_intraday_reversal_core(df: pd.DataFrame, freq: str = "5MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    a = -factor_vwap_deviation(df, freq, **kw)
    b = -factor_vol_weighted_close_thrust(df, freq, **kw)
    raw = 0.5 * a.fillna(0.0) + 0.5 * b.fillna(0.0)
    return rolling_mad_zscore(raw, p["norm"])
