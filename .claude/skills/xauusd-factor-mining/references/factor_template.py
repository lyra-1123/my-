# -*- coding: utf-8 -*-
"""
新因子模板：复制到 factors/library/<家族>.py，并在 factors/library/__init__.py 中 import 该模块。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore
from ..registry import register


@register(
    name="MyFactorName",                  # 英文驼峰，直观
    cn_name="中文名",
    family="breakout",                    # breakout/microstructure/volatility/trend/reversal/session/pullback/...
    hypothesis="一句话：为什么有效（经济学/行为金融/微观结构依据）",
    formula="MAD_Z( 算子表达式 )",
    risks=["过拟合点", "流动性/成本陷阱", "风格暴露", "数据定义敏感性"],
    freqs=("5MIN", "15MIN", "30MIN", "1H", "4H", "1D"),
    added="YYYY-MM-DD",
)
def factor_my_factor(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    """
    【逻辑】...
    【算子】...
    【风险点】...
    """
    p = params(freq, **kw)                 # 频率先验参数（chan/atr/vol_base/thrust/sq_short/sq_long/norm/trend/intraday）
    d = check_input(df)                    # 只保留 OHLCV
    # --- 只用 <= t 的数据；基准类统计量用 shift(1) 排除当前 K 线 ---
    raw = pd.Series(0.0, index=d.index)
    # --- 因子值 > 0 表示看多；反转类在这里取负号 ---
    return rolling_mad_zscore(raw, p["norm"])
