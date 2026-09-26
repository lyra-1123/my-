"""
regime_split.py
================
指南 5.4 节"多Regime验证"。实现了4种切分方法里的前2种：
  方法1 波动率（ATR vs 其100根均值）
  方法2 趋势强度（ADX>25 趋势市，否则震荡市）
HMM（需要额外装hmmlearn）和事件窗口（需要宏观事件日历数据）暂未实现。

⚠️ 与指南模板的一个差异：指南按 exit_time 给交易打regime标签，这里按 entry_idx（入场那根bar）
打标签——"这笔交易是在什么regime下开的"才是实盘能用来做过滤的条件，出场时的regime入场时还不知道。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from compute_indicators import compute_adx, compute_atr  # noqa: E402

from backtest_engine import calc_max_drawdown, calc_sharpe  # noqa: E402


def regime_by_volatility(df: pd.DataFrame, atr_window: int = 100) -> pd.Series:
    atr = df["atr"] if "atr" in df else compute_atr(df)
    atr_ma = atr.rolling(atr_window).mean()
    return pd.Series(
        np.where(atr > atr_ma * 1.2, "high_vol", np.where(atr < atr_ma * 0.8, "low_vol", "normal_vol")),
        index=df.index,
    )


def regime_by_trend(df: pd.DataFrame, adx_period: int = 14, adx_threshold: float = 25) -> pd.Series:
    adx = compute_adx(df, adx_period)
    return pd.Series(np.where(adx > adx_threshold, "trending", "ranging"), index=df.index)


def multi_regime_validation(trades: pd.DataFrame, regime: pd.Series, min_trades: int = 30) -> dict:
    labels = regime.iloc[trades["entry_idx"].values].values
    reports = {}
    for label in sorted(set(labels)):
        sub = trades[labels == label]
        if len(sub) < min_trades:
            reports[label] = {"n_trades": len(sub), "warning": f"样本不足 < {min_trades} 笔"}
            continue
        reports[label] = {
            "n_trades": len(sub),
            "win_rate": (sub["pnl"] > 0).mean(),
            "total_pnl": sub["pnl"].sum(),
            "sharpe_per_trade": calc_sharpe(sub["pnl"]),
            "max_drawdown": calc_max_drawdown(sub["pnl"]),
        }
    return reports
