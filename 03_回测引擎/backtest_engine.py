"""
backtest_engine.py
====================
指南 5.1 节"向量化回测引擎"，事件驱动版（逐bar检查SL/TP/HOLD到期，比纯向量化更精确，
但比事件驱动完整版简化——这是参数搜索/WF阶段用的版本，不是实盘前最终验证用的版本）。

⚠️ SL/TP 目前用对称的"信号bar当时的ATR"（跟 04_策略研究/signal_atr_momentum_v2.py 的
target一致），这是v1占位，不是真正的风控参数——第6章风控层会重新设计止损止盈逻辑。

严格因果：SL/TP判定只看信号bar之后的价格路径，不看信号bar本身之前的数据。
同一根bar内SL和TP都触发时，保守起见先判SL（假设最坏情况先发生）。
持仓期间出现的新信号一律忽略（指南5.1.4第2点），同一时刻最多一笔持仓。
"""

import numpy as np
import pandas as pd

from cost_model import SimpleCostModel


def run_backtest(df: pd.DataFrame, signal: pd.Series, target: pd.Series,
                  forward_bars: int, cost_model: SimpleCostModel) -> pd.DataFrame:
    """
    df: 必须有 time_utc/open/high/low/close，index 与 signal/target 对齐
    signal: {-1,0,1}
    target: 逐bar的SL=TP距离（美元），比如1倍ATR
    forward_bars: 最长持仓根数，到期还没碰到SL/TP就按收盘价强平（HOLD）
    """
    trades = []
    signal_positions = np.flatnonzero(signal.values != 0)
    n = len(df)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    busy_until = -1
    for pos in signal_positions:
        if pos <= busy_until:
            continue
        if pos + forward_bars >= n:
            continue
        direction = 1 if signal.iloc[pos] > 0 else -1
        entry_price = close[pos]
        dist = target.iloc[pos]
        if pd.isna(dist) or dist <= 0:
            continue

        exit_price, exit_reason, exit_idx = None, None, None
        for j in range(pos + 1, pos + 1 + forward_bars):
            if direction == 1:
                hit_sl = (low[j] - entry_price) <= -dist
                hit_tp = (high[j] - entry_price) >= dist
                if hit_sl:
                    exit_price, exit_reason, exit_idx = entry_price - dist, "SL", j
                    break
                if hit_tp:
                    exit_price, exit_reason, exit_idx = entry_price + dist, "TP", j
                    break
            else:
                hit_sl = (entry_price - high[j]) <= -dist
                hit_tp = (entry_price - low[j]) >= dist
                if hit_sl:
                    exit_price, exit_reason, exit_idx = entry_price + dist, "SL", j
                    break
                if hit_tp:
                    exit_price, exit_reason, exit_idx = entry_price - dist, "TP", j
                    break

        if exit_price is None:
            exit_idx = pos + forward_bars
            exit_price = close[exit_idx]
            exit_reason = "HOLD"

        busy_until = exit_idx
        raw_pnl = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        net_pnl = raw_pnl - cost_model.round_trip_cost()

        trades.append({
            "entry_idx": pos,
            "exit_idx": exit_idx,
            "entry_time": df["time_utc"].iloc[pos],
            "exit_time": df["time_utc"].iloc[exit_idx],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "target": dist,
            "raw_pnl": raw_pnl,
            "pnl": net_pnl,
            "return_pct": net_pnl / entry_price,
            "bars_held": exit_idx - pos,
            "exit_reason": exit_reason,
        })

    return pd.DataFrame(trades)


def calc_sharpe(pnl: pd.Series) -> float:
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return pnl.mean() / pnl.std()


def calc_max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    running_max = cum.cummax()
    return (cum - running_max).min()


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    return {
        "n_trades": len(trades),
        "win_rate": (trades["pnl"] > 0).mean(),
        "total_pnl": trades["pnl"].sum(),
        "avg_pnl": trades["pnl"].mean(),
        "sharpe": calc_sharpe(trades["pnl"]),
        "max_drawdown": calc_max_drawdown(trades["pnl"]),
        "avg_bars_held": trades["bars_held"].mean(),
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
    }
