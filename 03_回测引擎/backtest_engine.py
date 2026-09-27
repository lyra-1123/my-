"""
backtest_engine.py
====================
指南 5.1 节"向量化回测引擎"，事件驱动版：逐笔调用 lib/goldq/exits.py 的 simulate_trade()，
出场规则（止损/止盈/分批/指标反转/时间/缺口）全部定义在那里，第4章筛选用的是同一份实现。

持仓期间出现的新信号一律忽略（指南5.1.4第2点），同一时刻最多一笔持仓。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.exits import ExitRule, Market, simulate_trade  # noqa: E402

from cost_model import SimpleCostModel  # noqa: E402


def run_backtest(df: pd.DataFrame, market: Market, signal: pd.Series, rule: ExitRule,
                  cost_model: SimpleCostModel) -> pd.DataFrame:
    """df 只用来取时间戳；market = prepare_market(df, ...)；signal ∈ {-1,0,1}，与 df 行对齐。"""
    trades = []
    sig = signal.to_numpy()
    cost = cost_model.round_trip_cost()
    busy_until = -1
    for pos in np.flatnonzero(sig != 0):
        if pos <= busy_until:
            continue
        t = simulate_trade(market, int(pos), 1 if sig[pos] > 0 else -1, rule)
        if t is None:
            continue
        busy_until = t["exit_idx"]
        t["pnl"] = t["raw_pnl"] - cost - cost_model.swap_cost(t["direction"], t["nights"])
        t["net_r"] = t["pnl"] / t["risk"]
        t["return_pct"] = t["pnl"] / t["entry_price"]
        trades.append(t)

    out = pd.DataFrame(trades)
    if not out.empty:
        times = df["time_utc"].to_numpy()
        out["entry_time"] = times[out["entry_idx"].to_numpy()]
        out["exit_time"] = times[out["exit_idx"].to_numpy()]
    return out


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
        "gross_pnl": trades["raw_pnl"].sum(),
        "total_cost": (trades["raw_pnl"] - trades["pnl"]).sum(),
        "n_ambiguous": int(trades["ambiguous"].sum()),
        "avg_nights": trades["nights"].mean(),
        "avg_raw_r": trades["raw_r"].mean(),
        "avg_net_r": trades["net_r"].mean(),
    }
