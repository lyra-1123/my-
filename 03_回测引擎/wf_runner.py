"""
wf_runner.py
=============
指南 5.2 节"Walk-Forward窗口设计"。

核心原则（指南原话）："OOS窗口应由产生300+笔交易所需时间决定，而非固定bars数量。"
本项目信号频率很低（ATR动量v2约0.7次/天），直接照抄指南5.2.4的示例参数
（is_days=60, oos_days=30，对应5-10笔/天的短线策略）完全不够——那组参数是给
高频得多的策略设计的。所以这里按信号的实际频率反推窗口天数，而不是硬编码天数。
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class WFConfig:
    is_days: int
    oos_days: int
    step_days: int
    n_windows: int
    trades_per_day: float
    min_oos_trades: int


def design_wf_windows(df: pd.DataFrame, n_trades: int, min_oos_trades: int = 300,
                       is_oos_ratio: float = 1.0, min_oos_days: int = 30) -> tuple[list, WFConfig]:
    """
    按实际成交频率反推 OOS 窗口天数，确保每个 OOS 窗口预期能有 >= min_oos_trades 笔交易。
    n_trades: 全样本回测实际成交笔数（不是信号数——持仓中的信号会被忽略，成交数比信号数少）。
    is_oos_ratio: IS窗口天数 = OOS窗口天数 * 这个比例（指南5.2.3：中频套利用1:1，
                  低频趋势跟踪用2:1）。
    min_oos_days: OOS窗口天数下限，避免高频信号算出几天的窗口（只在冒烟测试时调小）。
    窗口之间不重叠（step_days = oos_days），这是最保守的做法，避免同一段OOS数据被
    多个窗口重复计入"连续通过"次数。
    """
    total_days = (df["time_utc"].iloc[-1] - df["time_utc"].iloc[0]).days
    trades_per_day = n_trades / total_days if total_days > 0 else 0

    if trades_per_day <= 0:
        raise ValueError("成交次数为0，无法设计WF窗口")

    oos_days = max(min_oos_days, int(-(-min_oos_trades // trades_per_day)))  # 向上取整
    is_days = int(oos_days * is_oos_ratio)
    step_days = oos_days

    bars_per_day = len(df) / total_days
    is_bars = int(is_days * bars_per_day)
    oos_bars = int(oos_days * bars_per_day)
    step_bars = max(1, int(step_days * bars_per_day))

    windows = []
    for start in range(0, len(df) - is_bars - oos_bars, step_bars):
        is_start, is_end = start, start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        windows.append({
            "is_start_idx": is_start, "is_end_idx": is_end,
            "oos_start_idx": oos_start, "oos_end_idx": oos_end,
            "is_start": df["time_utc"].iloc[is_start], "is_end": df["time_utc"].iloc[is_end - 1],
            "oos_start": df["time_utc"].iloc[oos_start], "oos_end": df["time_utc"].iloc[oos_end - 1],
        })

    config = WFConfig(is_days=is_days, oos_days=oos_days, step_days=step_days,
                       n_windows=len(windows), trades_per_day=trades_per_day,
                       min_oos_trades=min_oos_trades)
    return windows, config


def is_window_pass(oos_trades: pd.DataFrame, min_trades: int = 10) -> bool:
    """单个OOS窗口的通过判定：至少有min_trades笔交易、总PnL为正、逐笔Sharpe为正。"""
    if len(oos_trades) < min_trades:
        return False
    return oos_trades["pnl"].sum() > 0 and oos_trades["pnl"].mean() > 0


def rolling_wf_pass(window_results: list[bool], min_consecutive: int = 6) -> dict:
    """指南5.2.5：单次WF通过不够，必须连续>=6个OOS窗口通过。"""
    consecutive_pass = 0
    max_consecutive = 0
    for passed in window_results:
        if passed:
            consecutive_pass += 1
            max_consecutive = max(max_consecutive, consecutive_pass)
        else:
            consecutive_pass = 0
    return {
        "max_consecutive": max_consecutive,
        "min_required": min_consecutive,
        "passed": max_consecutive >= min_consecutive,
    }
