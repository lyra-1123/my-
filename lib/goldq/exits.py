"""
exits.py
=========
出场规则的唯一实现，第4章（04_策略研究 的方向性筛选）和第5章（03_回测引擎 的回测）共用，
保证两章用的是同一套"一笔交易怎么结束"的定义。

止损（stop）：
  structure  入场时放在最近 N 根bar（含信号bar）最低点/最高点外侧 buffer×ATR，之后不动；
             风险距离夹在 [min_risk_atr, max_risk_atr]×ATR 之间，避免过近（成本吃光）或过远
  atr_trail  初始 atr_stop_mult×ATR，之后按"入场以来最有利价 ∓ atr_stop_mult×入场ATR"只收紧不放松
  atr_fixed  atr_stop_mult×ATR，之后不动（旧版对称SL/TP的口径，仅为复现历史结果保留）

止盈（take_profit）：
  fixed_r    止盈 = 入场 ± r_multiple × 风险距离(1R)
  none       不设止盈，只靠止损/移动止损/时间出场
  indicator  收盘时出现反向指标信号（由调用方给出 exit_long / exit_short 布尔序列）按收盘价离场
  partial    价格到 partial_r×R 平掉 partial_fraction 仓位，止损移到保本；剩余仓位用移动止损
             （atr_trail 沿用原距离；structure 以 1R 为距离开始移动）

执行假设（全部偏保守、严格因果）：
  - 信号bar收盘价进场；第 j 根bar的止损位只用到第 j-1 根为止的信息
  - 同一根bar内先判止损，再判分批/止盈，最后看收盘指标；止损与止盈同bar都触发记为 ambiguous
  - 跳空越过止损按开盘价成交（更差），跳空越过止盈也按开盘价成交
  - 下一根bar与当前bar间隔超过 max_gap_minutes（周末/假期）：信号bar本身后面就是缺口则不开仓；
    持仓中遇到缺口，按缺口前最后一根bar收盘价平仓（GAP）
  - 持仓 max_bars 根仍未出场按收盘价平仓（TIME）
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

STOP_MODES = ("structure", "atr_trail", "atr_fixed")
TP_MODES = ("fixed_r", "none", "indicator", "partial")


@dataclass(frozen=True)
class ExitRule:
    name: str
    stop: str
    take_profit: str
    max_bars: int = 24
    atr_stop_mult: float = 1.5
    structure_lookback: int = 10
    structure_buffer_atr: float = 0.2
    min_risk_atr: float = 0.5
    max_risk_atr: float = 3.0
    r_multiple: float = 2.0
    partial_r: float = 1.0
    partial_fraction: float = 0.5
    max_gap_minutes: int = 180

    def __post_init__(self):
        if self.stop not in STOP_MODES:
            raise ValueError(f"stop 必须是 {STOP_MODES} 之一，收到 {self.stop}")
        if self.take_profit not in TP_MODES:
            raise ValueError(f"take_profit 必须是 {TP_MODES} 之一，收到 {self.take_profit}")


def standard_exit_family(max_bars: int = 24) -> list[ExitRule]:
    """2026-09-27 与用户确认的出场组合：止损{结构, ATR移动} × 止盈{2R, 不设, 指标反转, 分批}。
    "结构止损 + 不设止盈"没有任何主动出场机制（只剩时间出场），不纳入。"""
    return [
        ExitRule("struct_2R", "structure", "fixed_r", max_bars),
        ExitRule("struct_ind", "structure", "indicator", max_bars),
        ExitRule("struct_partial", "structure", "partial", max_bars),
        ExitRule("trail_2R", "atr_trail", "fixed_r", max_bars),
        ExitRule("trail_only", "atr_trail", "none", max_bars),
        ExitRule("trail_ind", "atr_trail", "indicator", max_bars),
        ExitRule("trail_partial", "atr_trail", "partial", max_bars),
    ]


def legacy_symmetric_exit(atr_mult: float = 1.0, max_bars: int = 6) -> ExitRule:
    """第5章v1用过的对称 SL=TP=k×ATR、30分钟到期口径。"""
    return ExitRule(f"sym{atr_mult}ATR_{max_bars}bars", "atr_fixed", "fixed_r", max_bars,
                    atr_stop_mult=atr_mult, r_multiple=1.0)


def ema_macd_reversal_exits(close: pd.Series, ema: pd.Series,
                            golden_cross: pd.Series, dead_cross: pd.Series) -> tuple[pd.Series, pd.Series]:
    """指标反转出场：多单在收盘跌破EMA或MACD死叉时离场，空单在收盘站上EMA或MACD金叉时离场。"""
    exit_long = (close < ema) | dead_cross.astype(bool)
    exit_short = (close > ema) | golden_cross.astype(bool)
    return exit_long, exit_short


@dataclass
class Market:
    time_min: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    exit_long: np.ndarray | None
    exit_short: np.ndarray | None

    def __len__(self):
        return len(self.close)


def prepare_market(df: pd.DataFrame, exit_long: pd.Series | None = None,
                   exit_short: pd.Series | None = None) -> Market:
    """df 需要 time_utc/open/high/low/close/atr。一次转成 numpy，供逐笔模拟反复使用。"""
    t = pd.to_datetime(df["time_utc"])
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    time_min = t.to_numpy().astype("datetime64[m]").astype("int64")  # 与底层存储精度(ns/us)无关
    return Market(
        time_min=time_min,
        open=df["open"].to_numpy(dtype=float), high=df["high"].to_numpy(dtype=float),
        low=df["low"].to_numpy(dtype=float), close=df["close"].to_numpy(dtype=float),
        atr=df["atr"].to_numpy(dtype=float),
        exit_long=None if exit_long is None else exit_long.fillna(False).to_numpy(dtype=bool),
        exit_short=None if exit_short is None else exit_short.fillna(False).to_numpy(dtype=bool),
    )


def simulate_trade(m: Market, pos: int, direction: int, rule: ExitRule) -> dict | None:
    """从 pos（信号bar）收盘开一笔 1 单位仓位，按 rule 模拟到出场。返回成本前结果，无法开仓返回 None。"""
    n = len(m)
    atr = m.atr[pos]
    if pos + 1 >= n or not np.isfinite(atr) or atr <= 0:
        return None
    if m.time_min[pos + 1] - m.time_min[pos] > rule.max_gap_minutes:
        return None
    if rule.take_profit == "indicator" and (m.exit_long is None or m.exit_short is None):
        raise ValueError("indicator 止盈需要在 prepare_market 里传入 exit_long / exit_short")

    d = direction
    entry = m.close[pos]

    if rule.stop == "structure":
        lo = max(0, pos - rule.structure_lookback + 1)
        extreme = m.low[lo:pos + 1].min() if d == 1 else m.high[lo:pos + 1].max()
        risk = d * (entry - extreme) + rule.structure_buffer_atr * atr
        risk = float(np.clip(risk, rule.min_risk_atr * atr, rule.max_risk_atr * atr))
    else:
        risk = rule.atr_stop_mult * atr
    stop = entry - d * risk

    trailing = rule.stop == "atr_trail"
    trail_dist = rule.atr_stop_mult * atr if trailing else risk
    tp = entry + d * rule.r_multiple * risk if rule.take_profit == "fixed_r" else None
    partial_level = entry + d * rule.partial_r * risk if rule.take_profit == "partial" else None
    ind_flags = (m.exit_long if d == 1 else m.exit_short) if rule.take_profit == "indicator" else None

    remaining = 1.0
    realized = 0.0          # 已平仓部分的 成交价×数量（方向无关，最后统一算）
    partial_filled = False
    ambiguous = False
    best = entry
    reason = None
    exit_idx = None
    last = min(pos + rule.max_bars, n - 1)

    for j in range(pos + 1, last + 1):
        if j > pos + 1 and m.time_min[j] - m.time_min[j - 1] > rule.max_gap_minutes:
            realized += remaining * m.close[j - 1]
            remaining, reason, exit_idx = 0.0, "GAP", j - 1
            break

        o, h, l, c = m.open[j], m.high[j], m.low[j], m.close[j]
        fav, adv = (h, l) if d == 1 else (l, h)

        if d * (adv - stop) <= 0:
            fill = stop if d * (o - stop) >= 0 else o
            upside = tp if tp is not None else (partial_level if not partial_filled else None)
            ambiguous = upside is not None and d * (fav - upside) >= 0
            realized += remaining * fill
            remaining, exit_idx = 0.0, j
            if partial_filled and d * (stop - entry) <= 0:
                reason = "BE"
            elif trailing or partial_filled:
                reason = "TRAIL" if d * (stop - entry) > 0 else "SL"
            else:
                reason = "SL"
            break

        if partial_level is not None and not partial_filled and d * (fav - partial_level) >= 0:
            fill = partial_level if d * (o - partial_level) <= 0 else o
            realized += rule.partial_fraction * fill
            remaining -= rule.partial_fraction
            partial_filled = True
            trailing = True

        if tp is not None and d * (fav - tp) >= 0:
            fill = tp if d * (o - tp) <= 0 else o
            realized += remaining * fill
            remaining, reason, exit_idx = 0.0, "TP", j
            break

        if ind_flags is not None and ind_flags[j]:
            realized += remaining * c
            remaining, reason, exit_idx = 0.0, "IND", j
            break

        best = max(best, fav) if d == 1 else min(best, fav)
        if trailing:
            new_stop = best - d * trail_dist
            if d * (new_stop - stop) > 0:
                stop = new_stop
        if partial_filled and d * (entry - stop) > 0:
            stop = entry

    if remaining > 0:
        realized += remaining * m.close[last]
        remaining, reason, exit_idx = 0.0, "TIME", last

    exit_price = realized  # 数量总和=1，加权平均成交价
    raw_pnl = d * (exit_price - entry)
    return {
        "entry_idx": pos, "exit_idx": exit_idx, "direction": d,
        "entry_price": entry, "exit_price": exit_price, "risk": risk,
        "raw_pnl": raw_pnl, "raw_r": raw_pnl / risk,
        "bars_held": exit_idx - pos, "exit_reason": reason,
        "ambiguous": ambiguous, "partial_filled": partial_filled,
    }
