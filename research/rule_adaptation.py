# -*- coding: utf-8 -*-
"""
规则适配：IntradayReversalCore（5MIN / 15MIN）。纪律见 .claude/skills/xauusd-factor-mining/references/rule_adaptation.md

预先登记的候选规则（运行前写定，运行后不改）：
  R0 迟滞（统一规则，对照）          entry=1.5, exit=0.3
  R1 固定持有期（到期仍同向则续持） k ∈ {1.0,1.5,2.0} × h ∈ {h*/2, h*, 2h*}；h* 取样本内 IC 衰减峰值（5MIN=8，15MIN=2）
  R2 信号回归到 0 出场                k ∈ {1.0,1.5,2.0}
  R3 成本过滤（ATR_{t-1} ≥ m×0.2$ 才开仓），套在 R1/R2 中样本内最优者上，m ∈ {5,10}
选择标准：样本内（2009-2019）"ATR 单位、按当前成本折算"的日度夏普；样本外（2020-）只跑一次。
共同执行规则：≤1H 换日前平仓 / 换日后 1 小时不开仓（与 factors.evaluate.execution_signal 相同窗口）。
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.evaluate import BAR, SPREAD, SWAP, swap_units
from research.factor_correlation import factor_values

SPLIT = pd.Timestamp("2020-01-01")
FACTOR = "IntradayReversalCore"
H_STAR = {"5MIN": 8, "15MIN": 2}
TRIALS_CSV = "reports/rule_trials.csv"


def flat_mask(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    ny = idx.tz_localize("UTC").tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    start = 17 * 60 - 2 * pd.Timedelta(BAR[freq]).seconds // 60
    return np.asarray((mins >= start) & (mins < 18 * 60))


def target_positions(z: np.ndarray, flat: np.ndarray, gate: np.ndarray, rule: str, k: float,
                     h: int = 0, exit_: float = 0.3) -> np.ndarray:
    """状态机：返回第 t 根收盘后的目标仓位（t+1 开盘成交）。gate=False 时不允许新开仓/反手。"""
    n = len(z)
    out = np.zeros(n)
    pos, cnt = 0.0, 0
    for t in range(n):
        if flat[t]:
            pos, cnt = 0.0, 0
            continue
        zt = z[t]
        s = 1.0 if zt > k else (-1.0 if zt < -k else 0.0)
        can = gate[t]
        if rule == "hyst":
            if s != 0 and can:
                pos = s
            elif abs(zt) < exit_:
                pos = 0.0
        elif rule == "zero":
            if s != 0 and s != pos and can:
                pos = s
            elif pos != 0 and zt * pos <= 0:
                pos = 0.0
        elif rule == "fixed":
            if pos == 0:
                if s != 0 and can:
                    pos, cnt = s, h
            else:
                cnt -= 1
                if s == -pos and can:
                    pos, cnt = s, h
                elif cnt <= 0:
                    if s == pos:
                        cnt = h
                    else:
                        pos = 0.0
        out[t] = pos
    return out


def metrics(df: pd.DataFrame, pos: pd.Series, a_prev: pd.Series, atr_now: float, mask: np.ndarray) -> dict:
    d, p, a = df[mask], pos[mask], a_prev[mask]
    move = d["open"].shift(-1) - d["open"]
    yr = d.index.year
    m_atr = move / a
    m_atr = m_atr - m_atr.groupby(yr).transform("mean")            # 逐年去漂移
    dpos = p.diff().abs().fillna(p.abs())
    sw = p.abs() * swap_units(d.index)
    # ATR 单位、按当前成本折算
    pnl_atr = (p * m_atr - dpos * (SPREAD / 2) / atr_now - sw * SWAP / atr_now).fillna(0.0)
    daily = pnl_atr.groupby(trading_day(pnl_atr.index)).sum()
    trips = float(dpos.sum() / 2)
    # 真实历史美元口径
    net_usd = float((p * move - dpos * SPREAD / 2 - sw * SWAP).fillna(0.0).sum())
    net_ex = float((p * (move - move.mean()) - dpos * SPREAD / 2 - sw * SWAP).fillna(0.0).sum())
    return {
        "trips": int(round(trips)),
        "gross_atr_per_trip": round(float((p * m_atr).sum() / max(trips, 1)), 4),
        "cost_atr_per_trip_now": round(float((dpos.sum() * SPREAD / 2 + sw.sum() * SWAP) / max(trips, 1) / atr_now), 4),
        "sharpe_atr_now": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else float("nan"),
        "net_usd": round(net_usd, 0), "net_ex_drift_usd": round(net_ex, 0),
        "time_in_mkt": round(float((p != 0).mean()), 3),
    }


def run_trial(df, z, flat, a_prev, atr_now, freq, rule, k, h=0, m=None):
    gate = np.ones(len(z), bool) if m is None else np.nan_to_num(a_prev.to_numpy()) >= m * SPREAD
    tgt = target_positions(np.nan_to_num(z.to_numpy()), flat, gate, rule, k, h)
    pos = pd.Series(tgt, index=df.index).shift(1).fillna(0.0)
    ins = df.index < SPLIT
    return metrics(df, pos, a_prev, atr_now, ins), metrics(df, pos, a_prev, atr_now, ~ins)


def main(freq: str) -> pd.DataFrame:
    df = pd.read_pickle(f"data/cache/{freq}.pkl")
    z = factor_values(freq, [FACTOR])[FACTOR]
    flat = flat_mask(df.index, freq)
    a = atr(df, params(freq)["atr"])
    a_prev = a.shift(1)
    atr_now = float(a[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
    hs = H_STAR[freq]

    trials = [("R0 迟滞", "hyst", 1.5, 0, None)]
    trials += [(f"R1 固定h k={k} h={h}", "fixed", k, h, None) for k in (1.0, 1.5, 2.0) for h in (max(1, hs // 2), hs, 2 * hs)]
    trials += [(f"R2 回归到0 k={k}", "zero", k, 0, None) for k in (1.0, 1.5, 2.0)]
    rows = []
    for name, rule, k, h, m in trials:
        i, o = run_trial(df, z, flat, a_prev, atr_now, freq, rule, k, h, m)
        rows.append((name, rule, k, h, m, i, o))
    best = max((r for r in rows if r[1] != "hyst"), key=lambda r: r[5]["sharpe_atr_now"])
    for m in (5, 10):
        name = f"R3 成本过滤 m={m} + [{best[0]}]"
        i, o = run_trial(df, z, flat, a_prev, atr_now, freq, best[1], best[2], best[3], m)
        rows.append((name, best[1], best[2], best[3], m, i, o))

    sel = max(rows, key=lambda r: r[5]["sharpe_atr_now"])
    print(f"\n{'#' * 110}\n{freq}  当前 ATR ≈ ${atr_now:.2f}；点差 0.2 ≈ {SPREAD / atr_now:.3f} ATR   样本内选中：{sel[0]}")
    table = pd.DataFrame([{
        "规则": r[0], "选中": "★" if r is sel else "",
        "夏普内(ATR,今成本)": r[5]["sharpe_atr_now"], "夏普外(ATR,今成本)": r[6]["sharpe_atr_now"],
        "毛利/笔ATR 内": r[5]["gross_atr_per_trip"], "外": r[6]["gross_atr_per_trip"], "成本/笔ATR": r[6]["cost_atr_per_trip_now"],
        "次数外": r[6]["trips"], "持仓外": r[6]["time_in_mkt"],
        "美元净利外": r[6]["net_usd"], "去漂移美元外": r[6]["net_ex_drift_usd"]} for r in rows]).set_index("规则")
    print(table.to_string())

    new = not os.path.exists(TRIALS_CSV) or os.path.getsize(TRIALS_CSV) == 0
    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "factor", "freq", "cluster", "rule", "params", "selected_on", "is_sharpe",
                        "is_net_ex_drift", "oos_sharpe", "oos_net_ex_drift", "oos_atr_edge_ratio",
                        "n_trials_for_factor", "note"])
        for r in rows:
            ratio = r[6]["gross_atr_per_trip"] / r[6]["cost_atr_per_trip_now"] if r[6]["cost_atr_per_trip_now"] else float("nan")
            w.writerow([date.today(), FACTOR, freq, "VWAPDev+VWCT", r[1], f"k={r[2]};h={r[3]};m={r[4]}",
                        "IS sharpe_atr_now", r[5]["sharpe_atr_now"], r[5]["net_ex_drift_usd"],
                        r[6]["sharpe_atr_now"], r[6]["net_ex_drift_usd"], round(ratio, 2), len(rows),
                        ("SELECTED " if r is sel else "") + r[0]])
    return table


if __name__ == "__main__":
    for fq in sys.argv[1:] or ["5MIN", "15MIN"]:
        main(fq)
