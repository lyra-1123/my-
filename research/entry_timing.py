# -*- coding: utf-8 -*-
"""
方向 1：用日内反转信号给 30MIN 趋势尾部策略做入场择时（不单独交易反转）。

预先登记（运行前写定）：
  基础策略：30MIN 趋势尾部簇成员 {TSMomentumVolScaled, TrendEfficiencyVolume, VWAPDeviation, MTFTrendResonance}
            及其等权组合（原方向 = 动量），统一迟滞规则（1.5/0.3）+ 换日前平仓；样本内夏普最高者为基础。
  择时（5MIN 网格执行，出场规则不变）：
    T0 立即入场（对照）
    T1 等待 5MIN IntradayReversalCore 与方向一致（r·dir ≥ k），最多等 W 根 5MIN，超时仍入场；k∈{0.5,1.0} × W∈{6,12}
    T2 同上但超时放弃该笔交易；k=1.0, W=12
  选择标准：样本内"ATR 单位、按当前成本折算"的日度夏普；样本外只跑一次。全部试验登记进 reports/rule_trials.csv。
时间对齐：30MIN K 线在 start+30min 收盘后，其状态才对 5MIN 网格可见（结束时间 merge_asof），无未来函数。
"""
from __future__ import annotations

import csv
from datetime import date

import numpy as np
import pandas as pd

from factors.core import atr, params, rolling_mad_zscore
from factors.evaluate import ENTRY, EXIT, execution_signal
from research.factor_correlation import factor_values
from research.rule_adaptation import SPLIT, TRIALS_CSV, flat_mask, metrics

MEMBERS = ["TSMomentumVolScaled", "TrendEfficiencyVolume", "VWAPDeviation", "MTFTrendResonance"]


def hysteresis_target(z: np.ndarray) -> np.ndarray:
    """统一迟滞规则的目标仓位（未 shift）。"""
    out = np.zeros(len(z)); pos = 0.0
    for t, zt in enumerate(z):
        if zt > ENTRY: pos = 1.0
        elif zt < -ENTRY: pos = -1.0
        elif abs(zt) < EXIT: pos = 0.0
        out[t] = pos
    return out


def align_30_to_5(s30: pd.Series, idx5: pd.DatetimeIndex) -> np.ndarray:
    """30MIN 值在其收盘（start+30min）后才可用；对齐到 5MIN K 线收盘时间（start+5min）。"""
    right = pd.DataFrame({"t": s30.index + pd.Timedelta("30min"), "v": s30.to_numpy()})
    left = pd.DataFrame({"t": idx5 + pd.Timedelta("5min")})
    return pd.merge_asof(left, right, on="t", direction="backward")["v"].fillna(0.0).to_numpy()


def timed_positions(desired: np.ndarray, r: np.ndarray, flat: np.ndarray, mode: str, k: float = 0.0, W: int = 0) -> np.ndarray:
    """5MIN 状态机：desired 为 30MIN 基础策略的目标方向；r 为 5MIN 反转信号（>0 看多）。"""
    n = len(desired); out = np.zeros(n)
    pos, pend, wait, skip = 0.0, 0.0, 0, 0.0
    for t in range(n):
        d = desired[t]
        if flat[t] or d == 0:
            pos, pend, wait = 0.0, 0.0, 0
            skip = 0.0 if d == 0 else skip
            out[t] = 0.0
            continue
        if d != skip:
            skip = 0.0
        if pos == d:
            out[t] = pos
            continue
        if pos != 0 and pos != d:          # 方向翻转：先平旧仓
            pos = 0.0
        if mode == "T0":
            pos = d
        elif d != skip:
            if pend != d:
                pend, wait = d, 0
            if r[t] * d >= k:
                pos, pend = d, 0.0
            else:
                wait += 1
                if wait >= W:
                    if mode == "T1":
                        pos, pend = d, 0.0
                    else:                   # T2：放弃，直到方向改变
                        skip, pend = d, 0.0
        out[t] = pos
    return out


def main() -> None:
    d30 = pd.read_pickle("data/cache/30MIN.pkl")
    d5 = pd.read_pickle("data/cache/5MIN.pkl")
    F30 = factor_values("30MIN", MEMBERS)
    bases = {m: F30[m] for m in MEMBERS}
    bases["等权组合"] = rolling_mad_zscore(F30.fillna(0.0).mean(axis=1), params("30MIN")["norm"])

    r5 = np.nan_to_num(factor_values("5MIN", ["IntradayReversalCore"])["IntradayReversalCore"].to_numpy())
    flat5 = flat_mask(d5.index, "5MIN")
    a5 = atr(d5, params("5MIN")["atr"]); a5_prev = a5.shift(1)
    atr_now = float(a5[d5.index >= d5.index[-1] - pd.Timedelta(days=365)].median())
    ins = d5.index < SPLIT

    def run(desired, mode, k=0.0, W=0):
        tgt = timed_positions(desired, r5, flat5, mode, k, W)
        pos = pd.Series(tgt, index=d5.index).shift(1).fillna(0.0)
        return metrics(d5, pos, a5_prev, atr_now, ins), metrics(d5, pos, a5_prev, atr_now, ~ins)

    rows = []
    # 1) 选基础策略（T0）
    desired_by = {}
    for name, z in bases.items():
        tgt30 = pd.Series(hysteresis_target(np.nan_to_num(execution_signal(z, "30MIN").to_numpy())), index=d30.index)
        desired_by[name] = align_30_to_5(tgt30, d5.index)
        i, o = run(desired_by[name], "T0")
        rows.append((f"基础={name} · T0 立即入场", name, "T0", None, None, i, o))
    base = max(rows, key=lambda r: r[5]["sharpe_atr_now"])[1]
    # 2) 择时
    for k in (0.5, 1.0):
        for W in (6, 12):
            i, o = run(desired_by[base], "T1", k, W)
            rows.append((f"基础={base} · T1 等待 k={k} W={W}", base, "T1", k, W, i, o))
    i, o = run(desired_by[base], "T2", 1.0, 12)
    rows.append((f"基础={base} · T2 过滤 k=1.0 W=12", base, "T2", 1.0, 12, i, o))

    sel = max(rows, key=lambda r: r[5]["sharpe_atr_now"])
    print(f"5MIN 当前 ATR ≈ ${atr_now:.2f}；样本内选中基础策略：{base}；总体样本内最优：{sel[0]}\n")
    tab = pd.DataFrame([{
        "方案": r[0], "选中": "★" if r is sel else "",
        "夏普内": r[5]["sharpe_atr_now"], "夏普外": r[6]["sharpe_atr_now"],
        "毛利/笔ATR5 内": r[5]["gross_atr_per_trip"], "外": r[6]["gross_atr_per_trip"],
        "成本/笔ATR5": r[6]["cost_atr_per_trip_now"], "次数内": r[5]["trips"], "次数外": r[6]["trips"],
        "美元净利内": r[5]["net_usd"], "美元净利外": r[6]["net_usd"], "去漂移外": r[6]["net_ex_drift_usd"]} for r in rows]).set_index("方案")
    print(tab.to_string())

    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            ratio = r[6]["gross_atr_per_trip"] / r[6]["cost_atr_per_trip_now"] if r[6]["cost_atr_per_trip_now"] else float("nan")
            w.writerow([date.today(), f"30MIN:{r[1]} × 5MIN:IntradayReversalCore", "30MIN→5MIN", "trend-tail + reversal timing",
                        r[2], f"k={r[3]};W={r[4]}", "IS sharpe_atr_now", r[5]["sharpe_atr_now"], r[5]["net_ex_drift_usd"],
                        r[6]["sharpe_atr_now"], r[6]["net_ex_drift_usd"], round(ratio, 2), len(rows),
                        ("SELECTED " if r is sel else "") + r[0]])
    tab.to_csv("reports/entry_timing_results.csv")


if __name__ == "__main__":
    main()
