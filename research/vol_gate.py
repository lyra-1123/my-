# -*- coding: utf-8 -*-
"""
波动率门槛检验（预登记）：30MIN 趋势尾部等权组合（chan=32, entry=1.5, exit=0.3，换日前平仓）。

规则：只有 ATR30_t ≥ k × 0.2$ 时允许开新仓 / 反手；已有仓位按原规则出场。
候选：k ∈ {5, 7.5, 10, 12.5, 15}（ATR ≥ 1.0 / 1.5 / 2.0 / 2.5 / 3.0 美元）。
三段：2009-2014 选 k（真实美元日度夏普最高者）→ 2015-2019 验证（低波动期）→ 2020- 观察（高波动期）。
对照：每段内随机屏蔽与门槛相同比例的 K 线（开仓许可），50 次，比较美元净利与夏普的分布。
"""
from __future__ import annotations

import csv
from datetime import date

import numpy as np
import pandas as pd

from factors.core import atr, trading_day
from factors.evaluate import ENTRY, EXIT, SPREAD, SWAP, execution_signal, swap_units
from research.rule_adaptation import TRIALS_CSV
from research.trend_tail_validation import FREQ, a_prev, atr_now, df, factor_z, m_atr, move

KS = (5, 7.5, 10, 12.5, 15)
PERIODS = {"选择 2009-2014": (df.index.year <= 2014),
           "验证 2015-2019": (df.index.year >= 2015) & (df.index.year <= 2019),
           "观察 2020-": (df.index.year >= 2020)}


def gated_target(z: np.ndarray, gate: np.ndarray) -> np.ndarray:
    out = np.zeros(len(z)); pos = 0.0
    for t, zt in enumerate(z):
        s = 1.0 if zt > ENTRY else (-1.0 if zt < -ENTRY else 0.0)
        if s != 0 and s != pos and gate[t]:
            pos = s
        elif abs(zt) < EXIT:
            pos = 0.0
        out[t] = pos
    return out


swu = swap_units(df.index)


def evaluate(pos: pd.Series, mask) -> dict:
    p = pos[mask]
    dpos = p.diff().abs().fillna(p.abs())
    usd = (p * move[mask] - dpos * SPREAD / 2 - p.abs() * swu[mask] * SWAP).fillna(0)
    atrn = (p * m_atr[mask] - dpos * SPREAD / 2 / atr_now - p.abs() * swu[mask] * SWAP / atr_now).fillna(0)
    d = usd.groupby(trading_day(usd.index)).sum()
    da = atrn.groupby(trading_day(atrn.index)).sum()
    eq = usd.cumsum()
    return {"net": float(usd.sum()), "sharpe_usd": float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0,
            "sharpe_atr": float(da.mean() / da.std() * np.sqrt(252)) if da.std() > 0 else 0.0,
            "trips": float(dpos.sum() / 2), "mdd": float((eq - eq.cummax()).min()),
            "in_mkt": float((p != 0).mean())}


def main() -> None:
    z = np.nan_to_num(execution_signal(factor_z("等权", 32), FREQ).to_numpy())
    a_now = atr(df, 32).to_numpy()                      # t 收盘时已知的 ATR
    to_pos = lambda g: pd.Series(gated_target(z, g), index=df.index).shift(1).fillna(0.0)
    base = to_pos(np.ones(len(z), bool))

    res = {}
    for k in KS:
        gate = np.nan_to_num(a_now) >= k * SPREAD
        res[k] = (gate, to_pos(gate))
    sel_mask = PERIODS["选择 2009-2014"]
    k_star = max(KS, key=lambda k: evaluate(res[k][1], sel_mask)["sharpe_usd"])

    print("每段开仓许可被屏蔽的 K 线比例：")
    print(pd.DataFrame({k: {p: round(1 - res[k][0][m].mean(), 2) for p, m in PERIODS.items()} for k in KS}).to_string())

    rows = []
    for name, pos in [("无门槛", base)] + [(f"k={k}" + (" ★" if k == k_star else ""), res[k][1]) for k in KS]:
        r = {"方案": name}
        for p, m in PERIODS.items():
            e = evaluate(pos, m)
            r[f"{p}|净利$"] = round(e["net"]); r[f"{p}|夏普$"] = round(e["sharpe_usd"], 2); r[f"{p}|笔数"] = round(e["trips"])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("方案"); t.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in t.columns])
    print(f"\n样本内（2009-2014）选中 k* = {k_star}\n"); print(t.to_string())

    # 随机对照：每段内按相同比例随机屏蔽
    gate_star = res[k_star][0]
    rng = np.random.default_rng(42)
    sims = {p: [] for p in PERIODS}
    for _ in range(50):
        g = np.ones(len(z), bool)
        for p, m in PERIODS.items():
            idx = np.flatnonzero(m)
            frac = 1 - gate_star[m].mean()
            g[idx[rng.random(len(idx)) < frac]] = False
        pos = to_pos(g)
        for p, m in PERIODS.items():
            sims[p].append(evaluate(pos, m))
    print(f"\n随机对照（k*={k_star} 的逐段屏蔽比例，50 次）：")
    for p, m in PERIODS.items():
        e = evaluate(res[k_star][1], m); b = evaluate(base, m)
        nets = np.array([s["net"] for s in sims[p]]); shs = np.array([s["sharpe_usd"] for s in sims[p]])
        print(f"  {p}: 门槛 净利 {e['net']:+.0f}$ 夏普 {e['sharpe_usd']:+.2f} | 无门槛 {b['net']:+.0f}$ {b['sharpe_usd']:+.2f} | "
              f"随机 净利 中位 {np.median(nets):+.0f}$ [5%~95%: {np.percentile(nets,5):+.0f} ~ {np.percentile(nets,95):+.0f}]，"
              f"夏普 中位 {np.median(shs):+.2f}；门槛优于随机的比例 {np.mean(e['net'] > nets):.2f}")
        print(f"      门槛 回撤 {e['mdd']:+.0f}$ / 无门槛 {b['mdd']:+.0f}$；在场时间 {e['in_mkt']:.2f} / {b['in_mkt']:.2f}；ATR·今夏普 {e['sharpe_atr']:+.2f} / {b['sharpe_atr']:+.2f}")

    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for k in KS:
            s, v, o = (evaluate(res[k][1], m) for m in PERIODS.values())
            w.writerow([date.today(), "30MIN:趋势尾部等权组合", FREQ, "trend-tail", "vol_gate", f"k={k}", "2009-2014 sharpe_usd",
                        round(s["sharpe_usd"], 2), round(s["net"]), round(o["sharpe_usd"], 2), round(o["net"]), "", len(KS),
                        ("SELECTED " if k == k_star else "") + f"验证2015-2019 净利{v['net']:+.0f} 夏普{v['sharpe_usd']:+.2f}"])


if __name__ == "__main__":
    main()
