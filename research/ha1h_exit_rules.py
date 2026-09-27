# -*- coding: utf-8 -*-
"""
HA1H 出场规则检验（预登记；v1 不改，结果作为 v2 候选依据）。

维度一：过夜处理
  strict   现行：纽约 [15:00,18:00) 的 1H K 线 z 视为 0 → 16:00 平仓；19:00 起重新入场须 |z|>1.5
  state    每日照样平仓，但 19:00 起只要迟滞状态仍在（未出现 |z|<exit）就按原方向回补
  overnight 不做每日平仓，持仓过夜并支付过夜费（0.47$/晚，周三 ×3）
维度二：信号出场阈值 exit ∈ {0.3（现行）, 0（z 过零）, 0.6}
附加：state + exit=0.3 叠加移动止损：从本段持仓最有利价格回撤 k×日线 ATR(14)（已收盘日线）出场，k∈{2,3}；
      止损后同方向需等迟滞状态复位（|z|<exit 或反向）才能再入场。盘中按 1H 最高/最低价触发，跳空按开盘价成交。
选择：样本内（2009-2019）ATR·今 夏普最高且优于现行；样本外只跑一次；11 个规则做 CSCV。
"""
from __future__ import annotations

import csv
from datetime import date

import numpy as np
import pandas as pd

from factors.core import atr, htf_feature, params, trading_day
from factors.evaluate import ENTRY, SPREAD, SWAP, swap_units
from factors.registry import get_factors
from research.overfitting_tests import cscv
from research.rule_adaptation import TRIALS_CSV

FREQ = "1H"
SPLIT = pd.Timestamp("2020-01-01")
df = pd.read_pickle(f"data/cache/{FREQ}.pkl")
o, h, l = (df[c].to_numpy() for c in ("open", "high", "low"))
z = np.nan_to_num(get_factors(["HighAnchorMomentum"])[0].func(df, FREQ).to_numpy())
ny = df.index.tz_localize("UTC").tz_convert("America/New_York")
FLAT = np.asarray((ny.hour >= 15) & (ny.hour < 18))            # 与 execution_signal 对 1H 的窗口一致
ATRD = np.nan_to_num(htf_feature(df, FREQ, "1D", lambda b: atr(b, 14)).to_numpy())
a1 = atr(df, params(FREQ)["atr"])
a_prev = a1.shift(1).to_numpy()
atr_now = float(a1[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
SWU = swap_units(df.index).to_numpy()
yr = df.index.year
drift = pd.Series((df["open"].shift(-1) - df["open"]).to_numpy() / a_prev, index=df.index).groupby(yr).transform("mean").to_numpy()


def simulate(mode: str, exit_th: float, trail: float | None = None):
    """返回 pos_open（第 t 根开盘时的持仓）、pnl（价格差）、dpos（仓位变动，计点差）、held_end（收盘后仍持仓，计过夜费）。"""
    n = len(z)
    pos_open = np.zeros(n); pnl = np.zeros(n); dpos = np.zeros(n); held_end = np.zeros(n)
    pos, hs, block, best, lvl = 0.0, 0.0, 0.0, 0.0, np.nan
    for t in range(n):
        pos_open[t] = pos
        nxt = o[t + 1] if t + 1 < n else np.nan
        stopped = False
        if pos != 0 and not np.isnan(lvl):
            if pos > 0 and l[t] <= lvl:
                px = min(o[t], lvl); stopped = True
            elif pos < 0 and h[t] >= lvl:
                px = max(o[t], lvl); stopped = True
        if stopped:
            pnl[t] = pos * (px - o[t])
            if t + 1 < n:
                dpos[t + 1] += abs(pos)
            block, pos = pos, 0.0
        else:
            pnl[t] = pos * (nxt - o[t]) if t + 1 < n else 0.0
            held_end[t] = abs(pos)
            if pos > 0:
                best = max(best, h[t])
            elif pos < 0:
                best = min(best, l[t])
        # 收盘：迟滞状态（不受每日平仓影响）
        zt = z[t]
        if zt > ENTRY:
            hs_new = 1.0
        elif zt < -ENTRY:
            hs_new = -1.0
        elif (exit_th == 0 and hs != 0 and zt * hs <= 0) or (exit_th > 0 and abs(zt) < exit_th):
            hs_new = 0.0
        else:
            hs_new = hs
        if hs_new != hs:
            block = 0.0 if hs_new != block else block
            if hs_new == 0:
                block = 0.0
        episode_start = hs_new != 0 and hs_new != hs
        hs = hs_new
        # 目标仓位
        if mode == "overnight":
            tgt = hs
        elif mode == "state":
            tgt = 0.0 if FLAT[t] else hs
        else:                                                   # strict：窗口内 z 视为 0，重新入场须新的 |z|>1.5
            if FLAT[t]:
                tgt = 0.0
            elif pos == 0:
                tgt = (1.0 if zt > ENTRY else (-1.0 if zt < -ENTRY else 0.0))
            else:
                tgt = hs if hs != 0 else 0.0
        if tgt != 0 and tgt == block:
            tgt = 0.0
        if tgt != pos:
            if t + 1 < n:
                dpos[t + 1] += abs(tgt - pos)
            if tgt != 0 and (pos == 0 or episode_start or np.sign(tgt) != np.sign(pos)):
                if episode_start or best == 0 or np.sign(tgt) != np.sign(pos):
                    best = nxt
            pos = tgt
        if episode_start and pos != 0:
            best = nxt
        lvl = (best - pos * trail * ATRD[t]) if (trail is not None and pos != 0) else np.nan
    return pos_open, pnl, dpos, held_end


def evaluate(res):
    pos_open, pnl, dpos, held_end = res
    sw = held_end * SWU * SWAP
    usd = pd.Series(pnl - dpos * SPREAD / 2 - sw, index=df.index)
    r = pd.Series(np.nan_to_num(pnl / a_prev - pos_open * drift - dpos * SPREAD / 2 / atr_now - sw / atr_now), index=df.index)
    d = r.groupby(trading_day(r.index)).sum()
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")
    ins = df.index < SPLIT
    p = pd.Series(pos_open, index=df.index)
    eq = usd[~ins].cumsum()
    return d, {"夏普内": round(sh(d[d.index < SPLIT]), 2), "夏普外": round(sh(d[d.index >= SPLIT]), 2),
               "美元内": round(usd[ins].sum()), "美元外": round(usd[~ins].sum()),
               "点差外": round(float((dpos[~ins] * SPREAD / 2).sum())), "过夜费外": round(float(sw[~ins].sum())),
               "开平次数外": int(round(dpos[~ins].sum() / 2)), "在场外": round(float((p[~ins] != 0).mean()), 3),
               "回撤外$": round(float((eq - eq.cummax()).min()))}


def main():
    rules = [(f"{m} exit={e}", dict(mode=m, exit_th=e)) for m in ("strict", "state", "overnight") for e in (0.3, 0.0, 0.6)]
    rules += [(f"state exit=0.3 + 移动止损 {k}×日ATR", dict(mode="state", exit_th=0.3, trail=k)) for k in (2, 3)]
    rows, daily = [], {}
    for name, kw in rules:
        d, m = evaluate(simulate(**kw))
        daily[name] = d; rows.append({"规则": name, **m})
    t = pd.DataFrame(rows).set_index("规则")
    base = "strict exit=0.3"
    cand = t[(t.index != base) & (t["夏普内"] > t.loc[base, "夏普内"])]
    sel = cand["夏普内"].idxmax() if len(cand) else base
    t["选中"] = ["★" if i == sel else "" for i in t.index]
    print(t.to_string())
    c = cscv(pd.DataFrame(daily).fillna(0.0))
    print(f"\n样本内选中：{sel}（现行 strict exit=0.3 样本内夏普 {t.loc[base, '夏普内']}）")
    print(f"11 个规则 CSCV：PBO={c['PBO']:.3f}；样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}")
    t.to_csv("reports/ha1h_exit_rules.csv")
    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for name, r in t.iterrows():
            w.writerow([date.today(), "HA1H 出场/过夜规则", FREQ, "high-anchor", "exit", name, "IS sharpe_atr_now (must beat strict 0.3)",
                        r["夏普内"], r["美元内"], r["夏普外"], r["美元外"], "", len(t), ("SELECTED " if name == sel else "") + name])


if __name__ == "__main__":
    main()
