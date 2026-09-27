# -*- coding: utf-8 -*-
"""
先背景、后形态（Al Brooks 市场周期）—— 预登记事件研究。

背景（形态确认时刻 t，用 [t−59, t] 共 60 根，只用已发生数据）：
  趋势/通道：ER60 = |C_t − C_{t−60}| / Σ|ΔC| ≥ 0.30，且 EMA20(t) − EMA20(t−10) 与净位移同号
  交易区间：ER60 ≤ 0.15，且最近 20 根相邻 K 线重叠度均值 ≥ 0.5（重叠 = 两根交集 / 两根并集）
  其余为过渡，不参与
假设（写在运行之前）：
  A H2/L2：趋势 > 区间          B 三推楔形反转：区间 > 0 且 区间 > 趋势
  C 旗形突破：趋势 > 区间        D 无旗杆区间突破：区间 < 0，趋势 > 区间
形态定义沿用第十一、十二批，一个参数都不改；持有期 h ∈ {12, 24}（前两批使用过的）。
"""
import numpy as np
import pandas as pd

from factors.core import atr, params
from factors.library.price_action import wedge_events
from factors.library.price_action_cont import flag_events, second_entry_events

SPLIT = pd.Timestamp("2020-01-01")
PATTERNS = {"A H2/L2": lambda d, a: second_entry_events(d, a, 2),
            "B 三推楔形反转": lambda d, a: wedge_events(d, a, diminishing=True),
            "C 旗形突破": lambda d, a: flag_events(d, a, True),
            "D 无旗杆区间突破": lambda d, a: flag_events(d, a, False)}
HYP = {"A H2/L2": "趋势>区间", "B 三推楔形反转": "区间>0 且 区间>趋势", "C 旗形突破": "趋势>区间", "D 无旗杆区间突破": "区间<0 且 趋势>区间"}


def context(d: pd.DataFrame) -> np.ndarray:
    c = d["close"]
    er = (c - c.shift(60)).abs() / (c.diff().abs().rolling(60, min_periods=60).sum() + 1e-12)
    ema = c.ewm(span=20, adjust=False, min_periods=20).mean()
    same = np.sign(ema - ema.shift(10)) == np.sign(c - c.shift(60))
    inter = (np.minimum(d["high"], d["high"].shift(1)) - np.maximum(d["low"], d["low"].shift(1))).clip(lower=0)
    union = np.maximum(d["high"], d["high"].shift(1)) - np.minimum(d["low"], d["low"].shift(1))
    overlap = (inter / (union + 1e-12)).rolling(20, min_periods=20).mean()
    reg = np.where((er >= 0.30) & same, "趋势", np.where((er <= 0.15) & (overlap >= 0.5), "区间", "过渡"))
    return reg


def tstat(x):
    return float(np.nanmean(x) / (np.nanstd(x) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


rows, diffs = [], []
for fq in ("5MIN", "15MIN", "30MIN", "1H", "4H"):
    d = pd.read_pickle(f"data/cache/{fq}.pkl")
    a = atr(d, params(fq)["atr"]); an = a.to_numpy()
    reg = context(d)
    share = pd.Series(reg).value_counts(normalize=True).round(2).to_dict()
    print(f"{fq} 背景占比（全部 K 线）：{share}")
    for pname, fn in PATTERNS.items():
        ev = fn(d, an)
        t = np.array([e[0] for e in ev]); side = np.array([e[1] for e in ev])
        for h in (12, 24):
            fwd = (d["open"].shift(-(1 + h)) - d["open"].shift(-1)) / a
            ex = (fwd - fwd.groupby(d.index.year).transform("mean")).to_numpy()
            ok = t < len(d) - h - 2
            tt, ss = t[ok], side[ok]
            r = ss * ex[tt]; rg = reg[tt]; ins = d.index[tt] < SPLIT
            for seg, m in (("内", ins), ("外", ~ins)):
                res = {}
                for g in ("趋势", "区间"):
                    x = r[m & (rg == g)]
                    x = x[~np.isnan(x)]
                    res[g] = x
                    rows.append({"频率": fq, "形态": pname, "h": h, "段": seg, "背景": g, "n": len(x),
                                 "收益ATR": round(float(np.mean(x)), 3) if len(x) else np.nan, "t": round(tstat(x), 2)})
                a_, b_ = res["趋势"], res["区间"]
                if len(a_) > 10 and len(b_) > 10:
                    dm = np.mean(a_) - np.mean(b_)
                    se = np.sqrt(np.var(a_) / len(a_) + np.var(b_) / len(b_))
                    diffs.append({"频率": fq, "形态": pname, "h": h, "段": seg, "趋势−区间": round(dm, 3), "t差": round(dm / se, 2),
                                  "n趋势": len(a_), "n区间": len(b_)})

T, Df = pd.DataFrame(rows), pd.DataFrame(diffs)
T.to_csv("reports/context_pattern_cells.csv", index=False); Df.to_csv("reports/context_pattern_diffs.csv", index=False)
for p in PATTERNS:
    x = T[T["形态"] == p].pivot_table(index=["频率", "h"], columns=["背景", "段"], values=["收益ATR", "t"], sort=False)
    y = Df[Df["形态"] == p].pivot_table(index=["频率", "h"], columns="段", values=["趋势−区间", "t差"], sort=False)
    print(f"\n==== {p}   假设：{HYP[p]}\n" + x.to_string() + "\n--- 背景之差（趋势 − 区间）\n" + y.to_string())
