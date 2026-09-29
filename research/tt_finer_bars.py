# -*- coding: utf-8 -*-
"""
TT30 的信号用在 15MIN / 5MIN（预登记：reports/tt_finer_bars_prereg.md）。
同样的两个因子等权、迟滞 1.5/0.3、换日前平仓；比较"同样回看 16 小时"与默认窗口。
用法：python -m research.tt_finer_bars
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from factors.core import rolling_mad_zscore, trading_day
from factors.evaluate import SWAP, execution_signal, positions, swap_units
from factors.registry import get_factors
from research.multifactor_combo import sh

SPLIT = pd.Timestamp("2020-01-01")
CFG = {"T30 基准": ("30MIN", 32, 32, 1000), "T15s 同16小时": ("15MIN", 64, 64, 2000), "T05s 同16小时": ("5MIN", 192, 192, 6000),
       "T15d 默认8小时": ("15MIN", 32, 32, 1500), "T05d 默认3小时": ("5MIN", 36, 48, 2000)}


def run(freq, chan, atr_n, norm):
    df = pd.read_pickle(f"data/cache/{freq}.pkl")
    te, vw = get_factors(["TrendEfficiencyVolume", "VWAPDeviation"])
    kw = dict(chan=chan, atr=atr_n, norm=norm)
    z = rolling_mad_zscore(0.5 * te.func(df, freq, **kw).fillna(0) + 0.5 * vw.func(df, freq, **kw).fillna(0), norm)
    pos = positions(execution_signal(z, freq))
    move = df["open"].shift(-1) - df["open"]
    dpos = pos.diff().abs().fillna(pos.abs())
    sw = pos.abs() * swap_units(df.index) * SWAP
    out = {}
    for s in (0.2, 0.3):
        net = (pos * move - dpos * s / 2 - sw).fillna(0)
        out[s] = net
    gross = (pos * move).fillna(0)
    return out, gross, pos


def main():
    rows, daily = [], {}
    for name, (freq, chan, atr_n, norm) in CFG.items():
        nets, gross, pos = run(freq, chan, atr_n, norm)
        td = trading_day(pos.index)
        trips = pos.diff().abs() / 2
        r = {"配置": name, "频率": freq}
        for seg, m in (("内", pos.index < SPLIT), ("外", pos.index >= SPLIT)):
            yrs = 11 if seg == "内" else 6.75
            r[f"笔/年{seg}"] = round(float(trips[m].sum()) / yrs)
            r[f"每笔毛利{seg}$"] = round(float(gross[m].sum() / max(trips[m].sum(), 1)), 2)
            for s in (0.2, 0.3):
                d = nets[s][m].groupby(td[m]).sum()
                r[f"夏普{seg}@{s}"] = round(sh(d), 2)
                r[f"美元{seg}@{s}"] = round(float(nets[s][m].sum()))
            r[f"多{seg}$"] = round(float(nets[0.3][m & (pos > 0)].sum()))
            r[f"空{seg}$"] = round(float(nets[0.3][m & (pos < 0)].sum()))
        daily[name] = nets[0.3].groupby(td).sum()
        rows.append(r)
    T = pd.DataFrame(rows).set_index("配置")
    D = pd.DataFrame(daily).fillna(0)
    T["与TT30相关"] = D.corr()["T30 基准"].round(2)
    pd.set_option("display.width", 250)
    for seg in ("内", "外"):
        cols = [c for c in T.columns if c.endswith(seg) or f"{seg}@" in c or f"{seg}$" in c]
        print(f"\n样本{'内 2009-2019' if seg == '内' else '外 2020-2026.09'}：\n" + T[cols].to_string())
    print("\n与 TT30（30MIN）日度相关：", T["与TT30相关"].to_dict())
    base = T.loc["T30 基准"]
    print("\n判定（点差 0.3$）：")
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for k in ("T15s 同16小时", "T05s 同16小时", "T15d 默认8小时", "T05d 默认3小时"):
            x = T.loc[k]
            c = {"样本内夏普高于 T30": x["夏普内@0.3"] > base["夏普内@0.3"], "样本外夏普不低于 T30": x["夏普外@0.3"] >= base["夏普外@0.3"],
                 "样本外多空都为正": x["多外$"] > 0 and x["空外$"] > 0}
            print(f"  {k}: " + "  ".join(f"{'✅' if v else '❌'}{kk}" for kk, v in c.items()) + f" → {'更好' if all(c.values()) else '不如 30MIN'}")
            w.writerow(["2026-09-29", "TT 信号换细K线", x["频率"], "trend-tail", "hyst", k, "IS usd sharpe @0.3 vs T30",
                        x["夏普内@0.3"], x["美元内@0.3"], x["夏普外@0.3"], x["美元外@0.3"], "", 4,
                        f"T30 内/外 {base['夏普内@0.3']}/{base['夏普外@0.3']}；{'更好' if all(c.values()) else '不如30MIN'}"])


if __name__ == "__main__":
    main()
