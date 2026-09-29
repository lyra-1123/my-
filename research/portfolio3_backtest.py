# -*- coding: utf-8 -*-
"""三策略组合（TT30-EW-v1 + HA1H-v1 + 候选 WK1H-v1）回测汇总：各 1 盎司，点差 0.3$，美元与 ATR·今 口径。
用法：python -m research.portfolio3_backtest  → reports/portfolio3/*.csv"""
import numpy as np
import pandas as pd

from factors.core import trading_day
from paper.specs import get_spec
from research.failed_factor_combo import daily_r, pos_s0
from research.multifactor_combo import Book, sh

SPLIT = pd.Timestamp("2020-01-01")
IDS = {"TT30-EW-v1": "TT30", "HA1H-v1": "HA1H", "WK1H-v1": "WK1H"}


def main():
    U, R, N = {}, {}, {}
    for sid, tag in IDS.items():
        s = get_spec(sid)
        B = Book(s.freq)
        pos = pos_s0(s.signal(B.df), s.freq, s.entry, s.exit)
        dp = pos.diff().abs().fillna(pos.abs())
        u = (pos * B.move - dp * 0.15 - pos.abs() * B.swu * 0.47).fillna(0)
        U[tag] = u.groupby(trading_day(u.index)).sum()
        R[tag] = daily_r(B, pos, 1.5)["r"]
        N[tag] = dp.groupby(pos.index.year).sum() / 2
    U, R = pd.DataFrame(U).fillna(0), pd.DataFrame(R).fillna(0)
    U["TT30+HA1H"], U["三策略"] = U.TT30 + U.HA1H, U.TT30 + U.HA1H + U.WK1H
    R["TT30+HA1H"], R["三策略"] = R.TT30 + R.HA1H, R.TT30 + R.HA1H + R.WK1H
    rows = []
    for c in U.columns:
        r = {"组合": c}
        for seg, m in (("2009-19", U.index < SPLIT), ("2020-26", U.index >= SPLIT), ("全期", U.index == U.index)):
            u = U.loc[m, c]
            eq = u.cumsum()
            act = u[u != 0]
            r[f"{seg} 美元"] = round(u.sum())
            r[f"{seg} 夏普(ATR今)"] = round(sh(R.loc[R.index.isin(u.index), c]), 2)
            r[f"{seg} 最大回撤$"] = round((eq - eq.cummax()).min())
            r[f"{seg} 盈利天数"] = f"{(act > 0).mean():.0%}"
        rows.append(r)
    T = pd.DataFrame(rows).set_index("组合")
    Y = U.groupby(U.index.year).sum().round(0).astype(int)
    C = R[R.index >= SPLIT][["TT30", "HA1H", "WK1H"]].corr().round(2)
    T.to_csv("reports/portfolio3/summary.csv")
    Y.to_csv("reports/portfolio3/yearly_usd.csv")
    U.cumsum().to_csv("reports/portfolio3/equity_usd.csv")
    pd.set_option("display.width", 250)
    for seg in ("2009-19", "2020-26", "全期"):
        print(f"\n【{seg}】\n" + T[[c for c in T.columns if c.startswith(seg)]].to_string())
    print("\n逐年美元（各 1 盎司）：\n" + Y.T.to_string())
    print("\n2020 年后日度相关（ATR·今）：\n" + C.to_string())
    lose = U["三策略"][U.index >= SPLIT]
    wk = lose.groupby(pd.Grouper(freq="W")).sum()
    mo = lose.groupby(pd.Grouper(freq="ME")).sum()
    print(f"\n三策略 2020 年后：盈利周占比 {(wk[wk != 0] > 0).mean():.0%}，盈利月占比 {(mo[mo != 0] > 0).mean():.0%}，"
          f"单日最大亏损 {lose.min():.0f}$，单日最大盈利 {lose.max():.0f}$")
    print("每年交易笔数（2020 后平均）：", {k: round(v[v.index >= 2020].mean()) for k, v in N.items()})


if __name__ == "__main__":
    main()
