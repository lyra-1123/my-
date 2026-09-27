# -*- coding: utf-8 -*-
"""
30MIN 趋势尾部等权组合：补扫其余参数（逐一变动，其余保持默认；稳健性检验，不选参）。
默认：chan=32, entry=1.5, exit=0.3, norm=1000, atr=32, vol_base=20, 权重 TrendEff:VWAPDev = 0.5:0.5
"""
import numpy as np
import pandas as pd

from factors.core import rolling_mad_zscore
from factors.evaluate import execution_signal, positions
from factors.registry import get_factors
from research.trend_tail_validation import FREQ, INS, df, pnl, sharpe

TE = get_factors(["TrendEfficiencyVolume"])[0].func
VW = get_factors(["VWAPDeviation"])[0].func


def combo(w=0.5, norm=1000, **kw):
    raw = w * TE(df, FREQ, norm=norm, **kw).fillna(0) + (1 - w) * VW(df, FREQ, norm=norm, **kw).fillna(0)
    return rolling_mad_zscore(raw, norm)


def row(tag, z):
    pa, pu = pnl(positions(execution_signal(z, FREQ), 1.5, 0.3))
    y = pa.groupby(df.index.year).sum()
    return {"参数": tag, "夏普内": round(sharpe(pa[INS]), 2), "夏普外": round(sharpe(pa[~INS]), 2),
            "美元内": round(pu[INS].sum()), "美元外": round(pu[~INS].sum()), "年度为正": f"{(y[y.index > 2008] > 0).sum()}/{(y.index > 2008).sum()}"}


rows = [row("默认", combo())]
rows += [row(f"norm={n}", combo(norm=n)) for n in (500, 2000, 4000)]
rows += [row(f"atr={a}", combo(atr=a)) for a in (16, 64)]
rows += [row(f"vol_base={v}", combo(vol_base=v)) for v in (10, 40)]
rows += [row(f"权重 TE={w}", combo(w=w)) for w in (0.0, 0.25, 0.75, 1.0)]
print(pd.DataFrame(rows).set_index("参数").to_string())
