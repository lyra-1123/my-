# -*- coding: utf-8 -*-
"""
候选策略与在跑策略的相关性和组合贡献（模拟盘准入的预检查，不登记）。
用法：python -m research.candidate_vs_live 因子名:频率 [因子名:频率 ...]
候选按统一规则（1.5/0.3 + 换日前平仓），收益口径与模拟盘引擎一致（R = 净收益 / 上一根 ATR，按交易日汇总）。
"""
import sys

import numpy as np
import pandas as pd

from paper.engine import compute, daily_from, load_bars
from paper.specs import SPECS, StrategySpec


def daily_R(spec, cache={}):
    if spec.freq not in cache:
        cache[spec.freq] = load_bars("data", spec.freq)[0]
    return daily_from(compute(spec, cache[spec.freq]))["net_R"]


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def main(args):
    live = {s.id: daily_R(s) for s in SPECS if s.status == "active"}
    cands = {}
    for a in args:
        name, freq = a.split(":")
        spec = StrategySpec(id=f"{name}@{freq}", description="候选", freq=freq, components=((name, 1.0, ()),), norm=1000,
                            forward_start="2099-01-01")
        cands[spec.id] = daily_R(spec)
    D = pd.DataFrame({**live, **cands}).fillna(0.0)
    for seg, m in (("全样本", slice(None)), ("2020-", slice("2020-01-01", None))):
        X = D.loc[m]
        print(f"\n== {seg}：日度 R 相关")
        print(X.corr().round(2).to_string())
        L = X[list(live)].sum(axis=1)
        print(f"   在跑组合夏普 {sharpe(L):.2f}")
        for c in cands:
            print(f"   + {c:<32} 单独夏普 {sharpe(X[c]):+.2f}；加入后组合夏普 {sharpe(L + X[c]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
