# -*- coding: utf-8 -*-
"""
日内对冲成分扫描（预登记：reports/sizing_hedge_prereg.md 第二部分）。只扫描已有因子库，不新增因子。

筛选：研究层矩阵（统一规则、日度 ATR·今）中 5MIN~1H 的"因子×频率"，
      与在跑组合（TT30-EW-v1 + HA1H-v1，模拟盘口径日度 R）相关 样本内<0 且 样本外<0，自身夏普 样本内>0 且 样本外>0。
通过筛选者：按模拟盘口径（历史真实成本）算加入后组合夏普（全样本 / 2020 年后）与在跑组合亏损日上的平均收益。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.candidate_vs_live import daily_R, sharpe
from research.overfitting_tests import matrix_B
from paper.specs import SPECS, StrategySpec

SPLIT = pd.Timestamp("2020-01-01")
INTRADAY = ("5MIN", "15MIN", "30MIN", "1H")


def main() -> None:
    live = pd.DataFrame({s.id: daily_R(s) for s in SPECS if s.status == "active"}).fillna(0.0)
    L = live.sum(axis=1)
    B = matrix_B()
    B = B[[c for c in B.columns if c.split("|")[1] in INTRADAY and B[c].std() > 0]]
    idx = B.index.intersection(L.index)
    B, Lb = B.loc[idx], L.loc[idx]
    ins = idx < SPLIT
    rows = []
    for c in B.columns:
        x = B[c]
        rows.append({"因子|频率": c, "相关内": round(float(x[ins].corr(Lb[ins])), 3), "相关外": round(float(x[~ins].corr(Lb[~ins])), 3),
                     "夏普内": round(sharpe(x[ins]), 2), "夏普外": round(sharpe(x[~ins]), 2)})
    S = pd.DataFrame(rows).set_index("因子|频率")
    neg = S[(S["相关内"] < 0) & (S["相关外"] < 0)]
    passed = neg[(neg["夏普内"] > 0) & (neg["夏普外"] > 0)]
    pd.set_option("display.width", 250)
    print(f"[1] 日内'因子×频率' {len(S)} 个；与在跑组合样本内外都负相关 {len(neg)} 个；其中自身夏普样本内外都 > 0：{len(passed)} 个")
    print(f"    全部相关分布（样本外）：分位 [5%,50%,95%] = {np.round(S['相关外'].quantile([.05, .5, .95]).to_numpy(), 3)}")
    print("\n    负相关的全部因子（按样本外相关排序）：\n" + neg.sort_values("相关外").to_string())
    if len(passed) == 0:
        print("\n结论：没有因子同时满足负相关与正期望。")
        return
    print("\n[2] 通过筛选者：模拟盘口径加入在跑组合")
    loss = L < 0
    out = []
    for c in passed.index:
        name, freq = c.split("|")
        spec = StrategySpec(id=c, description="对冲候选", freq=freq, components=((name, 1.0, ()),), norm=1000, forward_start="2099-01-01")
        r = daily_R(spec).reindex(L.index).fillna(0.0)
        for seg, m in (("全样本", slice(None)), ("2020-", slice("2020-01-01", None))):
            out.append({"候选": c, "段": seg, "单独夏普": round(sharpe(r.loc[m]), 2), "与组合相关": round(float(r.loc[m].corr(L.loc[m])), 3),
                        "组合夏普": round(sharpe(L.loc[m]), 2), "加入后": round(sharpe(L.loc[m] + r.loc[m]), 2),
                        "组合亏损日上的平均R": round(float(r.loc[m][loss.loc[m]].mean()), 4)})
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
