# -*- coding: utf-8 -*-
"""
影子版本 HA1H-TS2（按状态回补 + 移动止损 k×日 ATR）的补充过拟合检验。
注意：该规则是看过样本外后、从 11 个出场规则中挑出的最好者，下列检验只能量化这层挑选偏差，不能消除它。
  1) 移动止损倍数平原：k ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}（事后补测，只作参考）
  2) DSR：N 分别取 出场规则数 11、研究层 N_eff + 11、研究层 N_eff + 全部登记试验
"""
import numpy as np
import pandas as pd

from research.ha1h_exit_rules import evaluate, simulate
from research.overfitting_tests import dsr, matrix_B, n_eff

rows, daily = [], {}
for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    d, m = evaluate(simulate("state", 0.3, trail=k))
    daily[k] = d
    rows.append({"k(日ATR)": k, "夏普内": m["夏普内"], "夏普外": m["夏普外"], "美元内": m["美元内"], "美元外": m["美元外"], "回撤外$": m["回撤外$"]})
d0, m0 = evaluate(simulate("strict", 0.3))
rows.append({"k(日ATR)": "v1 现行", "夏普内": m0["夏普内"], "夏普外": m0["夏普外"], "美元内": m0["美元内"], "美元外": m0["美元外"], "回撤外$": m0["回撤外$"]})
print("[1] 移动止损倍数平原（按状态回补，exit=0.3）\n" + pd.DataFrame(rows).set_index("k(日ATR)").to_string())

B = matrix_B()
sel = daily[2.0].reindex(B.index).fillna(0.0)
sr_B = (B.mean() / B.std()).to_numpy()
NB = n_eff(B)
n_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
print(f"\n[2] DSR（影子版本 k=2，日度 ATR·今 收益，全样本）")
for tag, N in (("只算出场规则 N=11", 11), (f"研究层 N_eff({NB:.1f}) + 11", NB + 11), (f"研究层 N_eff + 全部登记试验 {n_logged}", NB + n_logged)):
    r = dsr(sel, sr_B, N)
    print(f"   {tag:<34} SR={r['SR_annual']:.2f}  SR0={r['SR0_annual']:.2f}  DSR={r['DSR']:.3f}  PSR={r['PSR(SR0=0)']:.4f}")
sel1 = d0.reindex(B.index).fillna(0.0)
r1 = dsr(sel1, sr_B, NB)
print(f"   （对照 HA1H-v1，N=研究层 N_eff：SR={r1['SR_annual']:.2f} DSR={r1['DSR']:.3f}）")
