# -*- coding: utf-8 -*-
"""第八批候选 C1（HighAnchorMomentum 1H）的研究层过拟合检验：重建因子库全部"因子×频率"矩阵，PBO 与 DSR。"""
import numpy as np
import pandas as pd

from factors.evaluate import execution_signal, positions
from research.candidate_validation import pnl, signal, trading_day
from research.overfitting_tests import cscv, dsr, matrix_B, n_eff

B = matrix_B()
c = cscv(B)
print(f"研究层（因子库全部 因子×频率，含第七、八批）：N={c['N']} N_eff≈{n_eff(B):.1f} PBO={c['PBO']:.3f} "
      f"样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f} 亏损概率 {c['prob_oos_loss']:.3f}")
pa, _ = pnl(positions(execution_signal(signal("C1", 250), "1H"), 1.5, 0.3))
sel = pa.groupby(trading_day(pa.index)).sum()
sel = sel.reindex(B.index).fillna(0.0)
n_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
sr_B = (B.mean() / B.std()).to_numpy()
NB = n_eff(B)
for tag, N in (("研究层 N=N_eff(B)", NB), (f"研究层+登记试验 N=N_eff(B)+{n_logged}", NB + n_logged),
               ("极端保守 N=库内全部+登记试验+150 网格", B.shape[1] + n_logged + 150)):
    r = dsr(sel, sr_B, N)
    print(f"  DSR {tag:<36} N={r['N']:>6}  SR={r['SR_annual']:.2f}  SR0={r['SR0_annual']:.2f}  DSR={r['DSR']:.3f}  (PSR={r['PSR(SR0=0)']:.4f})")
