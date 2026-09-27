# -*- coding: utf-8 -*-
"""
组合权重对比（权重只用样本内 2009-2019 的月度 IC 估计，样本外直接套用）：
  等权 / 最大 ICIR 权重 w ∝ Σ_IC^{-1}·mean(IC)（考虑信号 IC 的相关性）/ 只保留残差 ICIR 显著的成分
用法：python -m research.combo_weights [5MIN 15MIN ...]
"""
import sys

import numpy as np
import pandas as pd

from factors.core import params, rolling_mad_zscore
from factors.evaluate import HORIZONS, forward_return, ic_stats
from research.factor_correlation import SPLIT, factor_values, label, monthly_ic, representatives


def main(freq: str) -> None:
    df = pd.read_pickle(f"data/cache/{freq}.pkl")
    reps = representatives(freq)
    F = factor_values(freq, list(reps)).mul(pd.Series(reps))
    F.columns = [label(n, reps[n]) for n in reps]
    y = forward_return(df, HORIZONS[freq][1])
    ins, oos = F.index < SPLIT, F.index >= SPLIT
    ic_in = pd.DataFrame({c: monthly_ic(F.loc[ins, c], y[ins]) for c in F}).dropna()

    raw = np.linalg.solve(ic_in.cov().to_numpy(), ic_in.mean().to_numpy())
    w_opt = pd.Series(raw / np.abs(raw).sum(), index=F.columns)
    core = [c for c in F if "VWAPDev" in c or "VWCT" in c]

    def evaluate(tag, w):
        z = rolling_mad_zscore((F.fillna(0.0) * w).sum(axis=1), params(freq)["norm"])
        a, b = ic_stats(z, y, freq, ins), ic_stats(z, y, freq, oos)
        return {"组合": tag, "ICIR内": a["icir"], "ICIR外": b["icir"], "t外": b["ic_t"], "IC均值外": b["ic_mean"]}

    rows = [evaluate("等权(6)", pd.Series(1 / len(F), index=F.columns)),
            evaluate("最大ICIR权重(6)", w_opt),
            evaluate("核心2个等权(VWAPDev+VWCT)", pd.Series({c: 0.5 if c in core else 0.0 for c in F}))]
    best = max(F.columns, key=lambda c: ic_stats(F[c], y, freq, ins)["icir"])
    b = ic_stats(F[best], y, freq, oos)
    rows.append({"组合": f"最好单因子 {best}", "ICIR内": ic_stats(F[best], y, freq, ins)["icir"],
                 "ICIR外": b["icir"], "t外": b["ic_t"], "IC均值外": b["ic_mean"]})
    print(f"\n{freq}  最大 ICIR 权重（样本内估计）：{w_opt.round(2).to_dict()}")
    print(pd.DataFrame(rows).set_index("组合").to_string())


if __name__ == "__main__":
    for fq in sys.argv[1:] or ["5MIN", "15MIN"]:
        main(fq)
