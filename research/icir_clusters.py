# -*- coding: utf-8 -*-
"""
规则适配前的筛选与去重：
  1) 候选 = 月度 ICIR 的 |t| >= 5，且样本内、样本外 ICIR 同号（信息稳定）
  2) 方向统一为 ICIR > 0（负的取反），在样本内（2015-2019）计算因子值的 Spearman 相关
  3) 按 |相关| >= 0.6 做层次聚类，每簇视为一个独立信号
用法：python -m research.icir_clusters [5MIN 15MIN ...]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from factors.registry import get_factors


def candidates(freq: str, t_min: float = 5.0) -> dict:
    out = {}
    for f in glob.glob("reports/factors/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        m = r["results"].get(freq)
        if not m or m["icir"]["ic_t"] != m["icir"]["ic_t"]:
            continue
        if abs(m["icir"]["ic_t"]) >= t_min and np.sign(m["icir_in"]["icir"]) == np.sign(m["icir_oos"]["icir"]):
            out[r["name"]] = m
    return out


def main(freq: str) -> None:
    cand = candidates(freq)
    df = pd.read_pickle(f"data/cache/{freq}.pkl").loc["2015":"2019"]
    F = pd.DataFrame({n: np.sign(m["icir"]["icir"]) * get_factors([n])[0](df, freq) for n, m in cand.items()}).dropna()
    C = F.rank().corr()
    Z = linkage(squareform(1 - C.abs().values, checks=False), "average")
    lab = fcluster(Z, t=0.4, criterion="distance")
    print(f"\n{'=' * 90}\n{freq}：{len(cand)} 个候选 → {lab.max()} 个独立信号簇（簇内 |相关| ≳ 0.6）")
    for c in sorted(set(lab)):
        names = [n for n, l in zip(C.index, lab) if l == c]
        names.sort(key=lambda n: -abs(cand[n]["icir"]["ic_t"]))
        print(f"\n簇 {c}：")
        for n in names:
            m = cand[n]; b = m["bt_oos"]; e = m["atr_edge"]
            inner = C.loc[n, [x for x in names if x != n]].abs().mean() if len(names) > 1 else float("nan")
            print(f"  {('+' if m['icir']['icir'] > 0 else '−') + n:<26} ICIR {m['icir']['icir']:+.2f} t={m['icir']['ic_t']:+5.1f} "
                  f"(内{m['icir_in']['icir']:+.2f}/外{m['icir_oos']['icir']:+.2f})  簇内平均|ρ| {inner:.2f}  "
                  f"净利外 {b['net']:+6.0f} 夏普 {b['sharpe']:+.2f}  ATR近3年/成本 {e['per_trip_atr_recent3y'] / e['cost_now_atr']:+.1f}  {m['verdict']}")


if __name__ == "__main__":
    for fq in sys.argv[1:] or ["5MIN", "15MIN", "30MIN", "1H"]:
        main(fq)
