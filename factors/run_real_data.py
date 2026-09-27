# -*- coding: utf-8 -*-
"""
用真实 XAUUSD 数据评估 3 个因子（所有频率）。

用法：
    python factors/run_real_data.py --data-dir data/ [--years 2020 2021 ...] [--split 2022-01-01]

输出：每个频率的 未来函数自检、成本门槛、RankIC（样本内/样本外）、含成本回测（样本外）。
样本内（split 之前）只用来观察，参数不在这里调优；样本外（split 之后）才是可信的检验。
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_m1, resample_ohlcv  # noqa: E402
from xauusd_volume_breakout_factors import (  # noqa: E402
    backtest_with_cost, check_no_lookahead, compute_all_factors,
    cost_hurdle_report, forward_return, rank_ic,
)

HORIZONS = {"5MIN": (1, 6, 24), "15MIN": (1, 4, 16), "30MIN": (1, 4, 12),
            "1H": (1, 4, 12), "4H": (1, 3, 6), "1D": (1, 3, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--split", default="2022-01-01", help="样本内/样本外分割日期")
    ap.add_argument("--freqs", nargs="*", default=list(HORIZONS))
    args = ap.parse_args()

    m1 = load_m1(args.data_dir, args.years)
    print(f"M1: {len(m1):,} 根, {m1.index[0]} ~ {m1.index[-1]}")
    split = pd.Timestamp(args.split)

    for freq in args.freqs:
        df = resample_ohlcv(m1, freq)
        print(f"\n{'=' * 70}\n{freq}: {len(df):,} 根")
        print("  未来函数自检:", check_no_lookahead(df.tail(20000), freq))
        print(f"  成本门槛 spread/ATR: {cost_hurdle_report(df, freq):.3f}")
        fac = compute_all_factors(df, freq)
        ins, oos = fac.index < split, fac.index >= split

        rows = []
        for h in HORIZONS[freq]:
            fwd = forward_return(df, h)
            for c in fac:
                rows.append({"factor": c, "h": h,
                             "IC_in": rank_ic(fac[c][ins], fwd[ins]),
                             "IC_oos": rank_ic(fac[c][oos], fwd[oos])})
        print(pd.DataFrame(rows).pivot(index="factor", columns="h").round(4).to_string())

        print("  样本外含成本回测（0.01 手，点差 0.2 美元）:")
        for c in fac:
            print(f"    {c:<24}", backtest_with_cost(df[oos], fac[c][oos]))


if __name__ == "__main__":
    main()
