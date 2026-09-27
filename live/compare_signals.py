# -*- coding: utf-8 -*-
"""
上线前第 2 步：python -m live.compare_signals --dukascopy-dir 路径 [--days 60]
在两者都有数据的最近 N 天里，比较"用 MT5 数据算出的信号"与"用 Dukascopy 数据算出的信号"（与模拟盘相同的代码）：
  z 的相关、目标仓位一致率、各自的换仓次数与模型盈亏。
判定（事先写死）：目标仓位一致率 ≥ 90% 且 z 相关 ≥ 0.90 → 可以用 MT5 数据实时交易；否则该策略不宜直接上线。
"""
import argparse

import numpy as np
import pandas as pd

from live import config as C
from live.mt5_data import connect, fetch_bars
from paper.engine import compute
from paper.specs import get_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dukascopy-dir", required=True)
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()
    from factors.data_loader import load_m1, resample_ohlcv
    m1 = load_m1(args.dukascopy_dir)
    connect()
    rows = []
    for sid, cfg in C.STRATEGIES.items():
        if not cfg["enabled"]:
            continue
        spec = get_spec(sid)
        cm = compute(spec, fetch_bars(spec.freq, C.HISTORY_BARS[spec.freq]))
        cd = compute(spec, resample_ohlcv(m1, spec.freq))
        end = min(cm.index[-1], cd.index[-1]); start = end - pd.Timedelta(days=args.days)
        a, b = cm.loc[start:end], cd.loc[start:end]
        j = a.index.intersection(b.index)
        agree = float((a.loc[j, "target"] == b.loc[j, "target"]).mean())
        zc = float(np.corrcoef(a.loc[j, "z"].fillna(0), b.loc[j, "z"].fillna(0))[0, 1])
        ok = agree >= 0.90 and zc >= 0.90
        rows.append({"策略": sid, "重叠 K 线": len(j), "z 相关": round(zc, 3), "目标仓位一致率": round(agree, 3),
                     "换仓次数 MT5/Duka": f"{int(a.loc[j, 'target'].diff().abs().gt(0).sum())}/{int(b.loc[j, 'target'].diff().abs().gt(0).sum())}",
                     "模型美元 MT5/Duka": f"{a.loc[j, 'net'].sum():+.1f}/{b.loc[j, 'net'].sum():+.1f}", "判定": "可上线" if ok else "不宜直接上线"})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
