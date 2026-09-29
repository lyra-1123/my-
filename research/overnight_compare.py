# -*- coding: utf-8 -*-
"""三策略：换日前平仓（现状）vs 持仓过夜（不置 0，照付过夜费）。点差 0.3$，过夜费 0.47$/晚（周三×3）。
用法：python -m research.overnight_compare"""
import numpy as np
import pandas as pd

from factors.core import trading_day
from factors.evaluate import SWAP, execution_signal, positions, swap_units
from paper.specs import get_spec
from research.failed_factor_combo import daily_r
from research.multifactor_combo import Book, sh

SPLIT = pd.Timestamp("2020-01-01")


def run(B, pos):
    dp = pos.diff().abs().fillna(pos.abs())
    swc = pos.abs() * B.swu * SWAP
    gross = (pos * B.move).fillna(0)
    net = gross - dp * 0.15 - swc
    return gross, net.fillna(0), swc, dp


def main():
    rows = []
    for sid in ("TT30-EW-v1", "HA1H-v1", "WK1H-v1"):
        s = get_spec(sid)
        B = Book(s.freq)
        z = s.signal(B.df)
        ny = B.df.index.tz_localize("UTC").tz_convert("America/New_York")
        window = np.asarray((ny.hour >= 16) & (ny.hour < 19))            # 现状下空仓的 16:00~19:00
        for tag, p in (("换日前平仓（现状）", positions(execution_signal(z, s.freq), s.entry, s.exit)),
                       ("持仓过夜", positions(z, s.entry, s.exit))):
            g, n, swc, dp = run(B, p)
            r = daily_r(B, p, 1.5)["r"]
            row = {"策略": sid.split("-")[0], "方式": tag}
            for seg, m in (("内", B.df.index < SPLIT), ("外", B.df.index >= SPLIT)):
                row[f"美元{seg}"] = round(n[m].sum())
                row[f"夏普{seg}"] = round(sh(r[r.index < SPLIT] if seg == "内" else r[r.index >= SPLIT]), 2)
                row[f"过夜费{seg}"] = round(swc[m].sum())
                row[f"点差{seg}"] = round((dp[m] * 0.15).sum())
            row["16-19点毛利外"] = round(g[(B.df.index >= SPLIT) & window].sum())
            row["笔/年外"] = round(dp[B.df.index >= SPLIT].sum() / 2 / 6.75)
            rows.append(row)
    T = pd.DataFrame(rows).set_index(["策略", "方式"])
    pd.set_option("display.width", 250)
    print(T.to_string())
    # 每笔期望 vs 一晚过夜费
    print("\n一晚过夜费 0.47$（周三 1.41$），平均每晚约 0.47×(1+2/5)=0.66$")


if __name__ == "__main__":
    main()
