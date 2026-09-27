# -*- coding: utf-8 -*-
"""
实验：换日前强制平仓（纽约 16:00-18:00 因子置 0 → 迟滞规则平仓且不新开仓），
比较与原始持仓的样本外净利、过夜费、ATR 边际。用于判断"日内平仓"执行规则是否值得纳入因子设计。
"""
import pandas as pd

from factors.evaluate import SPLIT_DEFAULT, atr_edge, backtest
from factors.registry import get_factors

CASES = [("TrendEfficiencyVolume", "30MIN"), ("TSMomentumVolScaled", "30MIN"), ("VWAPDeviation", "30MIN"),
         ("TrendEfficiencyVolume", "1H"), ("VWAPDeviation", "1H"), ("MTFTrendResonance", "30MIN"),
         ("PullbackSwing", "5MIN"), ("TrendPullbackLowVolume", "5MIN"), ("TrendPullbackLowVolume", "15MIN")]


def flat_window(idx: pd.DatetimeIndex) -> pd.Series:
    ny = idx.tz_localize("UTC").tz_convert("America/New_York")
    return pd.Series((ny.hour >= 16) & (ny.hour < 18), index=idx)


if __name__ == "__main__":
    split = pd.Timestamp(SPLIT_DEFAULT)
    rows = []
    for name, fq in CASES:
        df = pd.read_pickle(f"data/cache/{fq}.pkl")
        z = get_factors([name])[0](df, fq)
        oos = df.index >= split
        for tag, zz in (("原始", z), ("换日前平仓", z.where(~flat_window(df.index), 0.0))):
            b = backtest(df[oos], zz[oos]); e = atr_edge(df, zz, fq)
            rows.append({"因子": f"{name} {fq}", "方案": tag, "样本外净利": b["net"], "过夜费": b["swap_cost"],
                         "点差": b["spread_cost"], "次数": b["trips"], "ATR近3年": e["per_trip_atr_recent3y"],
                         "成本ATR": e["cost_now_atr"], "倍数": round(e["per_trip_atr_recent3y"] / e["cost_now_atr"], 2),
                         "年度>0": e["per_trip_atr_year_pos_ratio"]})
    print(pd.DataFrame(rows).set_index(["因子", "方案"]).to_string())
