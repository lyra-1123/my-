#!/usr/bin/env python3
"""
compare_histdata_vs_dukascopy.py
==================================
交叉验证：同一年份，原始 HistData 文件（DAT_ASCII_XAUUSD_M1_{year}.csv）
与新导出的 Dukascopy 文件（DAT_ASCII_XAUUSD_M1_{year}_DUKAREAL.csv）
在重叠时间戳上的价格差异，判断新数据是否可信、是否存在时区或时间格式问题。

只对 2009-2016 有意义（两个来源都覆盖到的年份）。

用法：
    python 01_数据提取/compare_histdata_vs_dukascopy.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402
from goldq.dukascopy_io import read_dukascopy_csv  # noqa: E402


def compare_year(raw_dir: Path, year: int) -> None:
    hist_path = raw_dir / f"DAT_ASCII_XAUUSD_M1_{year}.csv"
    duka_path = raw_dir / f"DAT_ASCII_XAUUSD_M1_{year}_DUKAREAL.csv"

    if not hist_path.exists() or not duka_path.exists():
        print(f"[{year}] 缺文件（需要 {hist_path.name} 和 {duka_path.name} 同时存在），跳过")
        return

    hist_df = read_dukascopy_csv(hist_path)
    duka_df = read_dukascopy_csv(duka_path)

    merged = pd.merge(
        hist_df[["time_utc", "close"]].rename(columns={"close": "close_hist"}),
        duka_df[["time_utc", "close"]].rename(columns={"close": "close_duka"}),
        on="time_utc",
        how="inner",
    )

    if len(merged) == 0:
        print(f"[{year}] 没有任何时间戳重叠，方法本身可能时区不对齐或时间格式有问题")
        return

    merged["diff"] = (merged["close_duka"] - merged["close_hist"]).abs()
    merged["diff_pct"] = merged["diff"] / merged["close_hist"] * 100

    overlap_ratio = len(merged) / len(hist_df) * 100

    print(f"\n[{year}] 原文件{len(hist_df)}行, 新文件{len(duka_df)}行, "
          f"重叠时间戳{len(merged)}个 (占原文件{overlap_ratio:.1f}%)")
    print(f"  价格差异: 均值={merged['diff'].mean():.4f}, "
          f"中位数={merged['diff'].median():.4f}, "
          f"最大={merged['diff'].max():.4f}, "
          f"均值百分比={merged['diff_pct'].mean():.4f}%")

    worst = merged.nlargest(5, "diff")
    print("  差异最大的5个时间点:")
    print(worst.to_string(index=False))

    new_only = pd.merge(duka_df[["time_utc"]], hist_df[["time_utc"]], on="time_utc", how="left", indicator=True)
    new_only_count = (new_only["_merge"] == "left_only").sum()
    print(f"  新文件里原文件没有的时间戳数: {new_only_count} (这些是补充的数据，不参与差异比较)")


def main() -> None:
    cfg = load_data_paths()
    raw_dir = Path(cfg["raw_dir"])
    for year in range(2009, 2017):
        compare_year(raw_dir, year)


if __name__ == "__main__":
    main()
