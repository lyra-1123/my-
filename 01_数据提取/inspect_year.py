#!/usr/bin/env python3
"""
inspect_year.py
================
针对某一年做更细的人工核查：每日 bar 数、每日成交量、按小时的成交量分布。
用于 validate_data.py 发现某年 peak hour 异常时的二次确认——
区分"这年数据本来就稀疏导致指标失真"还是"这年时间戳真的偏移了"。

用法：
    python 01_数据提取/inspect_year.py 2009
    python 01_数据提取/inspect_year.py 2009 2010   # 多年对比
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.datastore import fetch_bars  # noqa: E402


def main() -> None:
    years = [int(y) for y in sys.argv[1:]] or [datetime.now().year]

    for year in years:
        df = fetch_bars(start=f"{year}-01-01", end=f"{year}-12-31 23:59:59")
        print(f"\n{'=' * 60}\n{year} 年\n{'=' * 60}")
        print(f"总行数: {len(df)}, 总成交量: {df['volume'].sum():.2f}, "
              f"日均成交量: {df.groupby(df['time_utc'].dt.date)['volume'].sum().mean():.2f}")

        hourly_volume = df.groupby(df["time_utc"].dt.hour)["volume"].sum()
        print("\n按 UTC 小时的成交量分布:")
        print(hourly_volume.to_string())

        zero_volume_ratio = (df["volume"] == 0).mean() * 100
        print(f"\nvolume=0 的行占比: {zero_volume_ratio:.1f}%")

        sample_day = df[df["time_utc"].dt.date == df["time_utc"].dt.date.iloc[len(df) // 2]]
        print(f"\n中间某一天（{sample_day['time_utc'].dt.date.iloc[0]}）前10条原始数据:")
        print(sample_day.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
