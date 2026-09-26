#!/usr/bin/env python3
"""
validate_data.py
=================
数据自检 SOP（对应指南 2.2 / 3.x）：
  1. 日均 bars/天连续性检查——识别"占位记录"式的假历史（如 22 条/月）
  2. 重复时间戳检查
  3. 交易时段内的异常缺口检查（> 30 分钟且非周末）
  4. 跨年小时分布检查——用于发现 Trap 类"某几年时间戳偷偷差了几小时未做 UTC 校准"的问题
     （本项目历史数据来自 dukascopy_python，2017 年后另有 mt5_export_direct.py 的兜底导出，
     两者时区约定不同，混用前必须核对，见 lib/goldq/dukascopy_io.py 顶部说明）

用法：
    python 01_数据提取/validate_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.config import load_data_paths  # noqa: E402

GAP_ALERT_MINUTES = 30


def check_duplicates(df: pd.DataFrame) -> int:
    n_dup = df["time_utc"].duplicated().sum()
    print(f"[重复时间戳] {n_dup} 条" + ("  ✅" if n_dup == 0 else "  ❌ 需排查 build_dataset 去重逻辑"))
    return n_dup


def check_daily_bar_count(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df["time_utc"].dt.date).size().rename("bar_count").reset_index()
    daily["weekday"] = pd.to_datetime(daily["time_utc"]).dt.weekday  # 0=Mon ... 6=Sun

    # XAUUSD 现货/CFD 市场：周一到周四应接近 1440 根 M1；周五提前收盘、周日晚开盘，天然更少；周六应为 0
    suspicious = daily[(daily["weekday"] < 4) & (daily["bar_count"] < 1000)]
    print(f"\n[日均 bars] 共 {len(daily)} 个自然日")
    print(daily["bar_count"].describe().to_string())
    if not suspicious.empty:
        print(f"\n[⚠️ 疑似占位记录/缺数据] 以下 {len(suspicious)} 个交易日（周一~周四）bar 数明显偏低（<1000）：")
        print(suspicious.head(20).to_string(index=False))
    else:
        print("\n[日均 bars] 未发现明显异常低量交易日  ✅")
    return daily


def check_gaps(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("time_utc").copy()
    d["gap_min"] = d["time_utc"].diff().dt.total_seconds() / 60
    d["weekday"] = d["time_utc"].dt.weekday
    # 排除周末收盘的正常缺口（周五 21:00 UTC 后 ~ 周日 22:00 UTC 前）
    is_weekend_gap = (d["weekday"] == 4) | (d["weekday"] == 5) | (d["weekday"] == 6)
    anomalies = d[(d["gap_min"] > GAP_ALERT_MINUTES) & (~is_weekend_gap)]
    print(f"\n[交易时段异常缺口 > {GAP_ALERT_MINUTES} 分钟] {len(anomalies)} 处")
    if not anomalies.empty:
        print(anomalies[["time_utc", "gap_min"]].head(20).to_string(index=False))
    return anomalies


def check_hourly_distribution_by_year(df: pd.DataFrame) -> None:
    d = df.copy()
    d["year"] = d["time_utc"].dt.year
    d["hour"] = d["time_utc"].dt.hour
    pivot = d.groupby(["year", "hour"]).size().unstack(fill_value=0)
    # 每年最活跃的小时应该在相近的 UTC 时段（黄金全球交易，分布应逐年稳定）
    peak_hour_by_year = pivot.idxmax(axis=1)
    print("\n[跨年小时分布 sanity check] 各年成交量最集中的 UTC 小时：")
    print(peak_hour_by_year.to_string())
    if peak_hour_by_year.nunique() > 3:
        print("[⚠️] 各年 peak hour 分布跨度较大，可能存在某几年时间戳未做 UTC 校准（见 dukascopy_io.py 说明），建议人工抽查")
    else:
        print("[✅] 各年 peak hour 接近，未见明显时区错位迹象")


def main() -> None:
    df = fetch_bars()
    cfg = load_data_paths()
    print(f"[数据集] {cfg['symbol']} {cfg['timeframe_base']}，共 {len(df)} 行")
    print(f"[数据集] 时间范围: {df['time_utc'].min()} ~ {df['time_utc'].max()}\n")

    check_duplicates(df)
    check_daily_bar_count(df)
    check_gaps(df)
    check_hourly_distribution_by_year(df)


if __name__ == "__main__":
    main()
