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

# 全球市场惯例假期（月, 日），落在这些日期附近的低量/缺口是预期的市场休市，不算异常。
# 只覆盖固定日期假期（圣诞/元旦），复活节等浮动假期不在此列，会被算进"疑似异常"里人工看一眼。
HOLIDAY_MONTH_DAYS = {(12, 24), (12, 25), (12, 26), (12, 31), (1, 1), (1, 2)}


def check_duplicates(df: pd.DataFrame) -> int:
    n_dup = df["time_utc"].duplicated().sum()
    print(f"[重复时间戳] {n_dup} 条" + ("  ✅" if n_dup == 0 else "  ❌ 需排查 build_dataset 去重逻辑"))
    return n_dup


def check_daily_bar_count(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df["time_utc"].dt.date).size().rename("bar_count").reset_index()
    daily["time_utc"] = pd.to_datetime(daily["time_utc"])
    daily["weekday"] = daily["time_utc"].dt.weekday  # 0=Mon ... 6=Sun
    daily["is_holiday"] = daily["time_utc"].apply(
        lambda d: (d.month, d.day) in HOLIDAY_MONTH_DAYS
    )

    low = daily[(daily["weekday"] < 4) & (daily["bar_count"] < 1000)]
    holiday_low = low[low["is_holiday"]]
    real_suspicious = low[~low["is_holiday"]]

    print(f"\n[日均 bars] 共 {len(daily)} 个自然日")
    print(daily["bar_count"].describe().to_string())
    print(f"\n[假期低量] {len(holiday_low)} 天（圣诞/元旦附近，市场休市，预期内，非异常）")

    if not real_suspicious.empty:
        print(f"\n[⚠️ 疑似占位记录/缺数据] 以下 {len(real_suspicious)} 个交易日（周一~周四，排除固定假期）bar 数明显偏低（<1000）：")
        print(real_suspicious.head(20).to_string(index=False))
    else:
        print("\n[日均 bars] 排除固定假期后，未发现明显异常低量交易日  ✅")
    return daily


def check_gaps(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("time_utc").copy()
    d["gap_min"] = d["time_utc"].diff().dt.total_seconds() / 60
    d["weekday"] = d["time_utc"].dt.weekday
    d["month_day"] = list(zip(d["time_utc"].dt.month, d["time_utc"].dt.day))
    is_weekend_gap = d["weekday"].isin([4, 5, 6])
    is_holiday_gap = d["month_day"].isin(HOLIDAY_MONTH_DAYS)
    anomalies = d[(d["gap_min"] > GAP_ALERT_MINUTES) & (~is_weekend_gap) & (~is_holiday_gap)]

    print(f"\n[交易时段异常缺口 > {GAP_ALERT_MINUTES} 分钟，排除周末/固定假期] {len(anomalies)} 处")
    if anomalies.empty:
        return anomalies

    # 如果这些缺口集中在同一个 UTC 小时附近，大概率是 broker 每日固定的 rollover 停牌，不是数据丢失
    anomalies = anomalies.copy()
    anomalies["gap_start_hour"] = anomalies["time_utc"].dt.hour
    hour_hist = anomalies["gap_start_hour"].value_counts().sort_index()
    print("\n[缺口发生的 UTC 小时分布]（若高度集中在 1-2 个小时，通常是每日固定停牌/rollover，非数据问题）：")
    print(hour_hist.to_string())

    dominant_hour, dominant_count = hour_hist.idxmax(), hour_hist.max()
    if dominant_count / len(anomalies) > 0.5:
        print(f"[信息] {dominant_count}/{len(anomalies)} ({dominant_count/len(anomalies)*100:.0f}%) 的缺口都发生在 "
              f"UTC {dominant_hour} 点附近，符合 broker 每日固定停牌模式，大概率不是数据丢失")

    large_gaps = anomalies[anomalies["gap_min"] > 180].sort_values("gap_min", ascending=False)
    print(f"\n[需要人工看一眼的大缺口 > 3 小时] {len(large_gaps)} 处（可能是浮动假期/黑天鹅/broker 故障）：")
    if not large_gaps.empty:
        print(large_gaps[["time_utc", "gap_min"]].head(20).to_string(index=False))
    return anomalies


def check_hourly_distribution_by_year(df: pd.DataFrame) -> None:
    """
    用成交量（而不是 bar 数）按小时聚合来看 session 分布。
    M1 K 线本身是按时间切的，活跃小时和清淡小时的 bar 数几乎都接近 60，
    用 bar 数选 peak hour 只是在几个 tie 里随便选一个，噪音很大、没有实际意义；
    用 volume 才能真正反映"伦敦/纽约重叠时段更活跃"这个应该逐年稳定的模式。
    """
    d = df.copy()
    d["year"] = d["time_utc"].dt.year
    d["hour"] = d["time_utc"].dt.hour
    pivot = d.groupby(["year", "hour"])["volume"].sum().unstack(fill_value=0)

    zero_volume_years = pivot.index[pivot.sum(axis=1) == 0].tolist()
    if zero_volume_years:
        print(f"\n[跨年小时分布 sanity check] 以下年份 volume 全为 0，跳过该项检查，改用人工核对: {zero_volume_years}")
        pivot = pivot.drop(index=zero_volume_years)
        if pivot.empty:
            return

    peak_hour_by_year = pivot.idxmax(axis=1)
    print("\n[跨年小时分布 sanity check] 各年成交量(volume)最集中的 UTC 小时：")
    print(peak_hour_by_year.to_string())

    # 允许 peak hour 有 ±2 小时的自然波动（流动性结构逐年略有变化），超出这个范围才提示人工核对
    mode_hour = peak_hour_by_year.mode().iloc[0]
    outliers = peak_hour_by_year[(peak_hour_by_year - mode_hour).abs() > 2]
    if not outliers.empty:
        print(f"\n[⚠️] 众数 peak hour 是 UTC {mode_hour} 点，以下年份偏离超过 2 小时，"
              f"可能是该年时间戳未做 UTC 校准，建议人工抽查该年 1-2 天原始数据：")
        print(outliers.to_string())
    else:
        print(f"[✅] 各年 peak hour 都在 UTC {mode_hour} 点 ±2 小时内，未见明显时区错位迹象")


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
