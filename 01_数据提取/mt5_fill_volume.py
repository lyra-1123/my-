#!/usr/bin/env python3
"""
mt5_fill_volume.py
====================
用 MT5 官方 Python API 拉取 XAUUSD 的 M1 数据（含 tick_volume），
补充/合并进现有 Dukascopy 格式 CSV 历史数据中（原 volume 列全为0的情况）。

⚠️ 时区铁律：MT5 返回的时间是 broker 服务器时间，不是 UTC；
Dukascopy 数据通常是 UTC 时间（部分海外数据源也有 UTC+0 的）。
本脚本用 07_配置参数/data_paths.yaml 的 mt5_utc_offset_hours 做校准，
verify_overlap() 会打印重叠样本比对，用于人工核实这个偏移是否设对了——
不要盲目信任默认值。

运行环境：本地Windows机器，需登录好MT5终端，pip install MetaTrader5 pandas pyyaml
"""

import glob
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402
from goldq.dukascopy_io import DUKASCOPY_COLUMNS  # noqa: E402

SYMBOL = "XAUUSD"
VOLUME_COLUMN = "tick_volume"  # 用tick_volume而不是real_volume（外汇/贵金属real_volume通常全为0）


def init_mt5() -> None:
    if mt5 is None:
        raise ImportError("请先 pip install MetaTrader5")
    if not mt5.initialize():
        raise RuntimeError(f"MT5初始化失败: {mt5.last_error()}")
    print(f"[MT5] 已连接，终端信息: {mt5.terminal_info()}")


def fetch_mt5_m1(symbol: str, start, end, utc_offset_hours: int) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None or len(rates) == 0:
        print(f"[MT5] 警告: {start} ~ {end} 区间没有拉到数据 (err={mt5.last_error()})")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["time_utc"] = df["time"] - timedelta(hours=utc_offset_hours)

    print(f"[MT5] 拉取到 {len(df)} 条记录，服务器时间范围: {df['time'].min()} ~ {df['time'].max()}")
    return df[["time_utc", VOLUME_COLUMN]].rename(columns={VOLUME_COLUMN: "mt5_volume"})


def load_dukascopy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None, names=DUKASCOPY_COLUMNS, dtype={"datetime_str": str})
    df["time_utc"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S")
    return df


def verify_overlap(dukascopy_df: pd.DataFrame, mt5_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """抽取重叠时间段，打印双方收盘价对比，用于人工核对 mt5_utc_offset_hours 是否设对了。"""
    merged = pd.merge(
        dukascopy_df[["time_utc", "close"]],
        mt5_df,
        on="time_utc",
        how="inner",
    )
    print(f"\n[核对] 重叠时间戳数量: {len(merged)}")
    if len(merged) == 0:
        print("[核对] 没有任何时间戳对得上，mt5_utc_offset_hours 很可能设错了，"
              "或两边数据时间范围本来就不重叠。请检查后重试。")
        return merged
    print(merged.sample(min(n, len(merged))).sort_values("time_utc"))
    return merged


def fill_volume(dukascopy_df: pd.DataFrame, mt5_df: pd.DataFrame) -> pd.DataFrame:
    """以time_utc为key left join，把MT5的volume填入现有数据行对应。
    原volume列非0的行保留原值，只补0/缺失的行。"""
    merged = pd.merge(dukascopy_df, mt5_df, on="time_utc", how="left")

    mask_fillable = (merged["volume"] == 0) & merged["mt5_volume"].notna()
    filled_count = mask_fillable.sum()

    merged.loc[mask_fillable, "volume"] = merged.loc[mask_fillable, "mt5_volume"]

    total = len(merged)
    print(f"\n[合并] 总行数: {total}, 本次补上volume的行数: {filled_count} "
          f"({filled_count/total*100:.1f}%)")

    return merged.drop(columns=["mt5_volume"])


def save_as_dukascopy_format(df: pd.DataFrame, out_path: str) -> None:
    out = df.copy()
    out["datetime_str"] = out["time_utc"].dt.strftime("%Y%m%d %H%M%S")
    out[DUKASCOPY_COLUMNS].to_csv(out_path, sep=";", header=False, index=False)
    print(f"[输出] 已保存: {out_path}")


def main() -> None:
    cfg = load_data_paths()
    raw_dir = cfg["raw_dir"]
    utc_offset_hours = cfg.get("mt5_utc_offset_hours", 0)

    init_mt5()

    csv_files = sorted(glob.glob(str(Path(raw_dir) / "DAT_ASCII_XAUUSD_M1_*.csv")))
    if not csv_files:
        print(f"[主流程] 在 {raw_dir} 没找到匹配的CSV文件")
        return

    for csv_path in csv_files:
        fname = Path(csv_path).name
        print(f"\n{'=' * 60}\n处理: {fname}\n{'=' * 60}")

        dukascopy_df = load_dukascopy_csv(csv_path)
        t_min, t_max = dukascopy_df["time_utc"].min(), dukascopy_df["time_utc"].max()

        mt5_start = t_min + timedelta(hours=utc_offset_hours)
        mt5_end = t_max + timedelta(hours=utc_offset_hours)
        mt5_df = fetch_mt5_m1(SYMBOL, mt5_start, mt5_end, utc_offset_hours)

        if mt5_df.empty:
            print(f"[主流程] {fname} 对应时段MT5没有数据（很可能超出服务器保存的历史深度），跳过")
            continue

        verify_overlap(dukascopy_df, mt5_df)

        merged_df = fill_volume(dukascopy_df, mt5_df)

        out_path = str(Path(raw_dir) / fname.replace(".csv", "_with_volume.csv"))
        save_as_dukascopy_format(merged_df, out_path)

    mt5.shutdown()


if __name__ == "__main__":
    main()
