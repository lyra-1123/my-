"""
数据源 SSoT（Single Source of Truth）唯一入口。

全系统（信号 / 回测 / 风控代码）一律通过 fetch_bars() 读取行情，
禁止在 02_增强处理/03_回测引擎/04_策略研究/06_运维脚本 中直接 pd.read_csv 原始数据。
"""

from pathlib import Path

import pandas as pd

from .config import load_data_paths


def dataset_path(symbol: str = "XAUUSD", timeframe: str = "M1") -> Path:
    cfg = load_data_paths()
    return Path(cfg["processed_dir"]) / f"{symbol}_{timeframe}.parquet"


def fetch_bars(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    path = dataset_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到处理后的数据集: {path}\n"
            f"请先运行: python 01_数据提取/build_dataset.py"
        )

    df = pd.read_parquet(path)
    if start is not None:
        df = df[df["time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["time_utc"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)
