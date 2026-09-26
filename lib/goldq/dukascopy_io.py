"""
Dukascopy ASCII 格式读写（永久铁律：本仓库唯一 CSV 格式解析入口）。

格式：datetime_str;open;high;low;close;volume（无表头，分号分隔）
示例：20260101 180000;4323.698000;4332.848000;4323.698000;4328.198000;0
时间约定：Dukascopy 原生导出（dukascopy_python.fetch）返回的时间戳是真 UTC。
    MT5 直连导出（mt5_export_direct.py）默认是 broker 服务器时间，不是 UTC——
    与 Dukascopy 来源的文件混用前必须先做 verify_overlap 式的时区校准（见 mt5_fill_volume.py），
    否则会静默产生跨源时间错位（详见 99_结果/traps_log.md）。
"""

from pathlib import Path

import pandas as pd

DUKASCOPY_COLUMNS = ["datetime_str", "open", "high", "low", "close", "volume"]


def read_dukascopy_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=DUKASCOPY_COLUMNS,
        dtype={"datetime_str": str},
    )
    df["time_utc"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S", utc=True)
    return df.drop(columns=["datetime_str"])[["time_utc", "open", "high", "low", "close", "volume"]]


def write_dukascopy_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = pd.DataFrame()
    out["datetime_str"] = df["time_utc"].dt.strftime("%Y%m%d %H%M%S")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col].values
    out.to_csv(path, sep=";", header=False, index=False)
