#!/usr/bin/env python3
"""
dukascopy_export_2017_now.py
=============================
从 Dukascopy 官方接口拉取 2017 年至今的 XAUUSD M1 数据（含真实 volume），
文件名直接用于生产（不带 _DUKAREAL 后缀），可直接拷贝替换 data/raw/ 里的旧数据。

当年（今年）只导出到当前日期为止，不会请求未来数据。

运行环境：本地机器，需要 pip install dukascopy-python pandas pyyaml
"""

import calendar
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dk_instruments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402

START_YEAR = 2017
OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID


def find_xau_instrument():
    candidates = [name for name in dir(dk_instruments) if "XAU" in name.upper()]
    if not candidates:
        raise RuntimeError("没有找到包含 XAU 的品种常量，请检查 instruments 模块")
    for c in candidates:
        if "USD" in c.upper():
            return getattr(dk_instruments, c), c
    return getattr(dk_instruments, candidates[0]), candidates[0]


def find_m1_interval():
    if hasattr(dukascopy_python, "INTERVAL_MIN_1"):
        return dukascopy_python.INTERVAL_MIN_1, "INTERVAL_MIN_1"
    all_intervals = [n for n in dir(dukascopy_python) if n.startswith("INTERVAL_")]
    raise RuntimeError(f"没有找到 INTERVAL_MIN_1，可选项: {all_intervals}")


def save_as_dukascopy_format(df: pd.DataFrame, out_path: Path) -> None:
    out = pd.DataFrame()
    out["datetime_str"] = (
        df.index.strftime("%Y%m%d %H%M%S")
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.to_datetime(df["timestamp"]).dt.strftime("%Y%m%d %H%M%S")
    )
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col].values
    out.to_csv(out_path, sep=";", header=False, index=False)
    print(f"[输出] {out_path} ({len(out)}行)")


def main() -> None:
    cfg = load_data_paths()
    output_dir = Path(cfg["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    instrument, instrument_name = find_xau_instrument()
    interval, interval_name = find_m1_interval()
    print(f"[信息] 品种常量: {instrument_name}, interval: {interval_name}\n")

    now = datetime.now()
    end_year = now.year

    for year in range(START_YEAR, end_year + 1):
        print(f"\n{'=' * 50}\n处理年份: {year}\n{'=' * 50}")
        monthly_frames = []
        last_month = 12 if year < end_year else now.month

        for month in range(1, last_month + 1):
            last_day = calendar.monthrange(year, month)[1]
            start = datetime(year, month, 1)
            end = now if (year == end_year and month == now.month) else datetime(year, month, last_day, 23, 59, 59)

            try:
                df = dukascopy_python.fetch(instrument, interval, OFFER_SIDE, start, end)
            except Exception as e:
                print(f"  [{year}-{month:02d}] 请求失败: {e}")
                continue

            if df is None or len(df) == 0:
                print(f"  [{year}-{month:02d}] 无数据")
                continue

            print(f"  [{year}-{month:02d}] 拿到 {len(df)} 条")
            monthly_frames.append(df)

        if monthly_frames:
            year_df = pd.concat(monthly_frames)
            out_path = output_dir / f"DAT_ASCII_XAUUSD_M1_{year}.csv"
            save_as_dukascopy_format(year_df, out_path)
        else:
            print(f"[{year}] 全年无数据")

    print(f"\n完成。文件输出在: {output_dir}")


if __name__ == "__main__":
    main()
