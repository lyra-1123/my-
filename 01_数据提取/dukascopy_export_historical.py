#!/usr/bin/env python3
"""
dukascopy_export_historical.py
================================
用 dukascopy-python 补拉 MT5 broker 拿不到的更早历史年份（默认 2009-2016），
覆盖到 2005 年左右。文件名带 _DUKAREAL 后缀，与直接 HistData 文件区分。

运行环境：本地机器，需要 pip install dukascopy-python pandas pyyaml
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dk_instruments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402

YEARS_TO_FETCH = list(range(2009, 2017))
OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID


def find_xau_instrument():
    candidates = [name for name in dir(dk_instruments) if "XAU" in name.upper()]
    if not candidates:
        raise RuntimeError("在 dukascopy_python.instruments 里没找到包含 XAU 的常量，请手动查看模块内容确认正确的实例名称。")
    print(f"[信息] 找到候选常见品种: {candidates}")
    for c in candidates:
        if "USD" in c.upper():
            return getattr(dk_instruments, c), c
    return getattr(dk_instruments, candidates[0]), candidates[0]


def find_m1_interval():
    candidates = [
        name for name in dir(dukascopy_python)
        if name.startswith("INTERVAL_MIN_1") or name == "INTERVAL_MIN_1"
    ]
    if candidates:
        return getattr(dukascopy_python, candidates[0]), candidates[0]
    all_intervals = [n for n in dir(dukascopy_python) if n.startswith("INTERVAL_")]
    raise RuntimeError(f"没自动找到1分钟interval常量，请从下列表里手动确认并修改脚本: {all_intervals}")


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
    print(f"[信息] 使用品种常量: {instrument_name}, interval: {interval_name}\n")

    for year in YEARS_TO_FETCH:
        print(f"\n{'=' * 50}\n处理年份: {year}\n{'=' * 50}")
        monthly_frames = []

        for month in range(1, 13):
            import calendar

            last_day = calendar.monthrange(year, month)[1]
            start = datetime(year, month, 1)
            end = datetime(year, month, last_day, 23, 59, 59)

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
            out_path = output_dir / f"DAT_ASCII_XAUUSD_M1_{year}_DUKAREAL.csv"
            save_as_dukascopy_format(year_df, out_path)
        else:
            print(f"[{year}] 全年无数据，可能超出该接口实际覆盖范围")

    print(f"\n完成。文件输出在: {output_dir}")


if __name__ == "__main__":
    main()
