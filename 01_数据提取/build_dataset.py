#!/usr/bin/env python3
"""
build_dataset.py
=================
把 data/raw/ 下的所有 DAT_ASCII_XAUUSD_M1_*.csv（Dukascopy 格式）合并、去重、排序，
落地为唯一的处理后数据集 data/processed/XAUUSD_M1.parquet。

这是唯一允许生成 processed 数据集的脚本——下游一律通过 lib/goldq/datastore.py 的
fetch_bars() 读取，不允许绕过本脚本直接拼接 CSV（SSoT 铁律）。

用法：
    python 01_数据提取/build_dataset.py
    GOLDQ_RAW_DIR=/path/to/csv python 01_数据提取/build_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402
from goldq.dukascopy_io import read_dukascopy_csv  # noqa: E402

# 同一年份可能存在多个版本文件（历史/MT5REAL/补volume版），按优先级去重：
# 优先用补齐了真实 volume 的版本 > Dukascopy 原生 > MT5 直连兜底版
FILENAME_PRIORITY = ["_with_volume.csv", ".csv"]


def _pick_files_by_year(raw_dir: Path) -> list[Path]:
    all_csv = sorted(raw_dir.glob("DAT_ASCII_XAUUSD_M1_*.csv"))
    by_year: dict[str, Path] = {}
    for path in all_csv:
        # 文件名形如 DAT_ASCII_XAUUSD_M1_2016_DUKAREAL.csv / DAT_ASCII_XAUUSD_M1_2020.csv
        stem = path.stem.replace("DAT_ASCII_XAUUSD_M1_", "")
        year = stem.split("_")[0]
        if not year.isdigit():
            continue
        prev = by_year.get(year)
        if prev is None or "_with_volume" in path.name:
            by_year[year] = path
    return [by_year[y] for y in sorted(by_year)]


def main() -> None:
    cfg = load_data_paths()
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = _pick_files_by_year(raw_dir)
    if not files:
        print(f"[build_dataset] 在 {raw_dir} 下没有找到 DAT_ASCII_XAUUSD_M1_*.csv")
        print("请先把 Google Drive 里的 CSV 放到该目录，或设置 GOLDQ_RAW_DIR 指向实际位置。")
        sys.exit(1)

    frames = []
    for path in files:
        print(f"[读取] {path.name}")
        frames.append(read_dukascopy_csv(path))

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset="time_utc", keep="last").sort_values("time_utc").reset_index(drop=True)
    after = len(df)

    out_path = processed_dir / f"{cfg['symbol']}_{cfg['timeframe_base']}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n[完成] 合并 {len(files)} 个文件，{before} 行 -> 去重排序后 {after} 行")
    print(f"[完成] 时间范围: {df['time_utc'].min()} ~ {df['time_utc'].max()}")
    print(f"[完成] 已写入: {out_path}")
    print("\n下一步：运行 python 01_数据提取/validate_data.py 做数据自检")


if __name__ == "__main__":
    main()
