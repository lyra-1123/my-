#!/usr/bin/env python3
"""Build the cleaned XAUUSD M1 dataset (+ resampled timeframes) from raw
yearly Dukascopy-format CSVs.

Usage:
    python scripts/build_clean_dataset.py \
        --raw-dir data/raw --out-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.pipeline import load_all_raw, clean, resample  # noqa: E402

TIMEFRAMES = {"M5": "5min", "M15": "15min", "H1": "h", "H4": "4h", "D1": "D"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print(f"[1/4] Loading raw CSVs from {args.raw_dir} ...")
    raw = load_all_raw(args.raw_dir)
    print(f"      {len(raw):,} raw rows loaded")

    print("[2/4] Cleaning ...")
    df, report = clean(raw)
    print(f"      {report.n_clean_rows:,} clean rows "
          f"({report.n_duplicate_timestamps} dup, "
          f"{report.n_non_monotonic} non-monotonic, "
          f"{report.n_ohlc_inconsistent} OHLC-inconsistent, "
          f"{report.n_zero_or_negative_price} zero/neg price dropped; "
          f"{report.n_suspected_spikes} suspected spikes flagged, not dropped)")

    m1_path = os.path.join(args.out_dir, "XAUUSD_M1.parquet")
    df.to_parquet(m1_path, index=False)
    print(f"[3/4] Wrote {m1_path}")

    for name, rule in TIMEFRAMES.items():
        tf_df = resample(df[~df["is_suspected_spike"]], rule)
        tf_path = os.path.join(args.out_dir, f"XAUUSD_{name}.parquet")
        tf_df.to_parquet(tf_path, index=False)
        print(f"      Wrote {tf_path} ({len(tf_df):,} rows)")

    report_path = os.path.join(args.report_dir, "data_quality_report.md")
    with open(report_path, "w") as f:
        f.write(report.to_markdown())
    print(f"[4/4] Wrote {report_path}")


if __name__ == "__main__":
    main()
