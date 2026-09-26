#!/usr/bin/env python3
"""Build and cache the M5 factor table (same library as Phase 2's H1 work,
computed directly on M5 bars) for the directional signal mining in
02h_directional_factor_mining.py. Separate from factors_H1.parquet/
build_factor_table's H1 usage in Phase 2 — this is signal-design track,
not the regime-classification track, though it reuses the same indicator
code (src/factors/library.py) since the formulas are timeframe-agnostic.

v2: --windows lets this build with M5-NATIVE bar-count windows instead of
the default (10,20,50,100), which were validated on H1 and mean a very
different absolute time span on M5 (e.g. "_100" = ~100h on H1 vs ~8.3h on
M5) — see reports/02j_window_optimization_report.md.

Usage:
    python scripts/02g_build_m5_factors.py --clean-dir data/clean
    python scripts/02g_build_m5_factors.py --clean-dir data/clean \
        --windows 6,12,24,48,96,192,288 --out-name factors_M5_opt.parquet
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import build_factor_table  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--windows", default=None,
                         help="comma-separated bar-count windows, e.g. 6,12,24,48,96,192,288 "
                              "(default: library.py's WINDOWS = 10,20,50,100)")
    parser.add_argument("--out-name", default="factors_M5.parquet")
    args = parser.parse_args()
    windows = tuple(int(w) for w in args.windows.split(",")) if args.windows else None

    print("[1/2] Loading M5 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"])
    df = df.reset_index(drop=True)
    print(f"      {len(df):,} M5 bars")

    print(f"[2/2] Building factor table (windows={windows or 'default'}) ...")
    t0 = time.time()
    factors = build_factor_table(df, windows=windows) if windows else build_factor_table(df)
    factors["close"] = df["close"].values
    factors["open"] = df["open"].values
    out_path = os.path.join(args.clean_dir, args.out_name)
    factors.to_parquet(out_path, index=False)
    print(f"      {factors.shape[1]} columns, {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
