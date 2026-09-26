#!/usr/bin/env python3
"""Build and cache the M5 factor table (same library as Phase 2's H1 work,
computed directly on M5 bars) for the directional signal mining in
02h_directional_factor_mining.py. Separate from factors_H1.parquet/
build_factor_table's H1 usage in Phase 2 — this is signal-design track,
not the regime-classification track, though it reuses the same indicator
code (src/factors/library.py) since the formulas are timeframe-agnostic.

Usage:
    python scripts/02g_build_m5_factors.py --clean-dir data/clean
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
    args = parser.parse_args()

    print("[1/2] Loading M5 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"])
    df = df.reset_index(drop=True)
    print(f"      {len(df):,} M5 bars")

    print("[2/2] Building factor table (same library as Phase 2, computed on M5) ...")
    t0 = time.time()
    factors = build_factor_table(df)
    factors["close"] = df["close"].values
    factors["open"] = df["open"].values
    out_path = os.path.join(args.clean_dir, "factors_M5.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"      {factors.shape[1]} columns, {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
