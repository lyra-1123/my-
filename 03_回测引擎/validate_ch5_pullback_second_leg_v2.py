"""
validate_ch5_pullback_second_leg_v2.py
=========================================
假设9 v2 第5章。候选 = v2 第4章挑出的周期 + 第一段门槛；变体家族 = 同一周期上的 2 个门槛（5 / 8 倍 ATR），
同一次回调只做一笔。

持仓过周末（v2b），成本含隔夜利息。
n_trials_prior = 74：此前累计 76（含 v2 停盘前平仓的 4 个 + v2b 的 4 个）减去本家族 2 个。DSR 用 n_trials = 76。

用法：
    python 03_回测引擎/validate_ch5_pullback_second_leg_v2.py <H1/H4> <5.0/8.0>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from signal_pullback_second_leg import (V2_LEG_ATR_GRID as LEG_ATR_GRID, V2_TIMEFRAMES as TIMEFRAMES,  # noqa: E402
                                        exit_rule_for, load_features, load_m1, signals_for)

N_TRIALS_PRIOR = 74


def main() -> None:
    try:
        tf, leg = sys.argv[1], float(sys.argv[2])
        assert tf in TIMEFRAMES and leg in LEG_ATR_GRID
    except (IndexError, ValueError, AssertionError):
        sys.exit(f"用法：python {Path(__file__).name} <{'/'.join(TIMEFRAMES)}> <{'/'.join(map(str, LEG_ATR_GRID))}>")

    print(f"[加载] M1 -> {tf} -> 识别回调形态 ...")
    feat = load_features(load_m1(), tf)
    variants = []
    for g in LEG_ATR_GRID:
        sig, _ = signals_for(feat, g, dedupe=True)
        market = prepare_market(feat, stop_dist=sig["stop_dist"], target_dist=sig["target_dist"],
                                entry_price=sig["entry_price"], entry_hi=sig["entry_hi"], entry_lo=sig["entry_lo"])
        variants.append(Variant(f"leg{g}", sig["signal"], exit_rule_for(tf, hold_through_gaps=True), market))
    candidate = next(v for v in variants if v.name == f"leg{leg}")

    print(f"[回测] 候选 {tf}|{candidate.name}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, candidate.market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, f"{tf}|{candidate.name}")


if __name__ == "__main__":
    main()
