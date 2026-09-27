"""
validate_ch5_support_resistance.py
=====================================
假设7（支撑/阻力）第5章。候选 = 第4章挑出的 周期 + 用法 + 价位识别方法；
变体家族 = 同一周期、同一用法下的 4 种价位识别方法。

n_trials_prior = 56：此前累计 36 + 本假设第4章里其余 20 个组合。加上本家族 4 个，DSR 用 n_trials = 60。

用法：
    python 03_回测引擎/validate_ch5_support_resistance.py <M5/M15/H1> <filter/bounce> <cluster/htf/prev_dw/round>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from signal_support_resistance import (EXIT_RULE, METHODS, TIMEFRAMES, USES, all_signals,  # noqa: E402
                                       build_features, load_m1, load_tf)

N_TRIALS_PRIOR = 56


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3 or args[0] not in TIMEFRAMES or args[1] not in USES or args[2] not in METHODS:
        sys.exit(f"用法：python {Path(__file__).name} <{'/'.join(TIMEFRAMES)}> <{'/'.join(USES)}> <{'/'.join(METHODS)}>")
    tf, use, method = args[:3]

    print(f"[加载] M1 -> {tf} -> 形态与支撑阻力 ...")
    feat, patterns = build_features(load_tf(load_m1(), tf))
    signals = all_signals(feat, patterns, tf)
    variants = [Variant(m, signals[(use, m)][0], EXIT_RULE, prepare_market(feat, stop_dist=signals[(use, m)][1]))
                for m in METHODS]
    candidate = next(v for v in variants if v.name == method)

    print(f"[回测] 候选 {tf}|{use}|{method}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, candidate.market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, f"{tf}|{use}|{method}")


if __name__ == "__main__":
    main()
