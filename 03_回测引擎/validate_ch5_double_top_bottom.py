"""
validate_ch5_double_top_bottom.py
====================================
假设6（双顶/双底）第5章。候选 = 第4章筛选挑出的 周期 + 出场；变体家族 = 该周期上的 3 种出场。

n_trials_prior = 33：此前累计 27 + 本假设第4章在另外两个周期上测过的 6 个组合。
加上本家族 3 个，DSR 用 n_trials = 36。

用法：
    python 03_回测引擎/validate_ch5_double_top_bottom.py <周期 M5/M15/H1> <出场名>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from signal_double_top_bottom import EXIT_RULES, TIMEFRAMES, build_features, load_m1, load_tf  # noqa: E402

N_TRIALS_PRIOR = 33


def main() -> None:
    names = [r.name for r in EXIT_RULES]
    if len(sys.argv) < 3 or sys.argv[1] not in TIMEFRAMES or sys.argv[2] not in names:
        sys.exit(f"用法：python {Path(__file__).name} <{'/'.join(TIMEFRAMES)}> <{'/'.join(names)}>")
    tf, rule_name = sys.argv[1], sys.argv[2]

    print(f"[加载] M1 -> {tf} -> 识别双顶/双底 ...")
    feat, _ = build_features(load_tf(load_m1(), tf))
    market = prepare_market(feat, stop_dist=feat["stop_dist"], target_dist=feat["target_dist"])
    variants = [Variant(r.name, feat["signal"], r) for r in EXIT_RULES]
    candidate = next(v for v in variants if v.name == rule_name)

    print(f"[回测] 候选 {tf}|{candidate.name}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, f"{tf}|{candidate.name}")


if __name__ == "__main__":
    main()
