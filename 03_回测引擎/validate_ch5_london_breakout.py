"""
validate_ch5_london_breakout.py
==================================
假设8（伦敦开盘突破）第5章。变体家族 = 2 种出场（当日收盘平仓 / 2R 或当日收盘）。

n_trials_prior = 60（见 00_方案/changelog.md），加上本家族 2 个，DSR 用 n_trials = 62。

用法：
    python 03_回测引擎/validate_ch5_london_breakout.py <stop_dayend/stop_2R_dayend>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from signal_london_breakout import EXIT_RULES, build_features, load_m15  # noqa: E402

N_TRIALS_PRIOR = 60


def main() -> None:
    names = [r.name for r in EXIT_RULES]
    if len(sys.argv) < 2 or sys.argv[1] not in names:
        sys.exit(f"用法：python {Path(__file__).name} <{'/'.join(names)}>")

    print("[加载] M1 -> M15 -> 亚洲盘区间 / 伦敦开盘突破 ...")
    feat = build_features(load_m15())
    market = prepare_market(feat, stop_dist=feat["stop_dist"], last_idx=feat["last_idx"])
    variants = [Variant(r.name, feat["signal"], r) for r in EXIT_RULES]
    candidate = next(v for v in variants if v.name == sys.argv[1])

    print(f"[回测] 候选 {candidate.name}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, candidate.name)


if __name__ == "__main__":
    main()
