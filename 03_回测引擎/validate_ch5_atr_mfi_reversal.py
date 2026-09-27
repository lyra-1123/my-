"""
validate_ch5_atr_mfi_reversal.py
===================================
假设5（ATR放大 + MFI 极值反转，00_方案/hypothesis_atr_mfi_reversal.md）第5章。

变体家族 = ATR 放大倍数 x ∈ {1.5, 2, 2.5, 3}，出场固定为 1.5ATR 移动止损、不限持仓。
候选 x 由第4章筛选脚本（04_策略研究/validate_signal_atr_mfi_reversal.py）挑出，作为参数传入。

n_trials_prior = 23（见 00_方案/changelog.md 累计数），加上本家族 4 个，DSR 用 n_trials = 27。

用法：
    python 03_回测引擎/validate_ch5_atr_mfi_reversal.py <x，如 2.0>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from signal_atr_mfi_reversal import (EXIT_RULE, EXPANSION_GRID, build_features, load_m5,  # noqa: E402
                                     signal_from_features)

N_TRIALS_PRIOR = 23


def main() -> None:
    try:
        x = float(sys.argv[1])
        assert x in EXPANSION_GRID
    except (IndexError, ValueError, AssertionError):
        sys.exit(f"用法：python {Path(__file__).name} <x>，可选：{EXPANSION_GRID}")

    print("[加载] M1 -> M5 -> ATR(排除跳空)/MFI ...")
    feat = build_features(load_m5())
    market = prepare_market(feat)
    variants = [Variant(f"x={e}", signal_from_features(feat, e), EXIT_RULE) for e in EXPANSION_GRID]
    candidate = next(v for v in variants if v.name == f"x={x}")

    print(f"[回测] 候选 {candidate.name}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, candidate.name)


if __name__ == "__main__":
    main()
