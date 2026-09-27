"""
validate_ch5_atr_momentum_v2_exits.py
========================================
ATR动量v2 用新出场规则复测第5章（2026-09-27 用户决定）。

信号固定为第4章候选（ATR扩张1.5倍），变体家族 = lib/goldq/exits.standard_exit_family()
的 7 种出场组合，最长持仓 24 根 M5（2小时）。候选出场由第4章出场筛选脚本
（04_策略研究/validate_exits_atr_momentum_v2.py）按"成本前方向边际"挑出，作为参数传入。

n_trials_prior = 16：此前已验证的策略定义（假设1-4共7个 + ATR v2 对称出场家族9个）。
加上本家族 7 个，DSR 用 n_trials = 23。

用法：
    python 03_回测引擎/validate_ch5_atr_momentum_v2_exits.py <候选出场名，如 trail_2R>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from goldq.exits import ema_macd_reversal_exits, prepare_market, standard_exit_family  # noqa: E402
from signal_atr_momentum_v2 import build_features, load_m5, signal_from_features  # noqa: E402

N_TRIALS_PRIOR = 16
MAX_BARS = 24


def main() -> None:
    rules = standard_exit_family(MAX_BARS)
    names = [r.name for r in rules]
    if len(sys.argv) < 2 or sys.argv[1] not in names:
        sys.exit(f"用法：python {Path(__file__).name} <候选出场名>，可选：{', '.join(names)}")
    candidate_name = sys.argv[1]

    print("[加载] M1 -> M5 -> 计算特征 ...")
    feat = build_features(load_m5())
    market = prepare_market(feat, *ema_macd_reversal_exits(feat["close"], feat["ema20"],
                                                           feat["golden_cross"], feat["dead_cross"]))
    signal = signal_from_features(feat)
    variants = [Variant(r.name, signal, r) for r in rules]
    candidate = next(v for v in variants if v.name == candidate_name)

    print(f"[回测] 候选出场 {candidate.name}，变体家族 {len(variants)} 个 ...")
    result = run_chapter5_validation(feat, market, candidate, variants, N_TRIALS_PRIOR)
    print_chapter5_report(result, candidate.name)


if __name__ == "__main__":
    main()
