"""
validate_ch5_atr_momentum_v2.py
==================================
把 ATR动量v2（00_方案/hypothesis_atr_momentum_v2.md，第4章唯一通过的候选）送进第5章流水线。

变体家族（CSCV/PBO 和 WF选择法用）：候选参数周围的 3x3 邻域——
  ATR扩张倍数 ∈ {1.2, 1.5, 2.0}  ×  SL/TP的ATR倍数 ∈ {0.75, 1.0, 1.5}
这是"研究员手动调参时自然会试的那几组"，候选 (1.5, 1.0) 在正中间。

n_trials_prior = 8：跑本脚本之前，本项目已在全量真实数据上正式验证过的策略定义数
  假设1 MACD水下金叉 ................................ 1
  假设2 三周期共振 $10 目标 + $5 目标复测 ............. 2
  假设3 单一大周期 1H / 30m / 15m ...................... 3
  假设4 ATR动量 v1（RSI，固定$3） ..................... 1
  假设4 ATR动量 v2（MFI，1倍ATR，即本候选） ............ 1
这个数只会低估不会高估（每个假设文档内部还有没单独计数的参数选择），DSR 因此偏宽松。

用法：
    python 03_回测引擎/validate_ch5_atr_momentum_v2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from chapter5_pipeline import Variant, print_chapter5_report, run_chapter5_validation  # noqa: E402
from signal_atr_momentum_v2 import build_features, load_m5, signal_from_features, target_series  # noqa: E402

FORWARD_BARS = 6
N_TRIALS_PRIOR = 8
EXPANSION_GRID = [1.2, 1.5, 2.0]
TARGET_GRID = [0.75, 1.0, 1.5]
CANDIDATE = (1.5, 1.0)


def variant_name(expansion: float, target_mult: float) -> str:
    return f"exp{expansion}_tgt{target_mult}"


def main() -> None:
    print("[加载] M1 -> M5 -> 计算特征（只算一次，9个变体共用）...")
    feat = build_features(load_m5())
    print(f"[加载] M5 共 {len(feat)} 根")

    signals = {exp: signal_from_features(feat, exp) for exp in EXPANSION_GRID}
    variants = [
        Variant(variant_name(exp, tgt), signals[exp], target_series(feat, tgt))
        for exp in EXPANSION_GRID for tgt in TARGET_GRID
    ]
    candidate = next(v for v in variants if v.name == variant_name(*CANDIDATE))

    print(f"[回测] 候选 {candidate.name}，变体家族 {len(variants)} 个，开始跑第5章流水线 ...")
    result = run_chapter5_validation(feat, candidate, variants, FORWARD_BARS, N_TRIALS_PRIOR)
    print_chapter5_report(result, candidate.name)


if __name__ == "__main__":
    main()
