"""
validate_exits_atr_momentum_v2.py
====================================
第4章方向性筛选（Trap-002 之后补的检验）：ATR动量v2 信号（ATR扩张1.5倍）在 7 种新出场规则下，
信号方向 vs 同一批bar随机方向的逐笔R。最后一行是旧对称口径（1倍ATR、30分钟），用来对照
第5章v1的结果（应显示方向边际≈0）。

通过筛选的出场规则里，成本后R最高的作为第5章候选。

用法：
    python 04_策略研究/validate_exits_atr_momentum_v2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_回测引擎"))
from cost_model import SimpleCostModel  # noqa: E402
from goldq.exits import (ema_macd_reversal_exits, legacy_symmetric_exit, prepare_market,  # noqa: E402
                         standard_exit_family)
from goldq.signal_validation import evaluate_exit_direction, print_exit_direction_table  # noqa: E402
from signal_atr_momentum_v2 import build_features, load_m5, signal_from_features  # noqa: E402

MAX_BARS = 24


def main() -> None:
    print("[加载] M1 -> M5 -> 计算特征 ...")
    feat = build_features(load_m5())
    market = prepare_market(feat, *ema_macd_reversal_exits(feat["close"], feat["ema20"],
                                                           feat["golden_cross"], feat["dead_cross"]))
    signal = signal_from_features(feat)
    cost = SimpleCostModel().round_trip_cost()
    print(f"[信号] {int((signal != 0).sum())} 次，成本 ${cost:.2f}/笔\n")

    rules = standard_exit_family(MAX_BARS) + [legacy_symmetric_exit(1.0, 6)]
    results = {r.name: evaluate_exit_direction(market, signal, r, cost) for r in rules}
    passed = print_exit_direction_table(results)
    passed = [p for p in passed if p in {r.name for r in standard_exit_family(MAX_BARS)}]

    if passed:
        best = max(passed, key=lambda name: results[name]["net_r"])
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_atr_momentum_v2_exits.py {best}")
    else:
        print("\n没有出场规则通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
