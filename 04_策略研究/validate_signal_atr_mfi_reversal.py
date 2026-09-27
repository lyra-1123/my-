"""
validate_signal_atr_mfi_reversal.py
======================================
假设5 第4章筛选：每个 ATR 放大倍数 x，在同一批信号bar上比较信号方向与随机方向（同一个
1.5ATR 移动止损、不限持仓的出场），再按年份看方向边际是否稳定。

用法：
    python 04_策略研究/validate_signal_atr_mfi_reversal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_回测引擎"))
from cost_model import SimpleCostModel  # noqa: E402
from goldq.exits import prepare_market  # noqa: E402
from goldq.signal_validation import (edge_by_year, evaluate_exit_direction,  # noqa: E402
                                     print_exit_direction_table)
from signal_atr_mfi_reversal import (EXIT_RULE, EXPANSION_GRID, build_features, load_m5,  # noqa: E402
                                     signal_from_features)


def main() -> None:
    print("[加载] M1 -> M5 -> ATR(排除跳空)/MFI ...")
    feat = build_features(load_m5())
    market = prepare_market(feat)
    cost = SimpleCostModel().round_trip_cost()
    print(f"[加载] M5 共 {len(feat)} 根，成本 ${cost:.2f}/笔，出场 {EXIT_RULE.name}\n")

    results = {}
    for x in EXPANSION_GRID:
        signal = signal_from_features(feat, x)
        print(f"x={x}: 做多 {int((signal == 1).sum())} 次，做空 {int((signal == -1).sum())} 次")
        results[f"x={x}"] = evaluate_exit_direction(market, signal, EXIT_RULE, cost)
    print()

    passed = print_exit_direction_table(results)

    for name, res in results.items():
        if not res.get("n"):
            continue
        rec = res["records"]
        print(f"\n[{name}] 按方向：")
        for d, label in [(1, "多头"), (-1, "空头")]:
            sub = rec[rec["direction"] == d]
            if len(sub):
                print(f"  {label}: n={len(sub)}, 方向边际 {sub['edge'].mean():+.3f}R, "
                      f"成本后 {(sub['r'] - sub['cost_r']).mean():+.3f}R")
        print(f"[{name}] 按年份（方向边际 edge_r、成本后 net_r，单位R）")
        print(edge_by_year(rec, feat["time_utc"]).to_string(float_format=lambda v: f"{v:.3f}"))

    if passed:
        best = max(passed, key=lambda n: results[n]["net_r"])
        x = best.split("=")[1]
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_atr_mfi_reversal.py {x}")
    else:
        print("\n没有任何 x 通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
