"""
validate_signal_london_breakout.py
=====================================
假设8（伦敦开盘突破亚洲盘区间）第4章筛选：2 种出场，同一批信号bar上信号方向 vs 随机方向
（反方向用同样的止损距离镜像，同样在当天收盘强制平仓）。

用法：
    python 04_策略研究/validate_signal_london_breakout.py
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
from signal_london_breakout import EXIT_RULES, build_features, describe_trades, load_m15  # noqa: E402


def main() -> None:
    cost = SimpleCostModel().round_trip_cost()
    print(f"[加载] M1 -> M15 -> 亚洲盘区间 / 伦敦开盘突破 ... 成本 ${cost:.2f}/笔")
    feat = build_features(load_m15())
    sig = feat["signal"]
    rows = feat[sig != 0]
    days = feat["ldn_date"].nunique()
    print(f"[信号] {days} 个伦敦日，做多 {int((sig == 1).sum())}、做空 {int((sig == -1).sum())}"
          f"（{(sig != 0).sum() / days:.0%} 的日子有突破）；止损距离中位数 ${rows['stop_dist'].median():.2f}"
          f"（成本约 {cost / rows['stop_dist'].median():.3f}R）")
    print("[信号] 最近 5 笔（请在 MT5 图上核对，价位为 bid）：")
    print(describe_trades(feat).to_string(index=False))

    market = prepare_market(feat, stop_dist=feat["stop_dist"], last_idx=feat["last_idx"])
    results = {r.name: evaluate_exit_direction(market, sig, r, cost) for r in EXIT_RULES}
    print()
    passed = print_exit_direction_table(results)

    for name, res in results.items():
        rec = res["records"]
        parts = []
        for d, label in [(1, "做多"), (-1, "做空")]:
            sub = rec[rec["direction"] == d]
            parts.append(f"{label} n={len(sub)} 边际{sub['edge'].mean():+.3f}R 成本后{(sub['r'] - sub['cost_r']).mean():+.3f}R")
        print(f"  {name}: " + "；".join(parts))
        print(f"\n[{name}] 按年份")
        print(edge_by_year(rec, feat["time_utc"]).to_string(float_format=lambda v: f"{v:.3f}"))

    if passed:
        best = max(passed, key=lambda n: results[n]["net_r"])
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_london_breakout.py {best}")
    else:
        print("\n没有出场规则通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
