"""
validate_signal_support_resistance.py
========================================
假设7（支撑/阻力）第4章筛选：{过滤双顶双底, 触及反弹} × 4 种价位识别 × {M5, M15, H1}，
出场统一为给定止损 + 2R。另列出不过滤的双顶双底（同样 2R 出场，假设6已计数）作参照，
直接回答"加了支撑阻力有没有变好"。

用法：
    python 04_策略研究/validate_signal_support_resistance.py
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
from signal_support_resistance import (EXIT_RULE, METHODS, TIMEFRAMES, USES, all_signals,  # noqa: E402
                                       build_features, describe_levels, load_m1, load_tf)

REF = "参照:不过滤"


def main() -> None:
    cost = SimpleCostModel().round_trip_cost()
    print(f"[加载] M1 ... 成本 ${cost:.2f}/笔，出场 {EXIT_RULE.name}")
    m1 = load_m1()

    results, times = {}, {}
    for tf in TIMEFRAMES:
        feat, patterns = build_features(load_tf(m1, tf))
        times[tf] = feat["time_utc"]
        print(f"\n[{tf}] 最后一根 {feat['time_utc'].iloc[-1]} 收盘 {feat['close'].iloc[-1]:.2f}，当时已知的价位（请在 MT5 图上核对）：")
        print(describe_levels(feat, tf).to_string(index=False))

        base = prepare_market(feat, stop_dist=feat["stop_dist"])
        results[f"{tf}|{REF}"] = evaluate_exit_direction(base, feat["signal"], EXIT_RULE, cost)
        for (use, method), (sig, stop) in all_signals(feat, patterns, tf).items():
            n_long, n_short = int((sig == 1).sum()), int((sig == -1).sum())
            print(f"  {USES[use]} / {METHODS[method]}: 做多 {n_long}，做空 {n_short}")
            market = prepare_market(feat, stop_dist=stop)
            results[f"{tf}|{use}|{method}"] = evaluate_exit_direction(market, sig, EXIT_RULE, cost)
    print()

    passed = [p for p in print_exit_direction_table(results) if REF not in p]
    print("（带'参照'的行不参与筛选，只用来对比过滤前后）")

    for name, res in results.items():
        if not res.get("n"):
            continue
        rec = res["records"]
        parts = []
        for d, label in [(1, "做多"), (-1, "做空")]:
            sub = rec[rec["direction"] == d]
            if len(sub):
                parts.append(f"{label} n={len(sub)} 边际{sub['edge'].mean():+.3f}R 成本后{(sub['r'] - sub['cost_r']).mean():+.3f}R")
        print(f"  {name}: " + "；".join(parts))

    candidates = {k: v for k, v in results.items() if REF not in k and v.get("n", 0) >= 100}
    show = passed or ([max(candidates, key=lambda k: candidates[k]["edge_t"])] if candidates else [])
    for name in show:
        tf = name.split("|")[0]
        print(f"\n[{name}] 按年份{'（通过筛选）' if passed else '（未通过，仅展示 t 值最高的组合）'}")
        print(edge_by_year(results[name]["records"], times[tf]).to_string(float_format=lambda v: f"{v:.3f}"))

    if passed:
        best = max(passed, key=lambda n: results[n]["net_r"])
        tf, use, method = best.split("|")
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_support_resistance.py {tf} {use} {method}")
    else:
        print("\n没有任何组合通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
