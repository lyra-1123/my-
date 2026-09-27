"""
validate_signal_double_top_bottom.py
=======================================
假设6（双顶/双底）第4章筛选：M5 / M15 / H1 × 3 种出场，同一批信号bar上比较信号方向与随机方向
（反方向用同样的止损/止盈距离镜像）。另打印最近几个识别出的形态，供在 MT5 图上人工核对定义。

用法：
    python 04_策略研究/validate_signal_double_top_bottom.py
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
from signal_double_top_bottom import (EXIT_RULES, TIMEFRAMES, build_features, describe_patterns,  # noqa: E402
                                      load_m1, load_tf)


def main() -> None:
    cost = SimpleCostModel().round_trip_cost()
    print(f"[加载] M1 ... 成本 ${cost:.2f}/笔")
    m1 = load_m1()

    results, feats = {}, {}
    for tf in TIMEFRAMES:
        feat, patterns = build_features(load_tf(m1, tf))
        feats[tf] = feat
        s = feat["signal"]
        sig_rows = feat[s != 0]
        stop_atr = (sig_rows["stop_dist"] / sig_rows["atr"]).median()
        tgt_r = (sig_rows["target_dist"] / sig_rows["stop_dist"]).median()
        print(f"\n[{tf}] {len(feat)} 根，双底 {int((s == 1).sum())} 个，双顶 {int((s == -1).sum())} 个；"
              f"止损距离中位数 {stop_atr:.2f}×ATR，形态目标中位数 {tgt_r:.2f}R，"
              f"止损中位数 ${sig_rows['stop_dist'].median():.2f}（成本约 {cost / sig_rows['stop_dist'].median():.3f}R）")
        print(f"[{tf}] 最近 5 个形态（UTC 时间，bid 价），请在 MT5 图上抽查：")
        print(describe_patterns(feat, patterns).to_string(index=False))

        market = prepare_market(feat, stop_dist=feat["stop_dist"], target_dist=feat["target_dist"])
        for rule in EXIT_RULES:
            results[f"{tf}|{rule.name}"] = evaluate_exit_direction(market, feat["signal"], rule, cost)
    print()

    passed = print_exit_direction_table(results)

    for name, res in results.items():
        if not res.get("n"):
            continue
        rec = res["records"]
        parts = []
        for d, label in [(1, "双底做多"), (-1, "双顶做空")]:
            sub = rec[rec["direction"] == d]
            if len(sub):
                parts.append(f"{label} n={len(sub)} 边际{sub['edge'].mean():+.3f}R 成本后{(sub['r'] - sub['cost_r']).mean():+.3f}R")
        print(f"  {name}: " + "；".join(parts))

    show = passed or [max(results, key=lambda k: results[k].get("edge_t", float("-inf")))]
    for name in show:
        tf = name.split("|")[0]
        print(f"\n[{name}] 按年份{'（通过筛选）' if passed else '（未通过，仅展示 t 值最高的组合）'}")
        print(edge_by_year(results[name]["records"], feats[tf]["time_utc"])
              .to_string(float_format=lambda v: f"{v:.3f}"))

    if passed:
        best = max(passed, key=lambda n: results[n]["net_r"])
        tf, rule = best.split("|")
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_double_top_bottom.py {tf} {rule}")
    else:
        print("\n没有任何组合通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
