"""
validate_signal_pullback_second_leg_v2.py
============================================
假设9 v2 第4章筛选：{H1, H4} × 第一段 {5, 8}×ATR，同一次回调只交易一笔（dedupe），
大周期过滤 H1 看 H4、H4 看交易日。

默认（v2b）持仓过周末/假期，成本含隔夜利息（cost_model.py 的 swap_*）；
加 --weekend-flat 复现 v2（停盘前强制平仓，2026-09-27 已记录的那次结果）。

用法：
    python 04_策略研究/validate_signal_pullback_second_leg_v2.py [--weekend-flat]
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
from signal_pullback_second_leg import (V2_HTF_FOR as HTF_FOR, V2_LEG_ATR_GRID as LEG_ATR_GRID,  # noqa: E402
                                        V2_TIMEFRAMES as TIMEFRAMES, describe_patterns, exit_rule_for,
                                        load_features, load_m1, signals_for)


def market_for(feat, sig):
    return prepare_market(feat, stop_dist=sig["stop_dist"], target_dist=sig["target_dist"],
                          entry_price=sig["entry_price"], entry_hi=sig["entry_hi"], entry_lo=sig["entry_lo"])


def main() -> None:
    hold = "--weekend-flat" not in sys.argv
    cm = SimpleCostModel()
    cost = cm.round_trip_cost()
    print(f"[加载] M1 ... 成本 ${cost:.2f}/笔 + 隔夜利息 多${cm.swap_long_usd}/空${cm.swap_short_usd} 每盎司每晚，"
          f"出场 止损A + 第二段=第一段，同一次回调只做一笔，{'持仓过周末' if hold else '停盘前平仓（v2）'}")
    m1 = load_m1()

    results, times = {}, {}
    for tf in TIMEFRAMES:
        feat = load_features(m1, tf)
        times[tf] = feat["time_utc"]
        print(f"\n[{tf}] {len(feat)} 根，趋势过滤看 {HTF_FOR[tf]} EMA50")
        for leg in LEG_ATR_GRID:
            sig, pats = signals_for(feat, leg, dedupe=True)
            s = sig["signal"]
            rows = sig[s != 0]
            print(f"  第一段>={leg}ATR：做多 {int((s == 1).sum())}，做空 {int((s == -1).sum())}；"
                  f"止损中位数 ${rows['stop_dist'].median():.2f}（成本约 {cost / rows['stop_dist'].median():.3f}R），"
                  f"止盈/止损中位数 {(rows['target_dist'] / rows['stop_dist']).median():.2f}")
            if leg == LEG_ATR_GRID[0]:
                print(f"  [{tf} 第一段>={leg}ATR] 最近 5 个（UTC，bid 价），请在 MT5 图上核对：")
                print(describe_patterns(feat, pats).to_string(index=False))
            results[f"{tf}|leg{leg}"] = evaluate_exit_direction(market_for(feat, sig), s, exit_rule_for(tf, hold), cost,
                                                                cm.swap_long_usd, cm.swap_short_usd)
    print()

    passed = print_exit_direction_table(results)
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

    ok = {k: v for k, v in results.items() if v.get("n", 0) >= 100}
    show = passed or ([max(ok, key=lambda k: ok[k]["edge_t"])] if ok else [])
    for name in show:
        print(f"\n[{name}] 按年份{'（通过筛选）' if passed else '（未通过，仅展示 t 值最高的组合）'}")
        print(edge_by_year(results[name]["records"], times[name.split("|")[0]])
              .to_string(float_format=lambda v: f"{v:.3f}"))

    if passed:
        best = max(passed, key=lambda n: results[n]["net_r"])
        tf, leg = best.split("|")
        print(f"\n第5章候选：{best}。运行：\n  python 03_回测引擎\\validate_ch5_pullback_second_leg_v2.py {tf} {leg[3:]}")
    else:
        print("\n没有任何组合通过第4章方向性筛选，按流程不进第5章。")


if __name__ == "__main__":
    main()
