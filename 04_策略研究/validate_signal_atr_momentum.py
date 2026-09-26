"""
validate_signal_atr_momentum.py
==================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_atr_momentum.md 的假设（双向信号）。

用法：
    python 04_策略研究/validate_signal_atr_momentum.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from goldq.signal_validation import evaluate_hit_rate, evaluate_stability, print_report, baseline_hit_rate  # noqa: E402
from signal_atr_momentum import generate_signal  # noqa: E402

TARGET_USD = 3.0        # 假设里的反弹/下跌目标
FORWARD_BARS = 6        # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50
MIN_AVG_MFE = 5.0       # 用户原话："5块只是直觉"——比目标金额$3更高的独立可证伪标准


def blended_baseline(df: pd.DataFrame, n_long: int, n_short: int) -> dict:
    """双向信号要跟同方向的基线比，再按多空信号数量加权平均成一个综合基线。"""
    long_base = baseline_hit_rate(df, TARGET_USD, FORWARD_BARS, direction=1)
    short_base = baseline_hit_rate(df, TARGET_USD, FORWARD_BARS, direction=-1)
    total = n_long + n_short
    if total == 0:
        return long_base
    w_long, w_short = n_long / total, n_short / total
    return {
        "hit_rate": w_long * long_base["hit_rate"] + w_short * short_base["hit_rate"],
        "avg_mfe": w_long * long_base["avg_mfe"] + w_short * short_base["avg_mfe"],
        "avg_mae": w_long * long_base["avg_mae"] + w_short * short_base["avg_mae"],
        "n_bars": long_base["n_bars"],
    }


def main() -> None:
    print("[构建] 加载 M1 -> M5 -> EMA20/ATR14/MACD/RSI12 -> 生成双向信号 ...")
    df = generate_signal()
    n_long = int((df["signal"] == 1).sum())
    n_short = int((df["signal"] == -1).sum())
    print(f"[构建] M5 共 {len(df)} 根，做多信号 {n_long} 次，做空信号 {n_short} 次\n")

    hit_report = evaluate_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
    stability = evaluate_stability(df, forward_bars=FORWARD_BARS, return_type="close") \
        if hit_report["n_signals"] else pd.DataFrame()
    baseline = blended_baseline(df, n_long, n_short)
    print_report(hit_report, stability, TARGET_USD, MIN_HIT_RATE, MIN_AVG_MFE, baseline=baseline)

    if hit_report["n_signals"]:
        records = hit_report["records"]
        print("\n" + "=" * 60)
        print("按方向拆分（多头 vs 空头）")
        print("=" * 60)
        for direction, label in [(1, "多头"), (-1, "空头")]:
            sub = records[records["direction"] == direction]
            if sub.empty:
                print(f"{label}: 无信号")
                continue
            print(f"{label}: n={len(sub)}, 命中率={sub['hit'].mean()*100:.1f}%, "
                  f"平均MFE=${sub['mfe'].mean():.2f}, 平均MAE=${sub['mae'].mean():.2f}")


if __name__ == "__main__":
    main()
