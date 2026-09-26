"""
validate_signal_atr_momentum_v2.py
=====================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_atr_momentum_v2.md 的假设。
目标是"1倍当前ATR"（逐信号不同），不是固定金额；基线也用同样的ATR相对目标，
避免"筛选高波动时段"本身抬高固定美元目标基线这个假设4诊断出的混淆。

用法：
    python 04_策略研究/validate_signal_atr_momentum_v2.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from goldq.signal_validation import evaluate_hit_rate, evaluate_stability, print_report, baseline_hit_rate  # noqa: E402
from signal_atr_momentum_v2 import generate_signal, target_series, ATR_TARGET_MULTIPLIER  # noqa: E402

FORWARD_BARS = 6        # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50
MIN_MFE_TO_ATR_RATIO = 1.5  # 可证伪标准2：平均MFE/ATR比值 < 1.5 就放弃


def blended_baseline(df: pd.DataFrame, target: pd.Series, n_long: int, n_short: int) -> dict:
    long_base = baseline_hit_rate(df, target, FORWARD_BARS, direction=1)
    short_base = baseline_hit_rate(df, target, FORWARD_BARS, direction=-1)
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
    print("[构建] 加载 M1 -> M5 -> EMA20/ATR14/MACD/MFI12 -> 生成双向信号 ...")
    df = generate_signal()
    n_long = int((df["signal"] == 1).sum())
    n_short = int((df["signal"] == -1).sum())
    print(f"[构建] M5 共 {len(df)} 根，做多信号 {n_long} 次，做空信号 {n_short} 次\n")

    target = target_series(df)

    hit_report = evaluate_hit_rate(df, target_usd=target, forward_bars=FORWARD_BARS)
    stability = evaluate_stability(df, forward_bars=FORWARD_BARS, return_type="close") \
        if hit_report["n_signals"] else pd.DataFrame()
    baseline = blended_baseline(df, target, n_long, n_short)

    print_report(hit_report, stability, target, MIN_HIT_RATE, min_avg_mfe=0,
                 baseline=baseline, target_label=f"{ATR_TARGET_MULTIPLIER}倍当时ATR")

    if hit_report["n_signals"]:
        records = hit_report["records"]
        mfe_to_atr = (records["mfe"] / records["target"] * ATR_TARGET_MULTIPLIER)
        print(f"\n平均 MFE/ATR 比值: {mfe_to_atr.mean():.2f}")
        print(f"[假设可证伪判定2] 平均MFE/ATR比值>={MIN_MFE_TO_ATR_RATIO}: "
              + ("✅ 通过" if mfe_to_atr.mean() >= MIN_MFE_TO_ATR_RATIO else "❌ 不通过"))

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
