"""
validate_signal_mtf_golden_cross_cascade.py
=============================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_mtf_golden_cross_cascade.md 的假设。
三件套的通用实现见 lib/goldq/signal_validation.py。

用法：
    python 04_策略研究/validate_signal_mtf_golden_cross_cascade.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from goldq.signal_validation import evaluate_hit_rate, evaluate_stability, print_report, baseline_hit_rate  # noqa: E402
from signal_mtf_golden_cross_cascade import generate_signal  # noqa: E402

TARGET_USD = 10.0
FORWARD_BARS = 6        # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50
MIN_AVG_MFE = 5.0
MIN_SIGNALS_FOR_SIGNIFICANCE = 30  # 多重条件叠加，信号可能很稀疏，先检查样本量够不够


def main() -> None:
    print("[构建] 加载 M1 -> 1H/30m/15m/5m 多周期 -> 计算 MACD -> 生成信号 ...")
    df = generate_signal()
    n_signals = int(df["signal"].sum())
    print(f"[构建] M5 共 {len(df)} 根，触发信号 {n_signals} 次\n")

    if n_signals < MIN_SIGNALS_FOR_SIGNIFICANCE:
        print(f"[⚠️] 信号次数 {n_signals} < {MIN_SIGNALS_FOR_SIGNIFICANCE}，样本量太小，"
              f"命中率/IC 的结论统计上不可靠，仅供参考，不能直接下结论。")

    hit_report = evaluate_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
    stability = evaluate_stability(df, forward_bars=FORWARD_BARS) if hit_report["n_signals"] else pd.DataFrame()
    baseline = baseline_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
    print_report(hit_report, stability, TARGET_USD, MIN_HIT_RATE, MIN_AVG_MFE, baseline=baseline)


if __name__ == "__main__":
    main()
