"""
validate_signal_single_htf_golden_cross.py
=============================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_single_htf_golden_cross.md 的假设。
三个大周期候选（1H / 30min / 15min）都跑一遍，对比选出更合适的一个。

用法：
    python 04_策略研究/validate_signal_single_htf_golden_cross.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from goldq.signal_validation import evaluate_hit_rate, evaluate_stability, print_report, baseline_hit_rate  # noqa: E402
from signal_single_htf_golden_cross import generate_signal, HTF_CONFIG  # noqa: E402

TARGET_USD = 5.0         # 比假设1/2低，见假设3的操作定义
FORWARD_BARS = 6         # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50
MIN_AVG_MFE = 3.0        # 目标降到$5后，可证伪标准相应下调（见假设3第3段）


def main() -> None:
    summary = []

    for htf in HTF_CONFIG:
        print(f"\n{'#' * 60}\n大周期候选: {htf}\n{'#' * 60}")
        df = generate_signal(htf)
        n_signals = int(df["signal"].sum())
        print(f"[构建] M5 共 {len(df)} 根，触发信号 {n_signals} 次\n")

        hit_report = evaluate_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
        stability = evaluate_stability(df, forward_bars=FORWARD_BARS) if hit_report["n_signals"] else pd.DataFrame()
        baseline = baseline_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
        print_report(hit_report, stability, TARGET_USD, MIN_HIT_RATE, MIN_AVG_MFE, baseline=baseline)

        healthy_ic_ratio = (stability["ic"] > 0.05).mean() if not stability.empty else float("nan")
        summary.append({
            "htf": htf,
            "n_signals": hit_report.get("n_signals", 0),
            "hit_rate": hit_report.get("hit_rate", float("nan")),
            "avg_mfe": hit_report.get("avg_mfe", float("nan")),
            "baseline_hit_rate": baseline["hit_rate"],
            "lift_pp": (hit_report.get("hit_rate", float("nan")) - baseline["hit_rate"]) * 100,
            "ic_windows": len(stability),
            "healthy_ic_ratio": healthy_ic_ratio,
        })

    print(f"\n{'=' * 60}\n三个大周期候选对比汇总\n{'=' * 60}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
