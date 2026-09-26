"""
validate_signal_macd_underwater_cross.py
==========================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_macd_underwater_cross.md 的假设。
三件套的通用实现见 lib/goldq/signal_validation.py。

用法：
    python 04_策略研究/validate_signal_macd_underwater_cross.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from goldq.signal_validation import evaluate_hit_rate, evaluate_stability, print_report  # noqa: E402
from signal_macd_underwater_cross import build_m5_with_signal  # noqa: E402

TARGET_USD = 10.0       # 假设里的反弹目标
FORWARD_BARS = 6        # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50     # 可证伪阈值
MIN_AVG_MFE = 5.0       # 可证伪阈值（假设里"5块只是直觉"）


def main() -> None:
    print("[构建] 加载 M1 -> 重采样 M5 -> 计算 MACD -> 生成信号 ...")
    df = build_m5_with_signal()
    print(f"[构建] M5 共 {len(df)} 根，触发信号 {int(df['signal'].sum())} 次\n")

    hit_report = evaluate_hit_rate(df, target_usd=TARGET_USD, forward_bars=FORWARD_BARS)
    stability = evaluate_stability(df, forward_bars=FORWARD_BARS) if hit_report["n_signals"] else pd.DataFrame()
    print_report(hit_report, stability, TARGET_USD, MIN_HIT_RATE, MIN_AVG_MFE)


if __name__ == "__main__":
    main()
