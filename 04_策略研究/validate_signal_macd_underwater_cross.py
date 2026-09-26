"""
validate_signal_macd_underwater_cross.py
==========================================
指南 4.4 节"信号验证三件套"，针对 00_方案/hypothesis_macd_underwater_cross.md 的假设。

三件套 1：命中率与平均收益（MFE 口径：未来窗口内最高价 - 入场价）
三件套 2：信号稳定性（按滚动窗口算 IC，看是否所有 regime 都有效还是只在个别时段有效）
三件套 3（信号相互独立性）：本假设只有 1 个信号，不适用，跳过。

用法：
    python 04_策略研究/validate_signal_macd_underwater_cross.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_策略研究"))
from signal_macd_underwater_cross import build_m5_with_signal  # noqa: E402

TARGET_USD = 10.0       # 假设里的反弹目标
FORWARD_BARS = 6        # 30 分钟 / M5 = 6 根
MIN_HIT_RATE = 0.50     # 可证伪阈值
MIN_AVG_MFE = 5.0       # 可证伪阈值（假设里"5块只是直觉"）


def evaluate_hit_rate(df: pd.DataFrame) -> dict:
    """三件套 1：命中率与平均收益（MFE/MAE 口径）"""
    signal_positions = np.flatnonzero(df["signal"].values == 1)

    records = []
    for pos in signal_positions:
        if pos + FORWARD_BARS >= len(df):
            continue  # 窗口不完整（数据末尾），跳过，不用不完整的未来数据凑数
        entry_price = df["close"].iloc[pos]
        window = df.iloc[pos + 1: pos + 1 + FORWARD_BARS]
        mfe = window["high"].max() - entry_price
        mae = window["low"].min() - entry_price
        records.append({
            "signal_time": df["time_utc"].iloc[pos],
            "entry_price": entry_price,
            "mfe": mfe,
            "mae": mae,
            "hit": mfe >= TARGET_USD,
        })

    if not records:
        return {"n_signals": 0}

    r = pd.DataFrame(records)
    sharpe_proxy = r["mfe"].mean() / r["mfe"].std() if r["mfe"].std() > 0 else float("nan")

    return {
        "n_signals": len(r),
        "hit_rate": r["hit"].mean(),
        "avg_mfe": r["mfe"].mean(),
        "avg_mae": r["mae"].mean(),
        "std_mfe": r["mfe"].std(),
        "sharpe_proxy": sharpe_proxy,
        "records": r,
    }


def evaluate_stability(df: pd.DataFrame, regime_window: int = 2000) -> pd.DataFrame:
    """三件套 2：信号稳定性——按滚动 regime 窗口算 IC（信号 vs 未来 MFE% 的 spearman 相关）"""
    entry_price = df["close"]
    forward_high = df["high"].shift(-1).rolling(FORWARD_BARS).max().shift(-(FORWARD_BARS - 1))
    forward_mfe_pct = (forward_high - entry_price) / entry_price

    ic_records = []
    step = regime_window // 2
    for start in range(0, len(df) - regime_window - FORWARD_BARS, step):
        end = start + regime_window
        regime_signal = df["signal"].iloc[start:end]
        regime_ret = forward_mfe_pct.iloc[start:end]

        if regime_signal.sum() < 5:  # 该窗口触发次数太少，IC 没有统计意义
            continue

        valid = regime_signal.notna() & regime_ret.notna()
        if valid.sum() < 30:
            continue

        ic, _ = spearmanr(regime_signal[valid], regime_ret[valid])
        ic_records.append({
            "start": df["time_utc"].iloc[start],
            "end": df["time_utc"].iloc[end - 1],
            "n_signals_in_window": int(regime_signal.sum()),
            "ic": ic,
        })

    return pd.DataFrame(ic_records)


def main() -> None:
    print("[构建] 加载 M1 -> 重采样 M5 -> 计算 MACD -> 生成信号 ...")
    df = build_m5_with_signal()
    print(f"[构建] M5 共 {len(df)} 根，触发信号 {int(df['signal'].sum())} 次\n")

    print("=" * 60)
    print("三件套 1：命中率与平均收益")
    print("=" * 60)
    hit_report = evaluate_hit_rate(df)
    if hit_report["n_signals"] == 0:
        print("没有任何信号触发，无法验证。检查信号定义/参数是否过严。")
        return

    print(f"信号次数: {hit_report['n_signals']}")
    print(f"命中率 (未来30分钟MFE>=${TARGET_USD}): {hit_report['hit_rate']*100:.1f}%")
    print(f"平均 MFE: ${hit_report['avg_mfe']:.2f} (std ${hit_report['std_mfe']:.2f})")
    print(f"平均 MAE: ${hit_report['avg_mae']:.2f}")
    print(f"逐笔 Sharpe 近似 (avg_mfe/std_mfe): {hit_report['sharpe_proxy']:.2f}")

    verdict_pass = hit_report["hit_rate"] >= MIN_HIT_RATE and hit_report["avg_mfe"] >= MIN_AVG_MFE
    print(f"\n[假设可证伪判定] 命中率>={MIN_HIT_RATE*100:.0f}% 且 平均MFE>=${MIN_AVG_MFE}: "
          + ("✅ 通过，可以继续往下（信号稳定性/进 03_回测引擎）" if verdict_pass
             else "❌ 不通过，按假设第3段应该放弃或修改这个假设"))

    print("\n" + "=" * 60)
    print("三件套 2：信号稳定性（滚动 regime IC）")
    print("=" * 60)
    stability = evaluate_stability(df)
    if stability.empty:
        print("信号太稀疏，没有足够样本量做滚动窗口IC检验（考虑放宽参数或扩大regime_window）")
    else:
        print(stability.to_string(index=False))
        healthy_ratio = (stability["ic"] > 0.05).mean()
        print(f"\nIC > 0.05 的窗口占比: {healthy_ratio*100:.0f}%")
        if healthy_ratio > 0.5:
            print("[✅] 信号在大多数 regime 下都有正向预测力，相对稳定")
        else:
            print("[⚠️] 信号可能只在个别 regime 有效，存在 regime 依赖，需要在第5章做多 Regime 验证")

    print("\n三件套 3（信号相互独立性）：本假设只有 1 个信号，跳过。")


if __name__ == "__main__":
    main()
