"""
指南 4.4 节"信号验证三件套"的通用实现，被各个 04_策略研究/validate_signal_*.py 复用。

三件套 1：命中率与平均收益（MFE/MAE 口径）
三件套 2：信号稳定性（滚动 regime IC）
三件套 3（信号相互独立性）：多信号场景才需要，本模块不提供（各自按需实现）。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_forward_mfe_mae(df: pd.DataFrame, forward_bars: int) -> tuple[pd.Series, pd.Series]:
    """未来 forward_bars 根（不含当前bar）内的 MFE/MAE（美元），向量化，供逐 bar 基线对比用。"""
    forward_high = df["high"].shift(-1).rolling(forward_bars).max().shift(-(forward_bars - 1))
    forward_low = df["low"].shift(-1).rolling(forward_bars).min().shift(-(forward_bars - 1))
    mfe = forward_high - df["close"]
    mae = forward_low - df["close"]
    return mfe, mae


def baseline_hit_rate(df: pd.DataFrame, target_usd: float, forward_bars: int) -> dict:
    """
    对照组：不看任何信号，随便一根 bar 作为"入场点"，未来 forward_bars 内 MFE>=target_usd 的比例。
    信号的命中率必须显著高于这个基线，否则说明信号毫无增益（甚至可能比瞎猜还差）。
    """
    mfe, mae = compute_forward_mfe_mae(df, forward_bars)
    valid = mfe.notna()
    return {
        "hit_rate": (mfe[valid] >= target_usd).mean(),
        "avg_mfe": mfe[valid].mean(),
        "avg_mae": mae[valid].mean(),
        "n_bars": int(valid.sum()),
    }


def evaluate_hit_rate(df: pd.DataFrame, target_usd: float, forward_bars: int) -> dict:
    """三件套 1：命中率与平均收益（MFE/MAE 口径：未来窗口内最高/最低价 - 入场价）"""
    signal_positions = np.flatnonzero(df["signal"].values == 1)

    records = []
    for pos in signal_positions:
        if pos + forward_bars >= len(df):
            continue  # 窗口不完整（数据末尾），跳过，不用不完整的未来数据凑数
        entry_price = df["close"].iloc[pos]
        window = df.iloc[pos + 1: pos + 1 + forward_bars]
        mfe = window["high"].max() - entry_price
        mae = window["low"].min() - entry_price
        records.append({
            "signal_time": df["time_utc"].iloc[pos],
            "entry_price": entry_price,
            "mfe": mfe,
            "mae": mae,
            "hit": mfe >= target_usd,
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


def evaluate_stability(df: pd.DataFrame, forward_bars: int, regime_window: int = 2000) -> pd.DataFrame:
    """三件套 2：信号稳定性——按滚动 regime 窗口算 IC（信号 vs 未来 MFE% 的 spearman 相关）"""
    entry_price = df["close"]
    forward_high = df["high"].shift(-1).rolling(forward_bars).max().shift(-(forward_bars - 1))
    forward_mfe_pct = (forward_high - entry_price) / entry_price

    ic_records = []
    step = regime_window // 2
    for start in range(0, len(df) - regime_window - forward_bars, step):
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


def print_report(hit_report: dict, stability: pd.DataFrame, target_usd: float,
                  min_hit_rate: float, min_avg_mfe: float, baseline: dict | None = None) -> None:
    print("=" * 60)
    print("三件套 1：命中率与平均收益")
    print("=" * 60)
    if hit_report["n_signals"] == 0:
        print("没有任何信号触发，无法验证。检查信号定义/参数是否过严。")
        return

    print(f"信号次数: {hit_report['n_signals']}")
    print(f"命中率 (未来窗口MFE>=${target_usd}): {hit_report['hit_rate']*100:.1f}%")
    print(f"平均 MFE: ${hit_report['avg_mfe']:.2f} (std ${hit_report['std_mfe']:.2f})")
    print(f"平均 MAE: ${hit_report['avg_mae']:.2f}")
    print(f"逐笔 Sharpe 近似 (avg_mfe/std_mfe): {hit_report['sharpe_proxy']:.2f}")

    if baseline is not None:
        lift = hit_report["hit_rate"] - baseline["hit_rate"]
        print(f"\n[对照组] 不看信号、随便一根bar的基线命中率: {baseline['hit_rate']*100:.1f}% "
              f"(基线平均MFE ${baseline['avg_mfe']:.2f}，基于 {baseline['n_bars']} 根bar)")
        print(f"[增益] 信号命中率 - 基线命中率 = {lift*100:+.1f} 个百分点"
              + ("  ⚠️ 信号并不比随便找个时间点更好，说明信号没有实际增益"
                 if lift <= 0 else ""))

    verdict_pass = hit_report["hit_rate"] >= min_hit_rate and hit_report["avg_mfe"] >= min_avg_mfe
    print(f"\n[假设可证伪判定] 命中率>={min_hit_rate*100:.0f}% 且 平均MFE>=${min_avg_mfe}: "
          + ("✅ 通过，可以继续往下（信号稳定性/进 03_回测引擎）" if verdict_pass
             else "❌ 不通过，按假设第3段应该放弃或修改这个假设"))

    print("\n" + "=" * 60)
    print("三件套 2：信号稳定性（滚动 regime IC）")
    print("=" * 60)
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

    print("\n三件套 3（信号相互独立性）：单信号场景不适用，跳过。")
