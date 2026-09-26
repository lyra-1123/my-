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


def baseline_hit_rate(df: pd.DataFrame, target_usd, forward_bars: int, direction: int = 1) -> dict:
    """
    对照组：不看任何信号，随便一根 bar 作为"入场点"，未来 forward_bars 内 MFE>=target_usd 的比例。
    信号的命中率必须显著高于这个基线，否则说明信号毫无增益（甚至可能比瞎猜还差）。
    direction=1 对应"随便做多"基线，direction=-1 对应"随便做空"基线——双向信号应该
    分别和同方向的基线比较（多头信号比多头基线，空头信号比空头基线），而不是混在一起。
    target_usd 可以是固定数字，也可以是逐 bar 不同的 pd.Series（比如按当前 ATR 定目标）——
    这样"随便一根bar"的基线也会用它自己那根bar的目标，跟波动率相关的信号做公平对比，
    不会因为信号本身就在筛选高波动时段而虚高。
    """
    mfe, mae = compute_forward_mfe_mae(df, forward_bars)
    if direction < 0:
        mfe, mae = -mae, -mfe  # 做空：MFE=entry-low=-mae(做多口径)，MAE=entry-high=-mfe(做多口径)
    target = target_usd.reindex(mfe.index) if isinstance(target_usd, pd.Series) else target_usd
    valid = mfe.notna() & (target.notna() if isinstance(target, pd.Series) else True)
    return {
        "hit_rate": (mfe[valid] >= (target[valid] if isinstance(target, pd.Series) else target)).mean(),
        "avg_mfe": mfe[valid].mean(),
        "avg_mae": mae[valid].mean(),
        "n_bars": int(valid.sum()),
    }


def evaluate_hit_rate(df: pd.DataFrame, target_usd, forward_bars: int) -> dict:
    """
    三件套 1：命中率与平均收益（MFE/MAE 口径：未来窗口内最高/最低价 - 入场价）。
    支持双向信号：df["signal"] 可以是 0/1（只做多，向后兼容）或 -1/0/1（做空/无/做多）。
    做多的 MFE = 未来窗口最高价-入场价；做空的 MFE = 入场价-未来窗口最低价（方向对齐后都是"越大越好"）。
    target_usd 可以是固定数字，也可以是 pd.Series（逐信号取入场那一刻的值，比如当前ATR）。
    """
    signal_values = df["signal"].values
    signal_positions = np.flatnonzero(signal_values != 0)
    target_is_series = isinstance(target_usd, pd.Series)

    records = []
    for pos in signal_positions:
        if pos + forward_bars >= len(df):
            continue  # 窗口不完整（数据末尾），跳过，不用不完整的未来数据凑数
        direction = 1 if signal_values[pos] > 0 else -1
        entry_price = df["close"].iloc[pos]
        window = df.iloc[pos + 1: pos + 1 + forward_bars]
        if direction > 0:
            mfe = window["high"].max() - entry_price
            mae = window["low"].min() - entry_price
        else:
            mfe = entry_price - window["low"].min()
            mae = entry_price - window["high"].max()
        target = target_usd.iloc[pos] if target_is_series else target_usd
        if target_is_series and pd.isna(target):
            continue  # ATR 还没热身（比如数据最开头），跳过
        records.append({
            "signal_time": df["time_utc"].iloc[pos],
            "direction": direction,
            "entry_price": entry_price,
            "mfe": mfe,
            "mae": mae,
            "target": target,
            "hit": mfe >= target,
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


def evaluate_stability(df: pd.DataFrame, forward_bars: int, regime_window: int = 2000,
                        return_type: str = "mfe") -> pd.DataFrame:
    """
    三件套 2：信号稳定性——按滚动 regime 窗口算 IC（信号 vs 未来收益的 spearman 相关）。
    return_type="mfe"（默认，向后兼容假设1/2/3）：只看多头视角的未来最大有利偏移百分比，
        适用于 signal 只有 0/1（只做多）的场景。
    return_type="close"：未来收盘价的原始涨跌百分比（可正可负），适用于 signal 是 -1/0/1
        的双向信号——signal为+1时预测正收益、-1时预测负收益，用spearman相关直接检验
        "信号方向"和"未来涨跌方向"是否一致，天然支持双向。
    """
    entry_price = df["close"]
    if return_type == "close":
        forward_ret = (df["close"].shift(-forward_bars) - entry_price) / entry_price
    else:
        forward_high = df["high"].shift(-1).rolling(forward_bars).max().shift(-(forward_bars - 1))
        forward_ret = (forward_high - entry_price) / entry_price
    forward_mfe_pct = forward_ret

    ic_records = []
    step = regime_window // 2
    for start in range(0, len(df) - regime_window - forward_bars, step):
        end = start + regime_window
        regime_signal = df["signal"].iloc[start:end]
        regime_ret = forward_mfe_pct.iloc[start:end]

        n_triggers = int((regime_signal != 0).sum())  # 用非零计数，双向信号(-1/+1)求和会互相抵消
        if n_triggers < 5:  # 该窗口触发次数太少，IC 没有统计意义
            continue

        valid = regime_signal.notna() & regime_ret.notna()
        if valid.sum() < 30:
            continue

        ic, _ = spearmanr(regime_signal[valid], regime_ret[valid])
        ic_records.append({
            "start": df["time_utc"].iloc[start],
            "end": df["time_utc"].iloc[end - 1],
            "n_signals_in_window": n_triggers,
            "ic": ic,
        })

    return pd.DataFrame(ic_records)


def print_report(hit_report: dict, stability: pd.DataFrame, target_usd,
                  min_hit_rate: float, min_avg_mfe: float, baseline: dict | None = None,
                  target_label: str | None = None) -> None:
    print("=" * 60)
    print("三件套 1：命中率与平均收益")
    print("=" * 60)
    if hit_report["n_signals"] == 0:
        print("没有任何信号触发，无法验证。检查信号定义/参数是否过严。")
        return

    label = target_label if target_label is not None else f"${target_usd}"
    print(f"信号次数: {hit_report['n_signals']}")
    print(f"命中率 (未来窗口MFE>={label}): {hit_report['hit_rate']*100:.1f}%")
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
