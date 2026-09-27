"""
chapter5_pipeline.py
======================
第5章"回测层：防御过拟合"的通用流水线。任何通过了第4章"验证三件套"的候选信号，
都通过 run_chapter5_validation() 走同一套检验：

  1. 回测引擎（backtest_engine.py）：真实进出场（SL/TP/HOLD）+ 交易成本
  2. Walk-Forward（wf_runner.py）：
       a. 固定参数滚动OOS：候选信号本身在每个OOS窗口的表现 + 连续通过次数
       b. WF选择：每个IS窗口从"变体家族"里选表现最好的，看它在下一个OOS窗口的表现
          ——模拟"研究员每次都挑回测最好看的那组参数"这个行为本身能不能泛化
  3. CSCV/PBO（cscv_pbo.py）：在变体家族里，IS最优的变体OOS跌到中位数以下的概率
  4. DSR：按"本项目累计验证过的策略定义总数"对候选Sharpe做多重检验折扣
  5. 多Regime（regime_split.py）：波动率/趋势两种切分下分别看表现
  6. judge_strategy()：指南5.3.4的5项综合判定

新假设接入方式：写一个 validate_ch5_<假设名>.py，构造 candidate + variants 调用本模块，
参考 validate_ch5_atr_momentum_v2.py。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_engine import calc_sharpe, run_backtest, summarize_trades
from cost_model import SimpleCostModel
from cscv_pbo import build_returns_matrix, calc_dsr, calc_pbo
from regime_split import multi_regime_validation, regime_by_trend, regime_by_volatility
from wf_runner import design_wf_windows, is_window_pass, rolling_wf_pass


@dataclass
class Variant:
    name: str
    signal: pd.Series
    target: pd.Series


def _slice(trades: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    if trades.empty:
        return trades
    mask = (trades["entry_idx"] >= start_idx) & (trades["entry_idx"] < end_idx)
    return trades[mask]


def judge_strategy(pbo: float, dsr: float, sharpe_oos_annual: float,
                   n_oos_trades: int, n_consecutive_pass: int) -> dict:
    """指南5.3.4：5项全过=PASS，>=4项=WARN，否则FAIL。"""
    checks = {
        "PBO < 25%": bool(pbo < 0.25),
        "DSR > 0.95": bool(dsr > 0.95),
        "OOS年化Sharpe > 1.0": bool(sharpe_oos_annual > 1.0),
        "OOS交易数 >= 300": bool(n_oos_trades >= 300),
        "连续通过OOS窗口 >= 6": bool(n_consecutive_pass >= 6),
    }
    n_pass = sum(checks.values())
    verdict = "PASS" if n_pass == len(checks) else ("WARN" if n_pass >= 4 else "FAIL")
    recommendation = {
        "PASS": "可进入 Paper Trading（先模拟盘1-3个月，见指南第7章）",
        "WARN": "修复未通过项后重新验证",
        "FAIL": "放弃此策略或重新设计假设",
    }[verdict]
    return {"checks": checks, "n_pass": n_pass, "n_total": len(checks),
            "verdict": verdict, "recommendation": recommendation}


def run_chapter5_validation(df: pd.DataFrame, candidate: Variant, variants: list[Variant],
                            forward_bars: int, n_trials_prior: int,
                            cost_model: SimpleCostModel | None = None,
                            min_oos_trades: int = 300, is_oos_ratio: float = 1.0,
                            cscv_splits: int = 16, min_oos_days: int = 30) -> dict:
    """
    n_trials_prior: 在跑本流水线之前，本项目已经正式验证过的策略定义数（含candidate本身）。
                    DSR 用的 n_trials = n_trials_prior + 变体家族里除candidate外的新变体数。
    """
    cost_model = cost_model or SimpleCostModel()
    all_variants = variants if any(v.name == candidate.name for v in variants) else [candidate] + variants

    trades_by_name = {v.name: run_backtest(df, v.signal, v.target, forward_bars, cost_model)
                      for v in all_variants}
    cand_trades = trades_by_name[candidate.name]
    if cand_trades.empty:
        raise ValueError("候选信号全样本回测没有任何成交，无法继续")

    total_years = (df["time_utc"].iloc[-1] - df["time_utc"].iloc[0]).days / 365.25
    trades_per_year = len(cand_trades) / total_years

    # 2. Walk-Forward
    windows, wf_cfg = design_wf_windows(df, len(cand_trades), min_oos_trades, is_oos_ratio, min_oos_days)

    fixed_rows, fixed_oos = [], []
    for w in windows:
        oos = _slice(cand_trades, w["oos_start_idx"], w["oos_end_idx"])
        fixed_oos.append(oos)
        fixed_rows.append({
            "oos_start": w["oos_start"].date(), "oos_end": w["oos_end"].date(),
            "n_trades": len(oos),
            "total_pnl": oos["pnl"].sum() if len(oos) else 0.0,
            "win_rate": (oos["pnl"] > 0).mean() if len(oos) else float("nan"),
            "passed": is_window_pass(oos),
        })
    consecutive = rolling_wf_pass([r["passed"] for r in fixed_rows])
    oos_all = pd.concat(fixed_oos) if fixed_oos else cand_trades.iloc[0:0]

    sel_rows, sel_oos = [], []
    for w in windows:
        is_pnl = {name: _slice(t, w["is_start_idx"], w["is_end_idx"])["pnl"].sum()
                  for name, t in trades_by_name.items()}
        best = max(is_pnl, key=is_pnl.get)
        oos = _slice(trades_by_name[best], w["oos_start_idx"], w["oos_end_idx"])
        sel_oos.append(oos)
        sel_rows.append({
            "oos_start": w["oos_start"].date(), "chosen_variant": best,
            "is_pnl": is_pnl[best],
            "oos_n": len(oos), "oos_pnl": oos["pnl"].sum() if len(oos) else 0.0,
        })
    sel_oos_all = pd.concat(sel_oos) if sel_oos else cand_trades.iloc[0:0]

    # 3. CSCV / PBO
    matrix = build_returns_matrix([trades_by_name[v.name] for v in all_variants], len(df))
    pbo = calc_pbo(matrix, cscv_splits)

    # 4. DSR
    n_trials = n_trials_prior + len(all_variants) - 1
    dsr = calc_dsr(cand_trades["return_pct"].to_numpy(), n_trials)

    # 5. 多 Regime
    regimes = {
        "波动率(ATR vs 100根均值)": multi_regime_validation(cand_trades, regime_by_volatility(df)),
        "趋势强度(ADX>25)": multi_regime_validation(cand_trades, regime_by_trend(df)),
    }

    # 6. 综合判定
    sharpe_oos_annual = calc_sharpe(oos_all["pnl"]) * np.sqrt(trades_per_year) if len(oos_all) > 1 else float("nan")
    verdict = judge_strategy(pbo.get("pbo", float("nan")), dsr.get("dsr_prob", float("nan")),
                             sharpe_oos_annual, len(oos_all), consecutive["max_consecutive"])

    return {
        "variant_summaries": {name: summarize_trades(t) for name, t in trades_by_name.items()},
        "candidate_full_sample": summarize_trades(cand_trades),
        "trades_per_year": trades_per_year,
        "wf_config": wf_cfg,
        "wf_fixed": pd.DataFrame(fixed_rows),
        "wf_consecutive": consecutive,
        "wf_fixed_oos_summary": summarize_trades(oos_all),
        "wf_selection": pd.DataFrame(sel_rows),
        "wf_selection_oos_summary": summarize_trades(sel_oos_all),
        "pbo": pbo,
        "dsr": dsr,
        "regimes": regimes,
        "sharpe_oos_annual": sharpe_oos_annual,
        "verdict": verdict,
    }


def print_chapter5_report(result: dict, candidate_name: str) -> None:
    line = "=" * 70

    print(f"\n{line}\n0. 变体家族全样本概览（仅供参考，不能作为判定依据——全样本就是被反复看过的那份数据）\n{line}")
    rows = []
    for name, s in result["variant_summaries"].items():
        rows.append({"variant": name, "n": s.get("n_trades", 0), "win_rate": s.get("win_rate"),
                     "gross_pnl": s.get("gross_pnl"), "total_pnl": s.get("total_pnl"),
                     "sharpe/笔": s.get("sharpe")})
    print(pd.DataFrame(rows).to_string(index=False))

    c = result["candidate_full_sample"]
    print(f"\n候选 [{candidate_name}] 全样本：{c['n_trades']} 笔，胜率 {c['win_rate']*100:.1f}%，"
          f"总PnL ${c['total_pnl']:.1f}/oz，逐笔Sharpe {c['sharpe']:.3f}，最大回撤 ${c['max_drawdown']:.1f}/oz")
    print(f"出场原因分布：{c['exit_reason_counts']}，年均成交 {result['trades_per_year']:.0f} 笔")
    print(f"成本前PnL ${c['gross_pnl']:.1f}/oz - 成本 ${c['total_cost']:.1f}/oz = 成本后 ${c['total_pnl']:.1f}/oz")
    print(f"同bar SL/TP双触发 {c['n_ambiguous']} 笔（已按SL记）；全部改判TP的乐观上界：${c['pnl_if_ambiguous_tp']:.1f}/oz")

    cfg = result["wf_config"]
    print(f"\n{line}\n1. Walk-Forward（固定参数滚动OOS）\n{line}")
    print(f"窗口设计：IS {cfg.is_days} 天 / OOS {cfg.oos_days} 天 / 步长 {cfg.step_days} 天，"
          f"共 {cfg.n_windows} 个窗口（按成交频率 {cfg.trades_per_day:.2f} 笔/天反推，保证每个OOS窗口预期≥{cfg.min_oos_trades}笔）")
    print(result["wf_fixed"].to_string(index=False))
    cons = result["wf_consecutive"]
    print(f"\n最长连续通过：{cons['max_consecutive']} 个窗口（门槛 {cons['min_required']}）"
          + (" ✅" if cons["passed"] else " ❌"))
    s = result["wf_fixed_oos_summary"]
    if s.get("n_trades"):
        print(f"全部OOS合计：{s['n_trades']} 笔，胜率 {s['win_rate']*100:.1f}%，总PnL ${s['total_pnl']:.1f}/oz，"
              f"年化Sharpe {result['sharpe_oos_annual']:.2f}")

    print(f"\n{line}\n2. Walk-Forward（每个IS窗口挑最好的变体，看它下一个OOS窗口）\n{line}")
    print(result["wf_selection"].to_string(index=False))
    s = result["wf_selection_oos_summary"]
    if s.get("n_trades"):
        print(f"WF选择法OOS合计：{s['n_trades']} 笔，胜率 {s['win_rate']*100:.1f}%，总PnL ${s['total_pnl']:.1f}/oz")

    p = result["pbo"]
    print(f"\n{line}\n3. CSCV / PBO（变体家族 {p.get('n_strategies')} 个，{p.get('n_splits')} 折，"
          f"{p.get('n_combinations')} 种IS/OOS组合）\n{line}")
    print(f"PBO = {p.get('pbo', float('nan'))*100:.1f}%  →  {p.get('interpretation', p.get('note'))}")
    if all(s.get("total_pnl", 0) <= 0 for s in result["variant_summaries"].values()):
        print("⚠️ 家族内所有变体全样本都不赚钱：PBO 只衡量相对排名是否稳定，此时低PBO只说明"
              "'亏得少的一直亏得少'，不代表存在正向边际")

    d = result["dsr"]
    print(f"\n{line}\n4. DSR（多重检验折扣，n_trials = {d.get('n_trials')}）\n{line}")
    if "sr_estimated" in d:
        print(f"逐笔Sharpe {d['sr_estimated']:.4f} vs 随机最优基准 {d['sr_benchmark']:.4f}，"
              f"DSR = {d['dsr_prob']:.3f}  →  {d['interpretation']}")
    else:
        print(d.get("note"))

    print(f"\n{line}\n5. 多 Regime 验证\n{line}")
    for split_name, report in result["regimes"].items():
        print(f"\n[{split_name}]")
        print(pd.DataFrame(report).T.to_string())

    v = result["verdict"]
    print(f"\n{line}\n6. 综合判定（指南5.3.4）\n{line}")
    for name, ok in v["checks"].items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n结论：{v['verdict']}（{v['n_pass']}/{v['n_total']}）  →  {v['recommendation']}")
