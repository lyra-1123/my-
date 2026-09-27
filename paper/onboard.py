# -*- coding: utf-8 -*-
"""
策略准入与登记：python -m paper.onboard <策略id> [--register]

准入检查（新因子/新策略加入模拟盘前必须通过）：
  1) 证据齐全：spec.evidence 需包含 研究层 PBO、DSR（有效试验数口径）、样本外夏普 等（来自因子库评估与过拟合检验）
  2) 与在跑策略的相关性：2020 年后回测日度 R 收益相关系数 < 0.5
  3) 组合贡献：加入后等仓位组合的回测夏普不下降
--register：写入 state/<id>/prereg.json（行为指纹 + 预期区间 + 评估标准），已存在则拒绝覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from paper.engine import STATE, compute, daily_from, fingerprint, load_bars
from paper.specs import SPECS, get_spec

BACKTEST_FROM = "2020-01-01"
REQUIRED_EVIDENCE = ("cscv_oos_sharpe_median", "research_PBO")
WINDOWS = (21, 63, 126, 252)


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def backtest_daily(spec, data_dir, cache={}) -> pd.DataFrame:
    if spec.freq not in cache:
        cache[spec.freq] = load_bars(data_dir, spec.freq)
    bars, _ = cache[spec.freq]
    d = daily_from(compute(spec, bars))
    return d.loc[BACKTEST_FROM:pd.Timestamp(spec.forward_start) - pd.Timedelta("1D")]


def bands(d: pd.DataFrame) -> dict:
    out = {}
    for w in WINDOWS:
        roll = d["net_R"].rolling(w).sum().dropna()
        out[str(w)] = {q: round(float(roll.quantile(p)), 3) for q, p in (("p5", .05), ("p50", .5), ("p95", .95))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()
    spec = get_spec(args.sid)
    d = backtest_daily(spec, args.data_dir)
    ok = True

    print(f"== 准入检查：{spec.id}（{spec.freq}）  回测区间 {BACKTEST_FROM} ~ {spec.forward_start}")
    miss = [k for k in REQUIRED_EVIDENCE if k not in spec.evidence]
    print(f"[1] 证据：{spec.evidence}" + (f"  ❌ 缺少 {miss}" if miss else "  ✅"))
    ok &= not miss

    others = [s for s in SPECS if s.status == "active" and s.id != spec.id]
    if spec.status == "shadow":
        print(f"[2][3] 影子版本（对照 {spec.shadow_of}）：不计入组合，跳过相关性与组合贡献检查")
        others = []
    if others:
        R = pd.DataFrame({s.id: backtest_daily(s, args.data_dir)["net_R"] for s in others}).fillna(0)
        cand = d["net_R"].reindex(R.index).fillna(0)
        corr = R.corrwith(cand)
        print("[2] 与在跑策略的日度 R 相关：", corr.round(2).to_dict(), "✅" if (corr.abs() < 0.5).all() else "❌ ≥0.5")
        ok &= bool((corr.abs() < 0.5).all())
        before, after = sharpe(R.sum(axis=1)), sharpe(R.sum(axis=1) + cand)
        print(f"[3] 组合夏普（R 口径）：加入前 {before:.2f} → 加入后 {after:.2f}", "✅" if after >= before else "❌")
        ok &= after >= before
    elif spec.status != "shadow":
        print("[2][3] 目前没有在跑策略，跳过相关性与组合贡献检查")

    b = bands(d)
    print(f"[4] 回测（{BACKTEST_FROM} 起）日度 R 夏普 {sharpe(d['net_R']):.2f}，美元净利 {d['net_usd'].sum():+.0f}；"
          f"滚动窗口累计 R 区间：{b}")

    if not args.register:
        print("\n准入结论：", "通过" if ok else "未通过", "（加 --register 登记）")
        return
    if not ok:
        raise SystemExit("未通过准入检查，拒绝登记")
    path = os.path.join(STATE, spec.id, "prereg.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} 已存在：在跑策略的登记不可覆盖；如需修改请新增版本")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bars, _ = load_bars(args.data_dir, spec.freq)
    prereg = {
        "strategy": spec.id, "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forward_start": spec.forward_start, "spec": {k: v for k, v in spec.__dict__.items() if k != "evidence"},
        "evidence": spec.evidence, "fingerprint": fingerprint(spec, bars),
        "backtest_daily_R_sharpe": round(sharpe(d["net_R"]), 3), "bands_cum_R": b,
        "criteria": {
            "fail": "前向累计 R 低于同长度窗口回测分布的 5% 分位（在 63/126/252 个交易日检查）→ 停止该策略并复查",
            "warn": "前向累计 R 低于 50% 分位，或前向最大回撤超过回测最大回撤的 1.5 倍",
            "expectation": "日度 R 夏普基准取 CSCV 样本外中位数（evidence.cscv_oos_sharpe_median），而非全样本回测值",
            "min_evaluation": "252 个交易日之前不对'有效'下结论，只检查是否失败",
        },
        "backtest_max_drawdown_R": round(float((d["cum_R"] - d["cum_R"].cummax()).min()), 3),
    }
    if spec.status == "shadow":
        prereg["shadow_of"] = spec.shadow_of
        prereg["criteria"]["shadow_decision"] = (
            f"252 个交易日时比较前向结果：若本影子版本的前向日度 R 夏普高于 {spec.shadow_of}，且未触发自身失败线，"
            f"则以本规则建立 {spec.shadow_of} 的新版本（重新走准入与登记），否则停止本影子版本；252 日之前不做切换。")
    json.dump(prereg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    fills = os.path.join(STATE, spec.id, "fills_manual.csv")
    if not os.path.exists(fills):
        open(fills, "w", encoding="utf-8").write("time_utc,action,side,lots,price,note\n")
    print(f"\n已登记：{path}\n行为指纹 {prereg['fingerprint']}；实际成交可记录在 {fills}")


if __name__ == "__main__":
    main()
