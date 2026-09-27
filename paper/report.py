# -*- coding: utf-8 -*-
"""
前向测试报告：python -m paper.report  → 打印并写入 paper/state/REPORT.md
对照登记时写死的预期区间判断每个策略的状态；汇总组合；若有 fills_manual.csv，计算实际成交相对模型的滑点。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from paper.engine import STATE
from paper.specs import SPECS


def status(cum_r: float, days: int, bands: dict) -> str:
    ws = [int(w) for w in bands if int(w) <= days]
    if not ws:
        return f"观察期（{days} 个交易日 < 21，不做判断）"
    w = str(max(ws)); b = bands[w]
    if days >= 63 and cum_r < b["p5"]:
        return f"❌ 失败：累计 R {cum_r:+.2f} < {w} 日窗口 5% 分位 {b['p5']:+.2f} → 停止并复查"
    if cum_r < b["p50"]:
        return f"⚠️ 偏弱：累计 R {cum_r:+.2f} 低于 {w} 日中位数 {b['p50']:+.2f}（5%~95%: {b['p5']:+.2f} ~ {b['p95']:+.2f}）"
    return f"✅ 正常：累计 R {cum_r:+.2f}，{w} 日区间 {b['p5']:+.2f} ~ {b['p95']:+.2f}（中位 {b['p50']:+.2f}）"


def main() -> None:
    lines = ["# 模拟盘前向测试报告", ""]
    port = {}
    for spec in [s for s in SPECS if s.status == "active"]:
        base = os.path.join(STATE, spec.id)
        pre = json.load(open(os.path.join(base, "prereg.json"), encoding="utf-8"))
        sig = json.load(open(os.path.join(base, "signal.json"), encoding="utf-8")) if os.path.exists(os.path.join(base, "signal.json")) else {}
        d = pd.read_csv(os.path.join(base, "daily.csv"), index_col=0, parse_dates=True) if os.path.exists(os.path.join(base, "daily.csv")) else pd.DataFrame()
        tr = pd.read_csv(os.path.join(base, "trades.csv")) if os.path.exists(os.path.join(base, "trades.csv")) else pd.DataFrame()
        days = len(d); cum_r = float(d["net_R"].sum()) if days else 0.0
        closed = tr[~tr["is_open"].astype(bool)] if len(tr) else tr
        mdd = float((d["cum_R"] - d["cum_R"].cummax()).min()) if days else 0.0
        lines += [f"## {spec.id}", "", f"- {spec.description}",
                  f"- 前向起始 {spec.forward_start}；已运行 {days} 个交易日；数据截至 {sig.get('last_m1_utc', '—')}",
                  f"- 当前仓位 {sig.get('current_position', '—')}，下一根开盘操作：**{sig.get('action_at_next_open', '—')}**（z={sig.get('z', '—')}）",
                  f"- 累计：{cum_r:+.3f} R，{(d['net_usd'].sum() if days else 0):+.2f}$；已平仓 {len(closed)} 笔，"
                  f"胜率 {((closed['net'] > 0).mean() if len(closed) else float('nan')):.2f}；前向最大回撤 {mdd:+.2f} R（回测 {pre['backtest_max_drawdown_R']:+.2f} R）",
                  f"- 状态：{status(cum_r, days, pre['bands_cum_R'])}"]
        if mdd < 1.5 * pre["backtest_max_drawdown_R"]:
            lines.append("- ⚠️ 前向回撤超过回测最大回撤的 1.5 倍")
        fills = os.path.join(base, "fills_manual.csv")
        if os.path.exists(fills):
            f = pd.read_csv(fills, parse_dates=["time_utc"])
            if len(f) and len(tr):
                ent = tr.assign(t=pd.to_datetime(tr["entry_time"]))
                m = pd.merge_asof(f.sort_values("time_utc"), ent.sort_values("t")[["t", "entry_open", "side"]],
                                  left_on="time_utc", right_on="t", direction="nearest", tolerance=pd.Timedelta("30min"))
                m["slip"] = np.where(m["side_y"] == "LONG", m["price"] - m["entry_open"], m["entry_open"] - m["price"])
                lines.append(f"- 实际成交 {len(f)} 笔；相对模型开盘价的平均不利滑点 {m['slip'].mean():+.3f}$（已含点差）")
        lines.append("")
        if days:
            port[spec.id] = d["net_R"]
    if len(port) > 1:
        P = pd.DataFrame(port).fillna(0)
        lines += ["## 组合", "", f"- 等仓位组合累计 R {P.sum(axis=1).sum():+.3f}",
                  "- 策略间日度 R 相关：", "```", P.corr().round(2).to_string(), "```"]
    txt = "\n".join(lines)
    open(os.path.join(STATE, "REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
