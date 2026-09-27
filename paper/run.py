# -*- coding: utf-8 -*-
"""
每日运行：python -m paper.run [--data-dir data]

对每个 status="active" 的策略：
  1) 读取最新数据（丢弃未走完的 K 线），重算信号与仓位；
  2) 校验行为指纹（与登记时一致，否则报错：说明代码改动影响了在跑策略）；
  3) 输出下一根 K 线开盘的操作，并写 state/<id>/signal.json、trades.csv、daily.csv（仅前向部分）。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from paper.engine import STATE, compute, daily_from, fingerprint, load_bars, next_action, trades_from
from paper.specs import SPECS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    cache, rows = {}, []
    for spec in [s for s in SPECS if s.status == "active"]:
        pre_path = os.path.join(STATE, spec.id, "prereg.json")
        if not os.path.exists(pre_path):
            raise SystemExit(f"{spec.id} 未登记：先运行 python -m paper.onboard {spec.id} --register")
        pre = json.load(open(pre_path, encoding="utf-8"))
        if spec.freq not in cache:
            cache[spec.freq] = load_bars(args.data_dir, spec.freq)
        bars, last_m1 = cache[spec.freq]
        fp = fingerprint(spec, bars)
        if fp != pre["fingerprint"]:
            raise SystemExit(f"❌ {spec.id} 行为指纹变化（登记 {pre['fingerprint']}，当前 {fp}）：代码改动影响了在跑策略。"
                             "请撤销改动，或把策略升级为新版本重新登记。")
        c = compute(spec, bars)
        fwd = c.loc[spec.forward_start:]
        d = daily_from(fwd)
        tr = trades_from(fwd, spec.lots)
        sig = next_action(c, spec, last_m1)
        stale_h = (now - last_m1.tz_localize("UTC")).total_seconds() / 3600
        sig.update({"run_at_utc": now.isoformat(timespec="seconds"), "data_age_hours": round(stale_h, 1),
                    "forward_days": int(len(d)), "forward_cum_usd": round(float(d["net_usd"].sum()), 2) if len(d) else 0.0,
                    "forward_cum_R": round(float(d["net_R"].sum()), 3) if len(d) else 0.0, "fingerprint_ok": True})
        out = os.path.join(STATE, spec.id)
        json.dump(sig, open(os.path.join(out, "signal.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        tr.to_csv(os.path.join(out, "trades.csv"), index=False)
        d.to_csv(os.path.join(out, "daily.csv"))
        rows.append(sig)

    if not rows:
        print("没有 active 策略")
        return
    t = pd.DataFrame(rows)[["strategy", "bar_close_utc", "z", "current_position", "target_position",
                            "action_at_next_open", "forward_days", "forward_cum_usd", "forward_cum_R", "data_age_hours"]]
    print(t.to_string(index=False))
    old = [r["strategy"] for r in rows if r["data_age_hours"] > 72]
    if old:
        print(f"\n⚠️ 数据超过 72 小时未更新：{old}。先运行 Dukascopy 导出脚本并更新 data/（或 scripts/fetch_data.sh）")


if __name__ == "__main__":
    main()
