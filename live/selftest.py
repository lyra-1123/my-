# -*- coding: utf-8 -*-
"""
实盘程序自测（不需要 MT5）：python -m live.selftest [--start 2026-08-24 --end 2026-09-25]

用 live/mock_mt5.py（行情 = data/cache 的 Dukascopy K 线，服务器时间 = 纽约 + 7 小时）模拟 MT5，
按真实节奏在每个 30 分钟收盘后 +8 秒运行 runner.cycle()，检查：
  1) 每次运行后各魔术号的实际持仓 == 用全部历史计算的模型目标仓位（paper.engine.compute）；
     影子 HA1H-TS2 的止损由 MT5 服务器（mock 按 1H 最高/最低价）触发，允许少量差异并单独列出；
  2) 服务器时间 → UTC 换算、未走完 K 线丢弃、同一根 K 线不重复处理；
  3) STOP 文件：全部平仓并不再开仓；
  4) 影子 HA1H-TS2 持仓时，服务器上的止损 == 模型的移动止损 next_stop（每根 1H K 线同步）；
  5) 终端显示允许交易、但服务器拒绝每一笔订单（模拟 09-28 的 10027）：写出告警文件；恢复后自动补齐仓位、告警解除。
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

os.environ["XAU_MT5_MOCK"] = "1"

import pandas as pd  # noqa: E402

from live import config as C  # noqa: E402

TMP = tempfile.mkdtemp(prefix="live_selftest_")
C.LOG_DIR = os.path.join(TMP, "logs")
C.KILL_FILE = os.path.join(TMP, "STOP")
C.RETRY_WAIT_SEC = 0

from live import broker, mock_mt5, mt5_data, runner  # noqa: E402
from paper.engine import compute  # noqa: E402
from paper.specs import get_spec  # noqa: E402

runner.STATE = os.path.join(C.LOG_DIR, "state.json")
mt5_data.utc_now_from_server = lambda: mock_mt5.NOW_UTC
runner.utc_now_from_server = lambda: mock_mt5.NOW_UTC


def model_targets(sid):
    spec = get_spec(sid)
    b = pd.read_pickle(f"data/cache/{spec.freq}.pkl")
    c = compute(spec, b)
    return c["target"], mt5_data.DUR[spec.freq], (c["next_stop"] if "next_stop" in c else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-24")
    ap.add_argument("--end", default="2026-09-25 21:00")
    ap.add_argument("--reject-from", default="2026-08-26 12:00", help="从此刻起服务器拒绝所有订单（10027）")
    ap.add_argument("--reject-to", default="2026-08-27 01:00")
    a = ap.parse_args()
    rej = (pd.Timestamp(a.reject_from), pd.Timestamp(a.reject_to))
    alert_seen, alert_after, stop_rows = False, None, []
    sids = [s for s, v in C.STRATEGIES.items() if v["enabled"]]
    tg = {s: model_targets(s) for s in sids}
    runner.preflight()
    state = runner.load_state()
    rows = []
    t0 = pd.Timestamp(a.start)
    mock_mt5.advance_to(t0)
    for b in pd.date_range(t0, pd.Timestamp(a.end), freq="30min"):
        now = b + pd.Timedelta(seconds=C.BAR_CLOSE_DELAY_SEC)
        mock_mt5.advance_to(now)
        rejecting = rej[0] <= now < rej[1]
        mock_mt5.REJECT_RETCODE[0] = 10027 if rejecting else None
        runner.cycle(state)
        if rejecting:
            alert_seen |= os.path.exists(runner.alert_file())
            continue   # 拒单期间仓位本来就会与模型脱节
        if alert_after is None and now >= rej[1]:
            alert_after = os.path.exists(runner.alert_file())
        for s in sids:
            tgt, dur, nstop = tg[s]
            done = tgt.index[tgt.index + dur <= now]
            if len(done) == 0:
                continue
            last = done[-1]
            if now - (last + dur) > pd.Timedelta(minutes=C.MAX_DATA_AGE_MIN):
                continue  # 休市：程序按设计不动
            want = round(float(tgt.loc[last]) * C.STRATEGIES[s]["lots"], 2)
            live = broker.net_lots(C.STRATEGIES[s]["magic"])
            rows.append((now, s, last, want, live))
            ps = broker.positions(C.STRATEGIES[s]["magic"])
            if nstop is not None and ps and want != 0 and abs(live - want) < 1e-9 and pd.notna(nstop.loc[last]):
                stop_rows.append((now, s, round(float(nstop.loc[last]), 2), ps[0].sl))
    df = pd.DataFrame(rows, columns=["now", "sid", "bar", "model", "live"])
    df["ok"] = (df["model"] - df["live"]).abs() < 1e-9
    print(f"区间 {a.start} → {a.end}，检查点 {len(df)}")
    fail = False
    for s, g in df.groupby("sid"):
        bad = g[~g["ok"]]
        trades = int((g["model"].diff().fillna(0) != 0).sum())
        print(f"  {s:18s} 检查 {len(g):5d}  模型换仓 {trades:4d}  不一致 {len(bad):3d}")
        if len(bad):
            print(bad.head(8).to_string(index=False))
        tol = 0.02 * len(g) if get_spec(s).rule != "standard" else 0
        fail |= len(bad) > tol

    st = pd.DataFrame(stop_rows, columns=["now", "sid", "model_stop", "server_sl"])
    st_bad = st[(st["model_stop"] - st["server_sl"]).abs() > 0.011]
    print(f"  移动止损：检查 {len(st)} 次，服务器止损 ≠ 模型止损 {len(st_bad)} 次")
    if len(st_bad):
        print(st_bad.head(8).to_string(index=False))
    fail |= len(st) == 0 or len(st_bad) > 0

    dec = pd.read_csv(os.path.join(C.LOG_DIR, "decisions.csv"))
    err = dec[dec["action"] == "ERROR"]
    dup = dec.duplicated(["strategy", "bar"]).sum()
    print(f"  决策记录 {len(dec)} 行，ERROR {len(err)}，同一根K线重复处理 {dup}")
    if len(err):
        print(err["note"].head().to_string())
    fail |= len(err) > 0 or dup > 0
    in_rej = pd.to_datetime(dec["utc_time"]).between(rej[0], rej[1] - pd.Timedelta(seconds=1))
    n_rej_fail = int((in_rej & (dec["action"] == "ORDER_FAILED")).sum())
    print(f"  拒单测试（{rej[0]} → {rej[1]}）：下单失败 {n_rej_fail} 次，告警文件出现 {alert_seen}，恢复后告警仍在 {alert_after}")
    fail |= n_rej_fail == 0 or not alert_seen or alert_after is not False

    # STOP 文件测试：从当前状态开始，先确保有持仓，再放 STOP
    open(C.KILL_FILE, "w").close()
    state["last_bar"] = {}
    runner.cycle(state, force=True)
    left = {s: broker.net_lots(C.STRATEGIES[s]["magic"]) for s in sids}
    print(f"  STOP 后持仓：{left}")
    fail |= any(v != 0 for v in left.values())
    shutil.rmtree(TMP, ignore_errors=True)
    print("自测结果：", "失败" if fail else "通过")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
