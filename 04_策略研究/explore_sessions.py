"""
explore_sessions.py
======================
时段效应描述统计（探索用，不是策略检验）。**只用探索期 2009-2019**，2020-2026 保留给之后的假设做
一次性验证，本脚本不读取保留期。

统计单位：每个交易日在该时间桶内的累计对数收益（M5 收盘到收盘，单位 bp = 0.01%）。
  平均   各交易日的均值；t = 均值 / (标准差/√天数)
  上涨天占比、平均|收益|（波动）
  同号年份  11 个年份里，年度均值与全期均值同号的比例（看是否每年都这样，而不是一两年撑起来）
  周末跳空（周五收盘到周日开盘）计入周一亚洲盘的第一根，所以亚洲盘/周一的数字包含周末跳空。

⚠️ 24 个小时 + 4 个时段 + 5 个星期几，纯随机下也会有 1-2 个 |t|>2。看"同号年份"是否稳定，
   不要只看 t 值；从这里挑出的任何规律，都必须写成假设、只在保留期上验证一次。

用法：
    python 04_策略研究/explore_sessions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.levels import trading_date  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from goldq.sessions import SESSIONS, london_time, session_label  # noqa: E402

EXPLORE_END = pd.Timestamp("2019-12-31")
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五"]


def load_exploration_bars() -> pd.DataFrame:
    df = resample_ohlcv(fetch_bars(symbol="XAUUSD"), rule="5min")
    df["td"] = trading_date(df["time_utc"])
    df = df[df["td"] <= EXPLORE_END].reset_index(drop=True)
    df["ret_bp"] = np.log(df["close"] / df["close"].shift(1)) * 1e4
    return df.iloc[1:]


def bucket_stats(daily: pd.DataFrame, bucket: str, order=None) -> pd.DataFrame:
    """daily: 每行一个 (交易日, 桶) 的累计收益，列 td / year / bucket / ret_bp。"""
    rows = []
    for key, g in daily.groupby(bucket, sort=False):
        r = g["ret_bp"]
        mean = r.mean()
        yearly = g.groupby("year")["ret_bp"].mean()
        rows.append({
            bucket: key, "天数": len(r), "平均(bp)": mean,
            "t": mean / (r.std() / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan,
            "上涨天占比": (r > 0).mean(), "平均|收益|(bp)": r.abs().mean(),
            "同号年份": (np.sign(yearly) == np.sign(mean)).mean(),
        })
    out = pd.DataFrame(rows).set_index(bucket)
    return out.reindex(order) if order is not None else out.sort_index()


def main() -> None:
    print(f"[加载] M5，只取探索期（交易日 <= {EXPLORE_END.date()}） ...")
    df = load_exploration_bars()
    df["year"] = df["td"].dt.year
    df["伦敦时间(时)"] = london_time(df["time_utc"]).dt.hour
    df["时段"] = session_label(df["time_utc"])
    df["星期"] = df["td"].dt.dayofweek.map(dict(enumerate(WEEKDAYS + ["周六", "周日"])))
    print(f"[加载] {len(df)} 根，{df['td'].min().date()} ~ {df['td'].max().date()}，"
          f"{df['td'].nunique()} 个交易日\n")

    fmt = {"平均(bp)": "{:+.2f}", "t": "{:+.2f}", "上涨天占比": "{:.1%}",
           "平均|收益|(bp)": "{:.1f}", "同号年份": "{:.0%}"}

    def show(title: str, table: pd.DataFrame) -> None:
        print("=" * 70 + f"\n{title}\n" + "=" * 70)
        t = table.copy()
        for col, f in fmt.items():
            t[col] = t[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
        print(t.to_string())
        print()

    for key, title, order in [
        ("伦敦时间(时)", "按伦敦当地时间的小时（该小时内的累计收益）", list(range(24))),
        ("时段", "按时段（亚洲盘 / 伦敦上午 / 伦敦纽约重叠 / 纽约下午）", SESSIONS),
        ("星期", "按星期几（整个交易日的收益）", WEEKDAYS),
    ]:
        daily = df.groupby(["td", "year", key], as_index=False)["ret_bp"].sum()
        show(title, bucket_stats(daily, key, order))

    total = df.groupby("td")["ret_bp"].sum()
    print(f"参照：探索期每个交易日平均收益 {total.mean():+.2f}bp（长期漂移），"
          f"日收益标准差 {total.std():.1f}bp")
    print("提示：34 个桶里纯随机也会出现 1-2 个 |t|>2；优先看'同号年份'是否 >= 80% 且 t 同时较大。")


if __name__ == "__main__":
    main()
