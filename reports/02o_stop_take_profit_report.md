# 止损止盈搜索报告(信号设计步骤3-2)

## 方法

每个候选用02n选出的自己的最优N，搜索25种止损止盈组合：不设止损(基线，只有持有到期这一种出场) + ATR倍数止损(0.5x/1x/1.5x/2x的ATR14) x 盈亏比(1:1/1.5:1/2:1) + 固定百分比止损(0.2%/0.3%/0.5%/1%) x 盈亏比(同上)。止盈距离=止损距离x盈亏比，同一根bar内先摸到止损位还是止盈位无法区分时按止损优先(保守假设)。入场=触发后下一根bar开盘价，出场=止损/止盈/持有到期三者中先发生的那个。

## 每个候选：不设止损 vs 最优止损止盈配置

| 候选 | N(根) | 无止损年化Sharpe | 最优止损类型 | 止损水平 | 盈亏比 | 最优配置年化Sharpe | 年触发 | 折数为正 | 止损止盈是否有帮助 |
|---|---|---|---|---|---|---|---|---|---|
| vol_of_vol_100 | 24 | +0.75 | pct | 0.003 | 1.5 | +0.92 | 640 | 5/5 | 是 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 192 | +0.38 | pct | 0.005 | 1.0 | +0.78 | 21 | 5/5 | 是 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 24 | +0.61 | pct | 0.005 | 1.0 | +0.76 | 21 | 5/5 | 是 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 6 | +0.51 | atr | 2.0 | 1.5 | +0.75 | 24 | 4/5 | 是 |
| bb_width_100 | 6 | +0.74 | none | nan | nan | +0.74 | 525 | 5/5 | 否 |
| parkinson_vol_60_pctrank6000 | 12 | +0.58 | pct | 0.003 | 1.5 | +0.65 | 745 | 4/5 | 是 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 24 | +0.65 | none | nan | nan | +0.65 | 24 | 4/5 | 否 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 192 | +0.56 | pct | 0.005 | 2.0 | +0.64 | 46 | 4/5 | 是 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 24 | +0.41 | pct | 0.003 | 2.0 | +0.63 | 35 | 5/5 | 是 |
| adx_50_pctrank24000 | 48 | +0.62 | none | nan | nan | +0.62 | 865 | 4/5 | 否 |
| garman_klass_vol_240 | 96 | +0.61 | none | nan | nan | +0.61 | 189 | 4/5 | 否 |
| garman_klass_vol_20_pctrank6000 | 6 | +0.42 | pct | 0.002 | 2.0 | +0.58 | 1526 | 4/5 | 是 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 12 | +0.35 | pct | 0.002 | 1.0 | +0.53 | 52 | 4/5 | 是 |
| avg_gap_50+roc_10 | 6 | +0.37 | pct | 0.002 | 1.5 | +0.50 | 57 | 4/5 | 是 |
| parkinson_vol_100_pctrank24000 | 12 | +0.36 | pct | 0.003 | 2.0 | +0.48 | 460 | 5/5 | 是 |
| bb_width_300_pctrank6000 | 24 | +0.38 | atr | 1.0 | 2.0 | +0.46 | 231 | 5/5 | 是 |
| realized_vol_100_pctrank2000+roc_10 | 12 | +0.39 | none | nan | nan | +0.39 | 38 | 4/5 | 否 |

**12/17个候选加止损止盈后比不设止损更好**（按walk-forward 年化OOS Sharpe比较，同一个N下）。

## 结论与下一步

- 完整数据：reports/02o_stop_take_profit_scan.csv；每候选最优配置：reports/02o_stop_take_profit_winners.csv。
- 下一步（步骤4）：多空分方向验证这批最终选定的止损止盈参数是否对多空都稳健。

