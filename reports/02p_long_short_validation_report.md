# 多空分方向止盈止损验证报告(信号设计步骤4)

## 方法

1. **诊断**：02o选出的共享止损止盈配置，分别只用多头触发/只用空头触发重新算一遍Sharpe，看这个共享配置是不是被某一个方向的表现拉高/拉低了整体数字。
2. **分方向搜索**：N固定用02o选出的值，止损止盈的25种配置分别只在多头/只在空头triggers上独立重新搜索一遍，各自选各自的最优配置；再把两个方向各自最优配置的真实交易收益合并，算一个"分方向优化后"的整体Sharpe，跟02o的"共享配置"整体Sharpe直接比较。

## 诊断：共享配置分方向表现

| 候选 | 多头Sharpe(未年化) | 多头折数为正 | 空头Sharpe(未年化) | 空头折数为正 |
|---|---|---|---|---|
| vol_of_vol_100 | +0.034 | 5/5 | +0.041 | 4/5 |
| adx_50_pctrank24000 | +0.036 | 4/5 | +0.001 | 4/5 |
| bb_width_100 | +0.044 | 5/5 | +0.018 | 5/5 |
| bb_width_300_pctrank6000 | +0.029 | 3/5 | +0.032 | 3/5 |
| parkinson_vol_100_pctrank24000 | +0.041 | 4/5 | -0.007 | 2/5 |
| garman_klass_vol_20_pctrank6000 | +0.025 | 5/5 | +0.003 | 2/5 |
| parkinson_vol_60_pctrank6000 | +0.008 | 4/5 | +0.045 | 4/5 |
| garman_klass_vol_240 | +0.087 | 5/5 | -0.018 | 2/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | +0.165 | 4/5 | +0.169 | 4/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | +0.133 | 4/5 | +nan | 0/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | +0.128 | 4/5 | -0.023 | 1/5 |
| realized_vol_100_pctrank2000+roc_10 | +0.106 | 4/5 | -0.002 | 3/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | +0.227 | 5/5 | +0.086 | 3/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | +0.076 | 4/5 | +0.071 | 3/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | +0.105 | 5/5 | +0.126 | 0/5 |
| avg_gap_50+roc_10 | +0.076 | 5/5 | -0.135 | 0/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | +0.067 | 3/5 | +0.267 | 4/5 |

## 分方向独立优化 vs 共享配置：整体年化Sharpe对比(口径一致，都已年化)

| 候选 | 共享配置年化Sharpe | 分方向优化后年化Sharpe | 分方向是否更好 |
|---|---|---|---|
| vol_of_vol_100 | +0.92 | +0.94 | 是 |
| adx_50_pctrank24000 | +0.62 | +0.74 | 是 |
| bb_width_100 | +0.74 | +0.90 | 是 |
| bb_width_300_pctrank6000 | +0.46 | +0.80 | 是 |
| parkinson_vol_100_pctrank24000 | +0.48 | +0.83 | 是 |
| garman_klass_vol_20_pctrank6000 | +0.58 | +0.74 | 是 |
| parkinson_vol_60_pctrank6000 | +0.65 | +0.76 | 是 |
| garman_klass_vol_240 | +0.61 | +0.98 | 是 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | +0.76 | +0.84 | 是 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | +0.65 | +0.65 | 否 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | +0.64 | +0.85 | 是 |
| realized_vol_100_pctrank2000+roc_10 | +0.39 | +0.54 | 是 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | +0.78 | +0.78 | 否 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | +0.53 | +0.65 | 是 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | +0.63 | +0.65 | 是 |
| avg_gap_50+roc_10 | +0.50 | +0.56 | 是 |
| stochastic_d_100_pctrank2000+keltner_width_20 | +0.75 | +0.76 | 是 |

**15/17个候选分方向独立优化止损止盈后比共享一套配置更好**。

## 每个候选：多空各自选出的止损止盈配置

| 候选 | 多头止损 | 多头水平 | 多头盈亏比 | 多头Sharpe | 多头笔数 | 空头止损 | 空头水平 | 空头盈亏比 | 空头Sharpe | 空头笔数 |
|---|---|---|---|---|---|---|---|---|---|
| vol_of_vol_100 | pct | 0.003 | 1.5 | +0.034 | 5360 | pct | 0.003 | 1.0 | +0.044 | 2456 |
| adx_50_pctrank24000 | none | nan | nan | +0.036 | 5938 | pct | 0.002 | 2.0 | +0.003 | 4620 |
| bb_width_100 | none | nan | nan | +0.044 | 3827 | atr | 1.0 | 2.0 | +0.032 | 2578 |
| bb_width_300_pctrank6000 | none | nan | nan | +0.062 | 1695 | atr | 1.0 | 2.0 | +0.032 | 1120 |
| parkinson_vol_100_pctrank24000 | none | nan | nan | +0.046 | 3225 | atr | 0.5 | 2.0 | +0.035 | 2391 |
| garman_klass_vol_20_pctrank6000 | pct | 0.005 | 1.0 | +0.028 | 10738 | atr | 1.5 | 2.0 | +0.008 | 7890 |
| parkinson_vol_60_pctrank6000 | none | nan | nan | +0.012 | 5256 | atr | 2.0 | 1.0 | +0.060 | 3842 |
| garman_klass_vol_240 | none | nan | nan | +0.087 | 1530 | atr | 0.5 | 2.0 | +0.026 | 782 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | pct | 0.002 | 2.0 | +0.204 | 138 | pct | 0.005 | 1.0 | +0.169 | 114 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | none | nan | nan | +0.133 | 292 | none | nan | nan | +nan | 5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | none | nan | nan | +0.144 | 431 | atr | 1.0 | 1.0 | +0.014 | 136 |
| realized_vol_100_pctrank2000+roc_10 | pct | 0.01 | 1.0 | +0.112 | 317 | atr | 1.5 | 1.5 | +0.033 | 151 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | pct | 0.005 | 1.0 | +0.227 | 152 | pct | 0.005 | 1.0 | +0.086 | 100 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | atr | 2.0 | 1.0 | +0.092 | 405 | pct | 0.002 | 2.0 | +0.088 | 225 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | pct | 0.003 | 2.0 | +0.105 | 419 | pct | 0.003 | 1.5 | +0.235 | 11 |
| avg_gap_50+roc_10 | pct | 0.003 | 1.0 | +0.077 | 671 | atr | 0.5 | 1.5 | -0.023 | 24 |
| stochastic_d_100_pctrank2000+keltner_width_20 | atr | 2.0 | 1.5 | +0.067 | 186 | atr | 1.5 | 1.0 | +0.299 | 101 |

## 结论与下一步

- 完整数据：reports/02p_diagnostic_shared_config_by_side.csv（诊断）、reports/02p_per_side_search.csv（分方向搜索全量）、reports/02p_final_comparison.csv（对比汇总）。
- 下一步（步骤5）：多空对冲互锁验证——检查同一时刻多头信号和空头信号是否会互相冲突（比如两个组合候选同时给出反向信号），以及是否需要互锁规则。

