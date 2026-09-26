# 方向性因子两两组合搜索报告

## 方法

从02h的399个因子里，每个家族挑|Sharpe|最强的1个代表变体（共36个），穷举两两配对（共630对），要求两个因子的方向AND一致（同时看多才算多头信号，同时看空才算空头信号，不是简单都非零）。

**筛选标准明确包含“能不能用”，不只是Sharpe**：触发次数>=200且年均触发>=20次才进入候选池——一个Sharpe很高但18年只响几次的组合不是能交易的信号，不纳入排名。

阶段2对Top30做5折walk-forward（两个因子的阈值都只在训练折定、冻结后用到测试折）。

## 阶段1候选池：482对满足触发频率要求

| 因子A | 因子B | 年均触发 | Sharpe(合计/多/空) |
|---|---|---|---|
| vol_of_vol_100 | autocorr_returns_50_pctrank2000 | 22 | +0.150/+0.160/+0.130 |
| stochastic_d_100_pctrank2000 | keltner_width_20 | 26 | +0.124/+0.038/+0.224 |
| avg_gap_50 | garman_klass_vol_20_pctrank500 | 22 | +0.119/+0.102/+0.206 |
| williams_r_100 | parkinson_vol_100_pctrank2000 | 31 | -0.116/-0.112/-0.140 |
| donchian_position_100 | parkinson_vol_100_pctrank2000 | 31 | -0.116/-0.112/-0.140 |
| stochastic_k_100 | parkinson_vol_100_pctrank2000 | 31 | -0.116/-0.112/-0.140 |
| avg_gap_50 | zscore_vs_ma_100_pctrank500 | 41 | +0.108/+0.133/+0.019 |
| avg_gap_50 | williams_r_100 | 42 | +0.107/+0.137/+0.018 |
| avg_gap_50 | stochastic_k_100 | 42 | +0.107/+0.137/+0.018 |
| avg_gap_50 | donchian_position_100 | 42 | +0.107/+0.137/+0.018 |
| adx_50_pctrank2000 | autocorr_returns_50_pctrank2000 | 24 | +0.106/+0.072/+0.144 |
| realized_vol_100_pctrank2000 | variance_ratio_2_50_pctrank2000 | 25 | +0.105/+0.158/+0.011 |
| dist_from_low_100_pctrank2000 | mfi_10_pctrank2000 | 41 | -0.104/-0.089/-0.169 |
| choppiness_index_100_pctrank2000 | cci_100_pctrank500 | 22 | -0.104/-0.077/-0.178 |
| choppiness_index_100_pctrank2000 | dist_from_high_100 | 54 | -0.102/-0.091/-0.116 |

## 阶段2 walk-forward结果（Top30）

| 因子A | 因子B | 年均触发 | OOS Sharpe(合计,未年化) | OOS Sharpe(年化) | OOS Sharpe(多/空,未年化) | 折数为正 |
|---|---|---|---|---|---|---|
| adx_50_pctrank2000 | autocorr_returns_50_pctrank2000 | 21 | +0.108 | +0.49 | +0.144/+0.079 | 4/5 |
| avg_gap_50 | garman_klass_vol_20_pctrank500 | 24 | +0.106 | +0.52 | +0.089/+nan | 4/5 |
| kurt_returns_100 | mean_reversion_speed_50_pctrank500 | 46 | +0.078 | +0.53 | +0.093/+0.016 | 5/5 |
| realized_vol_100_pctrank2000 | roc_10 | 38 | +0.063 | +0.39 | +0.105/+0.002 | 4/5 |
| realized_vol_100_pctrank2000 | variance_ratio_2_50_pctrank2000 | 21 | +0.055 | +0.25 | +0.139/-0.107 | 3/5 |
| bb_width_100 | variance_ratio_2_50_pctrank2000 | 24 | +0.054 | +0.26 | +0.029/+0.088 | 2/5 |
| parkinson_vol_100_pctrank2000 | aroon_up_10_pctrank500 | 52 | +0.047 | +0.34 | +0.075/-0.004 | 4/5 |
| avg_gap_50 | zscore_vs_ma_100_pctrank500 | 35 | +0.044 | +0.26 | +0.041/+0.132 | 3/5 |
| avg_gap_50 | roc_10 | 57 | +0.030 | +0.22 | +0.026/+0.101 | 3/5 |
| stochastic_d_100_pctrank2000 | keltner_width_20 | 24 | +0.024 | +0.12 | -0.050/+0.159 | 3/5 |
| vol_of_vol_100 | autocorr_returns_50_pctrank2000 | 14 | +0.019 | +0.07 | +0.001/+0.079 | 2/5 |
| parkinson_vol_100_pctrank2000 | aroon_down_10 | 23 | +0.013 | +0.06 | -0.018/+0.034 | 2/5 |
| ma_slope_100 | aroon_up_10_pctrank500 | 21 | +0.012 | +0.06 | -0.007/+0.039 | 2/5 |
| keltner_width_20 | rsi_100_pctrank500 | 28 | -0.010 | -0.05 | -0.012/-0.006 | 2/5 |
| avg_gap_50 | stochastic_k_100 | 34 | -0.012 | -0.07 | +0.002/-0.447 | 3/5 |
| avg_gap_50 | williams_r_100 | 34 | -0.012 | -0.07 | +0.002/-0.447 | 3/5 |
| avg_gap_50 | donchian_position_100 | 34 | -0.012 | -0.07 | +0.002/-0.447 | 3/5 |
| avg_gap_50 | stochastic_d_100_pctrank2000 | 19 | -0.013 | -0.05 | -0.018/+nan | 3/5 |
| ma_slope_100 | linreg_r2_20_pctrank500 | 34 | -0.045 | -0.26 | -0.153/+0.032 | 2/5 |
| mfi_10_pctrank2000 | rsi_100_pctrank500 | 23 | -0.051 | -0.24 | -0.084/+0.009 | 1/5 |
| hour | aroon_down_10 | 491 | -0.069 | -1.52 | -0.025/-0.084 | 0/5 |
| stochastic_d_100_pctrank2000 | efficiency_ratio_20 | 215 | -0.079 | -1.16 | -0.072/-0.086 | 0/5 |
| williams_r_100 | parkinson_vol_100_pctrank2000 | 27 | -0.081 | -0.42 | -0.082/-0.077 | 0/5 |
| stochastic_k_100 | parkinson_vol_100_pctrank2000 | 27 | -0.081 | -0.42 | -0.082/-0.077 | 0/5 |
| donchian_position_100 | parkinson_vol_100_pctrank2000 | 27 | -0.081 | -0.42 | -0.082/-0.077 | 0/5 |
| choppiness_index_100_pctrank2000 | dist_from_high_100 | 45 | -0.090 | -0.61 | -0.054/-0.123 | 1/5 |
| dist_from_low_100_pctrank2000 | ma_cross_count_50 | 38 | -0.111 | -0.68 | -0.105/-0.157 | 0/5 |
| hour | realized_vol_100_pctrank2000 | 70 | -0.120 | -1.00 | +0.139/-0.141 | 1/5 |
| dist_from_low_100_pctrank2000 | mfi_10_pctrank2000 | 38 | -0.125 | -0.77 | -0.123/-0.143 | 0/5 |
| choppiness_index_100_pctrank2000 | cci_100_pctrank500 | 17 | -0.134 | -0.55 | -0.105/-0.199 | 1/5 |

## 最终通过的候选（OOS Sharpe>0 且 >=3/5折为正 且 年均触发>=20次）：9个

| 因子A | 因子B | 年均触发 | OOS Sharpe(未年化) | OOS Sharpe(年化) |
|---|---|---|---|---|
| adx_50_pctrank2000 | autocorr_returns_50_pctrank2000 | 21 | +0.108 | +0.49 |
| avg_gap_50 | garman_klass_vol_20_pctrank500 | 24 | +0.106 | +0.52 |
| kurt_returns_100 | mean_reversion_speed_50_pctrank500 | 46 | +0.078 | +0.53 |
| realized_vol_100_pctrank2000 | roc_10 | 38 | +0.063 | +0.39 |
| realized_vol_100_pctrank2000 | variance_ratio_2_50_pctrank2000 | 21 | +0.055 | +0.25 |
| parkinson_vol_100_pctrank2000 | aroon_up_10_pctrank500 | 52 | +0.047 | +0.34 |
| avg_gap_50 | zscore_vs_ma_100_pctrank500 | 35 | +0.044 | +0.26 |
| avg_gap_50 | roc_10 | 57 | +0.030 | +0.22 |
| stochastic_d_100_pctrank2000 | keltner_width_20 | 24 | +0.024 | +0.12 |

## 结论与下一步

- 完整数据：reports/02i_pair_stage1_scan.csv（阶段1全量）、reports/02i_pair_stage2_walkforward.csv（阶段2明细）。
- 如果这一步仍然找不到可用信号，说明两两AND也不够，可能需要三因子组合、OR逻辑、或改变评分方式（比如不追求胜率/Sharpe，而是先看方向准确率）。
- 仍是占位持有期，步骤3定真实参数后需要重新验证。

