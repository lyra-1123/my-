# 方向性因子系统性挖掘报告（信号设计track）

## 方法

对M5上的399个因子变体，各自测试两种方向化模式：**动量模式**(因子处于自身历史80%分位以上时，方向=上一根bar涨跌方向延续)、**反转模式**(因子从自身历史20%分位以下上穿=多头，从80%分位以上下穿=空头，穿越事件而非静态阈值)。

两阶段筛选：阶段1(全样本阈值，粗筛)保留触发次数>=200的组合；阶段2对|Sharpe|最高的25个做5折walk-forward(阈值只在训练折定，冻结后用到测试折)，同时拆分多空分别验证——步骤2发现多空严重不对称，这里把这个检验制度化，不再是筛完就默认多空都行。

持有期占位值: 12根M5(~1小时)，真实值待步骤3确定。

## 阶段2 walk-forward结果（25个候选）

| 因子 | 分类 | 模式 | OOS Sharpe(未年化) | OOS Sharpe(年化) | OOS Sharpe(多) | OOS Sharpe(空) | 多空次数 | 折数为正 |
|---|---|---|---|---|---|---|---|---|
| vol_of_vol_100 | volatility | reversion | +0.028 | +0.72 | +0.028 | +0.029 | 5360/2457 | 5/5 |
| adx_50_pctrank2000 | trend_strength | reversion | +0.023 | +0.68 | +0.032 | +0.011 | 6110/4745 | 3/5 |
| bb_width_100 | volatility | reversion | +0.023 | +0.52 | +0.034 | +0.009 | 3827/2578 | 3/5 |
| bb_width_100_pctrank500 | volatility | reversion | +0.017 | +0.47 | +0.021 | +0.012 | 5369/3750 | 4/5 |
| parkinson_vol_100_pctrank2000 | volatility | reversion | +0.017 | +0.41 | +0.033 | -0.011 | 4325/3010 | 4/5 |
| garman_klass_vol_20_pctrank500 | volatility | reversion | +0.016 | +0.68 | +0.022 | +0.010 | 12257/8430 | 5/5 |
| parkinson_vol_20_pctrank500 | volatility | reversion | +0.014 | +0.57 | +0.018 | +0.009 | 12616/8636 | 4/5 |
| garman_klass_vol_20 | volatility | reversion | +0.002 | +0.07 | +0.005 | -0.001 | 9399/6281 | 4/5 |
| avg_gap_50 | price_action | reversion | -0.000 | -0.02 | -0.002 | +0.017 | 12473/481 | 2/5 |
| parkinson_vol_20 | volatility | reversion | -0.001 | -0.02 | +0.004 | -0.006 | 9843/6458 | 2/5 |
| atr_50 | volatility | reversion | -0.002 | -0.06 | +0.030 | -0.038 | 3787/3568 | 2/5 |
| ma_slope_50_pctrank500 | momentum | reversion | -0.009 | -0.25 | -0.006 | -0.013 | 4497/4528 | 2/5 |
| ma_slope_50_pctrank2000 | momentum | reversion | -0.012 | -0.31 | -0.003 | -0.022 | 4066/4130 | 0/5 |
| hour | other | momentum | -0.013 | -1.76 | +0.019 | -0.046 | 106222/105248 | 0/5 |
| stochastic_d_100 | mean_reversion | reversion | -0.018 | -0.93 | -0.005 | -0.031 | 16673/16916 | 0/5 |
| ma_slope_100 | momentum | reversion | -0.020 | -0.35 | -0.043 | -0.003 | 1899/1893 | 2/5 |
| choppiness_index_100_pctrank2000 | trend_strength | reversion | -0.021 | -0.78 | -0.008 | -0.031 | 7369/9517 | 1/5 |
| stochastic_d_100_pctrank2000 | mean_reversion | reversion | -0.024 | -1.29 | -0.013 | -0.036 | 17143/17652 | 0/5 |
| stochastic_k_100_pctrank2000 | mean_reversion | reversion | -0.026 | -1.85 | -0.028 | -0.025 | 29878/30867 | 0/5 |
| williams_r_100_pctrank2000 | mean_reversion | reversion | -0.026 | -1.85 | -0.028 | -0.025 | 29878/30867 | 0/5 |
| donchian_position_100_pctrank2000 | mean_reversion | reversion | -0.026 | -1.85 | -0.028 | -0.025 | 29878/30867 | 0/5 |
| donchian_position_100 | mean_reversion | reversion | -0.029 | -1.98 | -0.021 | -0.037 | 29103/29786 | 0/5 |
| stochastic_k_100 | mean_reversion | reversion | -0.029 | -1.98 | -0.021 | -0.037 | 29103/29786 | 0/5 |
| williams_r_100 | mean_reversion | reversion | -0.029 | -1.98 | -0.021 | -0.037 | 29103/29786 | 0/5 |
| ma_slope_100_pctrank500 | momentum | reversion | -0.030 | -0.61 | -0.032 | -0.028 | 2473/2484 | 2/5 |

## 最终通过筛选的候选（OOS Sharpe>0 且 >=3/5折为正 且 多空触发都>=30次）：8个

| 因子 | 分类 | 模式 | OOS Sharpe(未年化) | OOS Sharpe(年化) | OOS Sharpe(多/空) |
|---|---|---|---|---|---|
| vol_of_vol_100 | volatility | reversion | +0.028 | +0.72 | +0.028/+0.029 |
| adx_50_pctrank2000 | trend_strength | reversion | +0.023 | +0.68 | +0.032/+0.011 |
| bb_width_100 | volatility | reversion | +0.023 | +0.52 | +0.034/+0.009 |
| bb_width_100_pctrank500 | volatility | reversion | +0.017 | +0.47 | +0.021/+0.012 |
| parkinson_vol_100_pctrank2000 | volatility | reversion | +0.017 | +0.41 | +0.033/-0.011 |
| garman_klass_vol_20_pctrank500 | volatility | reversion | +0.016 | +0.68 | +0.022/+0.010 |
| parkinson_vol_20_pctrank500 | volatility | reversion | +0.014 | +0.57 | +0.018/+0.009 |
| garman_klass_vol_20 | volatility | reversion | +0.002 | +0.07 | +0.005/-0.001 |

## 结论与下一步

- 完整阶段1扫描(reports/02h_directional_stage1_scan.csv)和阶段2明细(reports/02h_directional_stage2_walkforward.csv)已保存。
- 这一步只测了单因子，没有测试因子组合（AND）；步骤2的经验是AND会大幅降低触发频率，如果这里找到的最强单因子触发率不够高，可能需要OR逻辑或加权打分而不是继续AND。
- 仍然是占位持有期，步骤3定真实持有期/止损后需要重新验证这里的候选。

