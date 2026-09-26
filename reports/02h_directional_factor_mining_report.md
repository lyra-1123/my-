# 方向性因子系统性挖掘报告（信号设计track）

## 方法

对M5上的399个因子变体，各自测试两种方向化模式：**动量模式**(因子处于自身历史80%分位以上时，方向=上一根bar涨跌方向延续)、**反转模式**(因子从自身历史20%分位以下上穿=多头，从80%分位以上下穿=空头，穿越事件而非静态阈值)。

两阶段筛选：阶段1(全样本阈值，粗筛)保留触发次数>=200的组合；阶段2对|Sharpe|最高的25个做5折walk-forward(阈值只在训练折定，冻结后用到测试折)，同时拆分多空分别验证——步骤2发现多空严重不对称，这里把这个检验制度化，不再是筛完就默认多空都行。

持有期占位值: 12根M5(~1小时)，真实值待步骤3确定。

## 阶段2 walk-forward结果（25个候选）

| 因子 | 分类 | 模式 | OOS Sharpe(多空合计) | OOS Sharpe(多) | OOS Sharpe(空) | 多空次数 | 折数为正 |
|---|---|---|---|---|---|---|---|
| garman_klass_vol_20 | volatility | reversion | +0.002 | +0.068 | -0.037 | 767/551 | 4/5 |
| adx_50_pctrank2000 | trend_strength | reversion | +0.002 | +0.080 | -0.057 | 516/390 | 2/5 |
| vol_of_vol_100 | volatility | reversion | +0.001 | +0.001 | +0.043 | 470/207 | 2/5 |
| ma_slope_50_pctrank500 | momentum | reversion | +0.001 | +0.016 | +0.012 | 401/391 | 3/5 |
| williams_r_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011 | -0.000 | 2438/2553 | 3/5 |
| stochastic_k_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011 | -0.000 | 2438/2553 | 3/5 |
| donchian_position_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011 | -0.000 | 2438/2553 | 3/5 |
| bb_width_100 | volatility | reversion | +0.001 | +0.047 | -0.035 | 311/220 | 3/5 |
| parkinson_vol_20 | volatility | reversion | +0.001 | +0.013 | +0.002 | 802/553 | 2/5 |
| parkinson_vol_20_pctrank500 | volatility | reversion | +0.001 | +0.014 | -0.003 | 1041/725 | 2/5 |
| ma_slope_100 | momentum | reversion | -0.001 | -0.223 | +0.043 | 151/165 | 2/5 |
| parkinson_vol_100_pctrank2000 | volatility | reversion | -0.001 | +0.073 | -0.150 | 357/224 | 3/5 |
| donchian_position_100 | mean_reversion | reversion | -0.001 | -0.025 | +0.014 | 2425/2510 | 2/5 |
| williams_r_100 | mean_reversion | reversion | -0.001 | -0.025 | +0.014 | 2425/2510 | 2/5 |
| stochastic_k_100 | mean_reversion | reversion | -0.001 | -0.025 | +0.014 | 2425/2510 | 2/5 |
| garman_klass_vol_20_pctrank500 | volatility | reversion | -0.002 | +0.011 | -0.037 | 984/748 | 1/5 |
| bb_width_100_pctrank500 | volatility | reversion | -0.002 | -0.038 | +0.012 | 436/299 | 2/5 |
| stochastic_d_100_pctrank2000 | mean_reversion | reversion | -0.003 | +0.001 | -0.038 | 1430/1527 | 2/5 |
| stochastic_d_100 | mean_reversion | reversion | -0.003 | -0.020 | -0.019 | 1402/1425 | 0/5 |
| atr_50 | volatility | reversion | -0.004 | -0.059 | -0.041 | 304/328 | 1/5 |
| avg_gap_50 | price_action | reversion | -0.005 | -0.037 | -0.120 | 1019/69 | 2/5 |
| ma_slope_50_pctrank2000 | momentum | reversion | -0.005 | -0.016 | -0.104 | 332/337 | 1/5 |
| ma_slope_100_pctrank500 | momentum | reversion | -0.005 | -0.137 | -0.036 | 196/225 | 1/5 |
| choppiness_index_100_pctrank2000 | trend_strength | reversion | -0.006 | -0.044 | -0.057 | 575/804 | 1/5 |
| hour | other | momentum | -0.008 | +0.015 | -0.054 | 8867/8771 | 0/5 |

## 最终通过筛选的候选（OOS Sharpe>0 且 >=3/5折为正 且 多空触发都>=30次）：6个

| 因子 | 分类 | 模式 | OOS Sharpe | OOS Sharpe(多/空) |
|---|---|---|---|---|
| garman_klass_vol_20 | volatility | reversion | +0.002 | +0.068/-0.037 |
| ma_slope_50_pctrank500 | momentum | reversion | +0.001 | +0.016/+0.012 |
| stochastic_k_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011/-0.000 |
| williams_r_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011/-0.000 |
| donchian_position_100_pctrank2000 | mean_reversion | reversion | +0.001 | +0.011/-0.000 |
| bb_width_100 | volatility | reversion | +0.001 | +0.047/-0.035 |

## 结论与下一步

- 完整阶段1扫描(reports/02h_directional_stage1_scan.csv)和阶段2明细(reports/02h_directional_stage2_walkforward.csv)已保存。
- 这一步只测了单因子，没有测试因子组合（AND）；步骤2的经验是AND会大幅降低触发频率，如果这里找到的最强单因子触发率不够高，可能需要OR逻辑或加权打分而不是继续AND。
- 仍然是占位持有期，步骤3定真实持有期/止损后需要重新验证这里的候选。

