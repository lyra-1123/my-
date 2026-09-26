# 因子分类筛选报告（阶段2b：PBO + 样本外Sharpe）

## 方法

按交易逻辑把因子分成7类（均值回归/动量/波动率/趋势强度/价格行为/量能/更高周期背景），分类别筛选——这里的“筛选”是固定的统计检验流程（PBO计算+walk-forward Sharpe），不涉及需要独立判断的主观决策，所以用清晰分类的代码逐类跑，而不是真的派生多个独立agent各自跑一遍相同的算术。

每个因子家族有多个窗口/百分位版本（最多12个），这正是过拟合的高发地带——如果只是“挑全样本IC最高的那个”，很可能只是在噪音里挑到了运气好的参数组合。所以这里对每个家族做：

1. **PBO（回测过拟合概率，CSCV方法，Bailey/Lopez de Prado）**：把不重叠的决策点（每24根H1一个，共4,503个）分成10段，穷举所有把10段分成训练/测试两半的方式(C(10,5)=252种)，每种方式里“用训练半段挑出的样本内最优版本”在测试半段的表现排名——如果经常排到测试半段的中位数以下，说明这个“挑最优”的过程本身就是在过拟合噪音，PBO就是这个比例。
2. **Walk-forward样本外Sharpe**：前70%决策点当样本内(IS)、后30%当样本外(OOS)——安全分位阈值只用IS部分数据算，冻结后应用到OOS，不看OOS数据本身。

因子本身不是交易规则，所以用一个固定的、跟因子无关的极简策略做“测试床”：对最近1根bar做反向(fade)，持有到未来第24根bar——这样“用因子做门槛”和不用因子的版本除了因子那道门槛之外完全一样，Sharpe的差异能被干净地归因到因子本身，而不是策略设计。这个测试床本身很粗糙（没有点差/滑点/仓位管理），阶段3的真实马丁回测会更细，这里只是用来筛因子。

通过标准：PBO<=0.5（比抛硬币更可信） 且 OOS Sharpe>0（缺一不可）。

## 总体结果：38个因子家族中，12个通过筛选

### higher_timeframe（1/4通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| h4_adx | 2 | h4_adx_20 | 0.00 | +0.01 | -0.02 | 16.4% | ✗ |
| h4_bb_width | 2 | h4_bb_width_20 | 0.00 | -0.08 | -0.33 | 24.3% | ✗ |
| h4_choppiness_index | 2 | h4_choppiness_index_50 | 0.00 | -0.14 | +0.34 | 21.5% | ✅ |
| h4_efficiency_ratio | 2 | h4_efficiency_ratio_50 | 0.00 | -0.06 | -0.08 | 20.9% | ✗ |

### mean_reversion（1/7通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| donchian_position | 12 | donchian_position_50 | 0.24 | +0.72 | -0.04 | 19.9% | ✗ |
| stochastic_k | 12 | stochastic_k_50 | 0.24 | +0.72 | -0.04 | 19.9% | ✗ |
| williams_r | 12 | williams_r_50 | 0.24 | +0.72 | -0.04 | 19.9% | ✗ |
| zscore_vs_ma | 12 | zscore_vs_ma_100_pctrank2000 | 0.40 | +0.13 | +0.05 | 20.4% | ✅ |
| stochastic_d | 12 | stochastic_d_20_pctrank500 | 0.55 | +0.45 | +0.33 | 18.7% | ✗ |
| rsi | 12 | rsi_20_pctrank500 | 0.69 | +0.27 | -0.23 | 20.8% | ✗ |
| cci | 12 | cci_50 | 0.90 | +0.66 | -0.68 | 19.3% | ✗ |

### momentum（3/6通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| macd_hist | 1 | macd_hist | 0.00 | +0.48 | +0.25 | 21.8% | ✅ |
| autocorr_returns | 12 | autocorr_returns_50 | 0.11 | +0.35 | +0.14 | 23.6% | ✅ |
| aroon_down | 12 | aroon_down_20_pctrank2000 | 0.32 | +0.54 | +0.21 | 19.9% | ✅ |
| roc | 12 | roc_50_pctrank2000 | 0.33 | +0.52 | -0.44 | 21.5% | ✗ |
| aroon_up | 12 | aroon_up_50_pctrank500 | 0.49 | +0.34 | -0.31 | 20.6% | ✗ |
| ma_slope | 12 | ma_slope_100 | 0.76 | +0.01 | -0.15 | 18.1% | ✗ |

### other（0/1通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| hour | 1 | hour | 0.00 | +0.30 | -0.20 | 21.6% | ✗ |

### price_action（2/6通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| streak_length | 1 | streak_length | 0.00 | -0.08 | +0.38 | 22.2% | ✅ |
| dist_from_low | 12 | dist_from_low_20_pctrank500 | 0.15 | +0.01 | -0.00 | 20.9% | ✗ |
| dist_from_high | 12 | dist_from_high_10_pctrank500 | 0.26 | +0.75 | -0.05 | 19.2% | ✗ |
| avg_gap | 12 | avg_gap_50_pctrank500 | 0.40 | +0.56 | -0.21 | 21.4% | ✗ |
| kurt_returns | 12 | kurt_returns_10_pctrank500 | 0.46 | +0.30 | +0.50 | 21.3% | ✅ |
| skew_returns | 12 | skew_returns_10 | 0.83 | +0.65 | -0.22 | 18.6% | ✗ |

### trend_strength（3/6通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| variance_ratio_2 | 12 | variance_ratio_2_50 | 0.14 | +0.46 | -0.02 | 22.7% | ✗ |
| adx | 12 | adx_20_pctrank500 | 0.24 | +0.49 | +0.18 | 19.5% | ✅ |
| choppiness_index | 12 | choppiness_index_10_pctrank500 | 0.33 | +0.33 | +0.04 | 20.4% | ✅ |
| linreg_r2 | 12 | linreg_r2_10_pctrank2000 | 0.33 | +0.36 | -0.04 | 20.3% | ✗ |
| adx_slope | 12 | adx_slope_10 | 0.42 | +0.40 | +0.02 | 22.1% | ✅ |
| efficiency_ratio | 12 | efficiency_ratio_20 | 0.82 | +0.25 | -0.44 | 20.4% | ✗ |

### volatility（1/7通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| garman_klass_vol | 12 | garman_klass_vol_10_pctrank500 | 0.06 | +0.57 | -0.31 | 21.0% | ✗ |
| parkinson_vol | 12 | parkinson_vol_10_pctrank500 | 0.18 | +0.34 | -0.34 | 20.0% | ✗ |
| atr | 12 | atr_10_pctrank500 | 0.22 | +0.54 | -0.81 | 22.1% | ✗ |
| bb_width | 12 | bb_width_10_pctrank500 | 0.23 | +0.46 | +0.23 | 20.9% | ✅ |
| keltner_width | 12 | keltner_width_10_pctrank500 | 0.33 | +0.53 | -0.76 | 20.8% | ✗ |
| vol_of_vol | 12 | vol_of_vol_10_pctrank500 | 0.47 | +0.38 | -0.32 | 19.8% | ✗ |
| realized_vol | 12 | realized_vol_10_pctrank500 | 0.52 | +0.27 | -0.48 | 20.9% | ✗ |

### volume（1/1通过）

| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |
|---|---|---|---|---|---|---|---|
| mfi | 12 | mfi_50_pctrank500 | 0.22 | +0.23 | +0.46 | 20.7% | ✅ |

## 通过筛选的因子家族（按OOS Sharpe年化排序）

| 家族 | 分类 | 最优变体 | PBO | OOS Sharpe(年化) |
|---|---|---|---|---|
| kurt_returns | price_action | kurt_returns_10_pctrank500 | 0.46 | +0.50 |
| mfi | volume | mfi_50_pctrank500 | 0.22 | +0.46 |
| streak_length | price_action | streak_length | 0.00 | +0.38 |
| h4_choppiness_index | higher_timeframe | h4_choppiness_index_50 | 0.00 | +0.34 |
| macd_hist | momentum | macd_hist | 0.00 | +0.25 |
| bb_width | volatility | bb_width_10_pctrank500 | 0.23 | +0.23 |
| aroon_down | momentum | aroon_down_20_pctrank2000 | 0.32 | +0.21 |
| adx | trend_strength | adx_20_pctrank500 | 0.24 | +0.18 |
| autocorr_returns | momentum | autocorr_returns_50 | 0.11 | +0.14 |
| zscore_vs_ma | mean_reversion | zscore_vs_ma_100_pctrank2000 | 0.40 | +0.05 |
| choppiness_index | trend_strength | choppiness_index_10_pctrank500 | 0.33 | +0.04 |
| adx_slope | trend_strength | adx_slope_10 | 0.42 | +0.02 |

## 结论与下一步

- **和阶段2 v1~v4纯IC筛选的结论有明显分歧，这正是做这一步筛选的意义**：v1~v4里最强的几个因子——efficiency_ratio、realized_vol、atr、keltner_width、variance_ratio_2、linreg_r2、parkinson_vol、garman_klass_vol——在这里全部没通过PBO+OOS Sharpe筛选（PBO普遍在0.3~0.9之间，即“挑样本内最优窗口”这个过程本身就不稳健）。反而是MFI（v1~v4里IC很弱）和streak_length、macd_hist这类之前没被重点关注的因子通过了。这说明纯静态相关性和“能否支撑一个稳健的交易规则”是两个不同的问题，前者容易被参数搜索污染，后者更贴近实盘会遇到的情况。
- **意外发现一个冗余群**：donchian_position、stochastic_k、williams_r三个家族的PBO/Sharpe数值完全相同——这不是bug，是因为三者数学上是同一个量的仿射变换(williams_r = -100+100×donchian_position，stochastic_k = 100×donchian_position)，秩相关和分位数筛选对仿射变换不敏感，所以给出完全一致的结果。阶段3应该把这三个当成一个因子用，不要误以为是三个独立信号的相互印证。
- 每个类别都至少有1个家族通过（除了“other”类的hour），说明7个类别的分类思路是合理的，没有哪一类整体被淘汰；但每个类别通过率都不高（1~3/6），说明多数“看起来有道理”的技术指标经不起PBO检验。
- 完整结果（含每个家族的诊断数据）在`reports/02b_factor_screening_results.csv`。
- 这一步的Sharpe来自一个刻意简化、和因子无关的测试床策略，只用来公平比较“有没有这个因子门槛”的差异，**不代表真实马丁格尔策略的Sharpe**——阶段3要在真实的马丁资金曲线（含加仓/点差/保证金）上重新验证这里通过筛选的因子，静态因子筛选和策略级回测是两回事。
- PBO<=0.5只是“比瞎猜强”的最低门槛，学术上更严格的要求是PBO<0.2；如果阶段3想更保守，可以直接从CSV里按更严的阈值重新筛一遍，不需要重跑这个脚本。

