# 因子挖掘报告（阶段2，v4：新增因子类型 + 全面百分位化 + 系统性组合搜索）

## 方法

马丁格尔策略的核心需求不是预测涨跌方向，而是区分“震荡/均值回归”（马丁网格能存活的regime）和“单边趋势持续”（会把网格打爆的regime）。所以这里不做常规的方向性IC分析，而是用未来Kaufman效率系数（Efficiency Ratio, ER）作为“危险程度”标签：ER→1代表未来价格路径高效率地朝一个方向走（趋势持续，危险），ER→0代表未来路径来回震荡但净位移很小（安全）。所有候选因子只使用截至当前bar的历史数据计算（无未来函数，H4衍生的因子额外做了“该H4 bar实际收盘时刻之后才可见”的对齐处理），标签则专门使用未来窗口数据，仅用于评估、不作为任何模型输入。

评估基于H1（108,083根），标签窗口：1天(24根H1)、3天(72根H1)
；候选因子从v1的25个扩大到本轮的383个：v2新增波动率类(keltner_width/vol_of_vol)、趋势类(adx_slope/variance_ratio_2/autocorr_returns/streak_length/macd_hist)、超买超卖类(stochastic_k/d、williams_r、cci、roc、dist_from_high/low、donchian_position)、分布形态类(skew/kurt_returns)，以及2个H4更高周期的regime背景因子(h4_adx、h4_bb_width、h4_efficiency_ratio)，且每类基本指标从3个回看窗口(14/24/48)扩到4个(10/20/50/100)；v3新增MFI(资金流量指标，用tick成交量代理，非真实成交量，解读需谨慎)，并对波动率/趋势强度类因子(atr、realized_vol、bb_width、keltner_width、vol_of_vol、adx、efficiency_ratio、mfi)额外算了相对其自身滚动500根/2000根历史的百分位排名——原始因子是黄金美元报价的绝对水平，18年里金价从~860涨到~5300，同样的ATR数值在2009年和2026年代表的“波动程度”完全不是一回事，百分位排名把它转成“相对当前regime”的量纲，跨样本可比。
v4新增专门为“震荡vs趋势”设计的Choppiness Index、Aroon up/down、更高效的OHLC波动率估计量(Parkinson、Garman-Klass，用到整根bar的高低点/开收盘信息而不只是收盘价)、线性回归拟合优度R²(方向无关的“趋势有多干净”)、平均开盘跳空幅度；并把百分位排名从v3的8个手选家族扩展到全部因子家族（不再预判哪些需要、哪些不需要）；同时把v1~v3手选一对做组合验证，换成了系统性搜索：先从每个指标家族挑IC最强的代表因子（15个），再穷举两两组合，也测试了多因子平均合成打分。

## 候选池：|IC|>=0.01的因子（286/383个，按max|IC|排序，完整CSV见`reports/02_factor_candidate_pool.csv`）

| 因子 | 1天(24根H1) | 3天(72根H1) | max\|IC\| |
|---|---|---|---|
| bb_width_50 | -0.087 | -0.072 | 0.087 |
| bb_width_50_pctrank500 | -0.086 | -0.047 | 0.086 |
| bb_width_50_pctrank2000 | -0.085 | -0.072 | 0.085 |
| bb_width_20 | -0.084 | -0.059 | 0.084 |
| bb_width_20_pctrank2000 | -0.082 | -0.054 | 0.082 |
| adx_20 | -0.081 | -0.035 | 0.081 |
| choppiness_index_50 | +0.081 | +0.043 | 0.081 |
| bb_width_20_pctrank500 | -0.081 | -0.035 | 0.081 |
| choppiness_index_50_pctrank2000 | +0.079 | +0.038 | 0.079 |
| adx_20_pctrank2000 | -0.079 | -0.029 | 0.079 |
| adx_20_pctrank500 | -0.076 | -0.023 | 0.076 |
| choppiness_index_20 | +0.076 | +0.028 | 0.076 |
| keltner_width_100_pctrank2000 | -0.037 | -0.076 | 0.076 |
| choppiness_index_20_pctrank2000 | +0.076 | +0.026 | 0.076 |
| atr_100_pctrank2000 | -0.037 | -0.076 | 0.076 |
| bb_width_100 | -0.061 | -0.075 | 0.075 |
| choppiness_index_50_pctrank500 | +0.074 | +0.028 | 0.074 |
| bb_width_100_pctrank2000 | -0.063 | -0.071 | 0.071 |
| h4_bb_width_50 | -0.049 | -0.071 | 0.071 |
| choppiness_index_20_pctrank500 | +0.071 | +0.023 | 0.071 |
| choppiness_index_100 | +0.067 | +0.049 | 0.067 |
| h4_bb_width_20 | -0.063 | -0.067 | 0.067 |
| parkinson_vol_100_pctrank2000 | -0.038 | -0.066 | 0.066 |
| h4_choppiness_index_20 | +0.066 | +0.043 | 0.066 |
| garman_klass_vol_100_pctrank2000 | -0.032 | -0.065 | 0.065 |
| choppiness_index_100_pctrank2000 | +0.065 | +0.041 | 0.065 |
| keltner_width_100 | -0.026 | -0.061 | 0.061 |
| efficiency_ratio_20 | -0.061 | -0.038 | 0.061 |
| realized_vol_100_pctrank2000 | -0.046 | -0.061 | 0.061 |
| efficiency_ratio_20_pctrank2000 | -0.060 | -0.035 | 0.060 |
| keltner_width_50_pctrank2000 | -0.038 | -0.060 | 0.060 |
| bb_width_100_pctrank500 | -0.060 | -0.039 | 0.060 |
| adx_10 | -0.059 | -0.020 | 0.059 |
| bb_width_10 | -0.058 | -0.044 | 0.058 |
| efficiency_ratio_20_pctrank500 | -0.058 | -0.029 | 0.058 |
| atr_50_pctrank2000 | -0.038 | -0.058 | 0.058 |
| bb_width_10_pctrank2000 | -0.058 | -0.038 | 0.058 |
| adx_10_pctrank2000 | -0.057 | -0.015 | 0.057 |
| realized_vol_20_pctrank500 | -0.057 | -0.021 | 0.057 |
| bb_width_10_pctrank500 | -0.056 | -0.020 | 0.056 |
| choppiness_index_100_pctrank500 | +0.055 | +0.020 | 0.055 |
| keltner_width_50 | -0.032 | -0.055 | 0.055 |
| atr_100 | -0.020 | -0.055 | 0.055 |
| adx_10_pctrank500 | -0.055 | -0.012 | 0.055 |
| efficiency_ratio_50 | -0.054 | -0.040 | 0.054 |
| realized_vol_20_pctrank2000 | -0.054 | -0.042 | 0.054 |
| realized_vol_20 | -0.053 | -0.042 | 0.053 |
| efficiency_ratio_50_pctrank2000 | -0.053 | -0.036 | 0.053 |
| atr_50 | -0.025 | -0.053 | 0.053 |
| realized_vol_50_pctrank500 | -0.053 | -0.024 | 0.053 |
| realized_vol_100 | -0.037 | -0.052 | 0.052 |
| keltner_width_100_pctrank500 | -0.047 | -0.052 | 0.052 |
| parkinson_vol_100 | -0.027 | -0.052 | 0.052 |
| choppiness_index_10_pctrank2000 | +0.052 | +0.016 | 0.052 |
| garman_klass_vol_100 | -0.023 | -0.052 | 0.052 |
| realized_vol_50_pctrank2000 | -0.046 | -0.051 | 0.051 |
| autocorr_returns_100 | -0.036 | -0.050 | 0.050 |
| atr_100_pctrank500 | -0.046 | -0.050 | 0.050 |
| choppiness_index_10 | +0.050 | +0.019 | 0.050 |
| parkinson_vol_20_pctrank500 | -0.050 | -0.016 | 0.050 |
| realized_vol_100_pctrank500 | -0.050 | -0.030 | 0.050 |
| variance_ratio_2_100 | -0.034 | -0.050 | 0.050 |
| keltner_width_20_pctrank2000 | -0.042 | -0.050 | 0.050 |
| realized_vol_50 | -0.040 | -0.049 | 0.049 |
| keltner_width_20_pctrank500 | -0.049 | -0.021 | 0.049 |
| h4_choppiness_index_50 | +0.049 | +0.046 | 0.049 |
| choppiness_index_10_pctrank500 | +0.048 | +0.013 | 0.048 |
| atr_20_pctrank500 | -0.048 | -0.021 | 0.048 |
| parkinson_vol_50_pctrank2000 | -0.035 | -0.048 | 0.048 |
| keltner_width_20 | -0.044 | -0.048 | 0.048 |
| atr_20 | -0.033 | -0.047 | 0.047 |
| realized_vol_10_pctrank2000 | -0.047 | -0.034 | 0.047 |
| realized_vol_10 | -0.047 | -0.035 | 0.047 |
| atr_20_pctrank2000 | -0.042 | -0.047 | 0.047 |
| realized_vol_10_pctrank500 | -0.047 | -0.014 | 0.047 |
| linreg_r2_20 | -0.046 | -0.027 | 0.046 |
| efficiency_ratio_50_pctrank500 | -0.046 | -0.024 | 0.046 |
| parkinson_vol_20 | -0.046 | -0.039 | 0.046 |
| linreg_r2_20_pctrank2000 | -0.046 | -0.024 | 0.046 |
| atr_50_pctrank500 | -0.045 | -0.031 | 0.045 |
| keltner_width_50_pctrank500 | -0.045 | -0.031 | 0.045 |
| garman_klass_vol_50_pctrank2000 | -0.030 | -0.045 | 0.045 |
| parkinson_vol_20_pctrank2000 | -0.045 | -0.040 | 0.045 |
| parkinson_vol_50 | -0.031 | -0.044 | 0.044 |
| linreg_r2_50 | -0.044 | -0.041 | 0.044 |
| garman_klass_vol_20_pctrank500 | -0.044 | -0.013 | 0.044 |
| linreg_r2_20_pctrank500 | -0.043 | -0.018 | 0.043 |
| garman_klass_vol_50 | -0.026 | -0.043 | 0.043 |
| parkinson_vol_100_pctrank500 | -0.043 | -0.038 | 0.043 |
| keltner_width_10_pctrank500 | -0.042 | -0.014 | 0.042 |
| parkinson_vol_10 | -0.042 | -0.036 | 0.042 |
| parkinson_vol_10_pctrank2000 | -0.042 | -0.035 | 0.042 |
| keltner_width_10 | -0.042 | -0.039 | 0.042 |
| dist_from_low_50_pctrank2000 | -0.029 | -0.042 | 0.042 |
| parkinson_vol_10_pctrank500 | -0.042 | -0.013 | 0.042 |
| h4_adx_50 | -0.012 | -0.042 | 0.042 |
| atr_10_pctrank500 | -0.042 | -0.014 | 0.042 |
| keltner_width_10_pctrank2000 | -0.041 | -0.038 | 0.041 |
| autocorr_returns_100_pctrank2000 | -0.033 | -0.041 | 0.041 |
| linreg_r2_50_pctrank2000 | -0.041 | -0.038 | 0.041 |
| variance_ratio_2_100_pctrank2000 | -0.031 | -0.041 | 0.041 |
| atr_10_pctrank2000 | -0.041 | -0.036 | 0.041 |
| avg_gap_100_pctrank2000 | -0.024 | -0.041 | 0.041 |
| atr_10 | -0.035 | -0.041 | 0.041 |
| dist_from_low_50 | -0.030 | -0.041 | 0.041 |
| garman_klass_vol_100_pctrank500 | -0.037 | -0.040 | 0.040 |
| adx_50 | -0.040 | -0.037 | 0.040 |
| garman_klass_vol_20 | -0.040 | -0.037 | 0.040 |
| parkinson_vol_50_pctrank500 | -0.039 | -0.017 | 0.039 |
| garman_klass_vol_20_pctrank2000 | -0.039 | -0.038 | 0.039 |
| linreg_r2_50_pctrank500 | -0.038 | -0.030 | 0.038 |
| h4_efficiency_ratio_20 | -0.038 | -0.034 | 0.038 |
| garman_klass_vol_10 | -0.038 | -0.035 | 0.038 |
| garman_klass_vol_10_pctrank500 | -0.038 | -0.012 | 0.038 |
| garman_klass_vol_10_pctrank2000 | -0.038 | -0.033 | 0.038 |
| dist_from_low_100_pctrank2000 | -0.023 | -0.036 | 0.036 |
| adx_50_pctrank2000 | -0.036 | -0.030 | 0.036 |
| dist_from_low_100 | -0.022 | -0.035 | 0.035 |
| vol_of_vol_50_pctrank2000 | -0.035 | -0.017 | 0.035 |
| dist_from_low_20_pctrank500 | -0.034 | -0.015 | 0.034 |
| vol_of_vol_20_pctrank2000 | -0.034 | -0.018 | 0.034 |
| dist_from_low_50_pctrank500 | -0.034 | -0.029 | 0.034 |
| efficiency_ratio_100 | -0.033 | -0.021 | 0.033 |
| adx_100 | -0.033 | -0.029 | 0.033 |
| efficiency_ratio_100_pctrank2000 | -0.033 | -0.017 | 0.033 |
| efficiency_ratio_10_pctrank2000 | -0.033 | -0.017 | 0.033 |
| adx_100_pctrank2000 | -0.033 | -0.019 | 0.033 |
| dist_from_high_20_pctrank2000 | +0.033 | +0.016 | 0.033 |
| vol_of_vol_20 | -0.033 | -0.020 | 0.033 |
| variance_ratio_2_100_pctrank500 | -0.027 | -0.033 | 0.033 |
| efficiency_ratio_10 | -0.033 | -0.020 | 0.033 |
| vol_of_vol_10_pctrank2000 | -0.033 | -0.019 | 0.033 |
| dist_from_high_20 | +0.032 | +0.019 | 0.032 |
| vol_of_vol_20_pctrank500 | -0.032 | -0.009 | 0.032 |
| dist_from_low_20 | -0.032 | -0.029 | 0.032 |
| autocorr_returns_100_pctrank500 | -0.029 | -0.032 | 0.032 |
| garman_klass_vol_50_pctrank500 | -0.032 | -0.014 | 0.032 |
| adx_slope_20 | -0.032 | -0.012 | 0.032 |
| vol_of_vol_10 | -0.032 | -0.020 | 0.032 |
| dist_from_low_20_pctrank2000 | -0.031 | -0.026 | 0.031 |
| dist_from_high_20_pctrank500 | +0.031 | +0.007 | 0.031 |
| dist_from_low_10_pctrank500 | -0.031 | -0.012 | 0.031 |
| mfi_10_pctrank2000 | -0.022 | -0.031 | 0.031 |
| efficiency_ratio_10_pctrank500 | -0.030 | -0.015 | 0.030 |
| vol_of_vol_10_pctrank500 | -0.030 | -0.012 | 0.030 |
| dist_from_low_10 | -0.030 | -0.026 | 0.030 |
| vol_of_vol_50 | -0.030 | -0.018 | 0.030 |
| adx_slope_20_pctrank2000 | -0.030 | -0.010 | 0.030 |
| avg_gap_50_pctrank2000 | -0.010 | -0.030 | 0.030 |
| adx_50_pctrank500 | -0.030 | -0.019 | 0.030 |
| dist_from_low_10_pctrank2000 | -0.029 | -0.023 | 0.029 |
| adx_slope_20_pctrank500 | -0.029 | -0.010 | 0.029 |
| kurt_returns_100_pctrank2000 | -0.028 | +0.010 | 0.028 |
| autocorr_returns_20_pctrank2000 | -0.028 | -0.014 | 0.028 |
| autocorr_returns_20_pctrank500 | -0.027 | -0.009 | 0.027 |
| dist_from_high_50 | +0.027 | +0.007 | 0.027 |
| dist_from_low_100_pctrank500 | -0.027 | -0.022 | 0.027 |
| dist_from_high_50_pctrank2000 | +0.027 | +0.004 | 0.027 |
| autocorr_returns_20 | -0.027 | -0.017 | 0.027 |
| linreg_r2_10 | -0.027 | -0.019 | 0.027 |
| vol_of_vol_50_pctrank500 | -0.027 | -0.000 | 0.027 |
| adx_100_pctrank500 | -0.027 | -0.002 | 0.027 |
| kurt_returns_50_pctrank2000 | -0.026 | -0.004 | 0.026 |
| linreg_r2_10_pctrank2000 | -0.026 | -0.014 | 0.026 |
| kurt_returns_50 | -0.026 | -0.001 | 0.026 |
| roc_50_pctrank500 | -0.008 | -0.026 | 0.026 |
| efficiency_ratio_100_pctrank500 | -0.026 | -0.004 | 0.026 |
| rsi_50_pctrank500 | -0.009 | -0.026 | 0.026 |
| aroon_up_50_pctrank500 | -0.025 | -0.026 | 0.026 |
| rsi_50_pctrank2000 | -0.001 | -0.026 | 0.026 |
| cci_100_pctrank500 | -0.011 | -0.025 | 0.025 |
| mfi_50_pctrank500 | -0.013 | -0.025 | 0.025 |
| skew_returns_100 | +0.004 | +0.025 | 0.025 |
| aroon_up_50_pctrank2000 | -0.020 | -0.025 | 0.025 |
| autocorr_returns_50 | -0.025 | -0.015 | 0.025 |
| roc_50_pctrank2000 | +0.001 | -0.025 | 0.025 |
| linreg_r2_10_pctrank500 | -0.024 | -0.012 | 0.024 |
| variance_ratio_2_20_pctrank2000 | -0.024 | -0.014 | 0.024 |
| autocorr_returns_50_pctrank2000 | -0.024 | -0.013 | 0.024 |
| aroon_down_50_pctrank2000 | -0.024 | +0.013 | 0.024 |
| variance_ratio_2_20_pctrank500 | -0.024 | -0.008 | 0.024 |
| variance_ratio_2_20 | -0.024 | -0.016 | 0.024 |
| dist_from_high_10 | +0.024 | +0.018 | 0.024 |
| aroon_down_50 | -0.024 | +0.011 | 0.024 |
| dist_from_high_10_pctrank2000 | +0.023 | +0.015 | 0.023 |
| kurt_returns_50_pctrank500 | -0.023 | -0.013 | 0.023 |
| aroon_up_50 | -0.021 | -0.023 | 0.023 |
| kurt_returns_100 | -0.023 | +0.017 | 0.023 |
| aroon_up_100 | -0.023 | +0.001 | 0.023 |
| dist_from_high_50_pctrank500 | +0.022 | -0.004 | 0.022 |
| dist_from_high_100 | +0.019 | +0.022 | 0.022 |
| zscore_vs_ma_100_pctrank500 | -0.009 | -0.022 | 0.022 |
| ma_slope_20_pctrank2000 | -0.004 | -0.022 | 0.022 |
| aroon_up_100_pctrank500 | -0.022 | -0.002 | 0.022 |
| avg_gap_100_pctrank500 | -0.022 | -0.020 | 0.022 |
| zscore_vs_ma_50_pctrank500 | -0.006 | -0.022 | 0.022 |
| variance_ratio_2_50 | -0.022 | -0.016 | 0.022 |
| ma_slope_20 | -0.004 | -0.022 | 0.022 |
| cci_50_pctrank500 | -0.003 | -0.022 | 0.022 |
| dist_from_high_10_pctrank500 | +0.022 | +0.004 | 0.022 |
| stochastic_d_50_pctrank2000 | +0.001 | -0.022 | 0.022 |
| variance_ratio_2_50_pctrank2000 | -0.022 | -0.013 | 0.022 |
| zscore_vs_ma_50_pctrank2000 | -0.000 | -0.022 | 0.022 |
| aroon_up_100_pctrank2000 | -0.021 | -0.000 | 0.021 |
| h4_adx_20 | -0.021 | -0.021 | 0.021 |
| rsi_50 | -0.002 | -0.021 | 0.021 |
| autocorr_returns_10_pctrank2000 | -0.021 | -0.012 | 0.021 |
| ma_slope_20_pctrank500 | -0.008 | -0.021 | 0.021 |
| williams_r_50_pctrank2000 | +0.001 | -0.021 | 0.021 |
| stochastic_k_50_pctrank2000 | +0.001 | -0.021 | 0.021 |
| donchian_position_50_pctrank2000 | +0.001 | -0.021 | 0.021 |
| skew_returns_100_pctrank2000 | +0.005 | +0.020 | 0.020 |
| autocorr_returns_50_pctrank500 | -0.020 | -0.004 | 0.020 |
| autocorr_returns_10_pctrank500 | -0.020 | -0.009 | 0.020 |
| stochastic_d_50_pctrank500 | -0.006 | -0.020 | 0.020 |
| roc_50 | -0.001 | -0.020 | 0.020 |
| cci_50_pctrank2000 | +0.002 | -0.020 | 0.020 |
| linreg_r2_100 | -0.018 | -0.020 | 0.020 |
| mfi_50_pctrank2000 | -0.009 | -0.020 | 0.020 |
| skew_returns_100_pctrank500 | +0.005 | +0.020 | 0.020 |
| williams_r_50_pctrank500 | -0.005 | -0.020 | 0.020 |
| donchian_position_50_pctrank500 | -0.005 | -0.020 | 0.020 |
| stochastic_k_50_pctrank500 | -0.005 | -0.020 | 0.020 |
| mfi_50 | -0.009 | -0.019 | 0.019 |
| adx_slope_50 | -0.019 | +0.001 | 0.019 |
| autocorr_returns_10 | -0.019 | -0.014 | 0.019 |
| zscore_vs_ma_50 | -0.000 | -0.019 | 0.019 |
| kurt_returns_100_pctrank500 | -0.019 | +0.001 | 0.019 |
| kurt_returns_20_pctrank2000 | -0.019 | -0.007 | 0.019 |
| mfi_100_pctrank500 | -0.019 | -0.019 | 0.019 |
| adx_slope_50_pctrank2000 | -0.019 | +0.005 | 0.019 |
| variance_ratio_2_10_pctrank2000 | -0.019 | -0.013 | 0.019 |
| linreg_r2_100_pctrank2000 | -0.017 | -0.019 | 0.019 |
| variance_ratio_2_10_pctrank500 | -0.018 | -0.009 | 0.018 |
| variance_ratio_2_50_pctrank500 | -0.018 | -0.005 | 0.018 |
| cci_100_pctrank2000 | -0.002 | -0.018 | 0.018 |
| stochastic_d_50 | +0.001 | -0.018 | 0.018 |
| cci_50 | +0.002 | -0.018 | 0.018 |
| zscore_vs_ma_100_pctrank2000 | -0.002 | -0.017 | 0.017 |
| variance_ratio_2_10 | -0.017 | -0.014 | 0.017 |
| williams_r_50 | +0.001 | -0.017 | 0.017 |
| stochastic_k_50 | +0.001 | -0.017 | 0.017 |
| donchian_position_50 | +0.001 | -0.017 | 0.017 |
| kurt_returns_20 | -0.017 | -0.003 | 0.017 |
| h4_efficiency_ratio_50 | -0.009 | -0.017 | 0.017 |
| aroon_down_50_pctrank500 | -0.017 | +0.013 | 0.017 |
| dist_from_high_100_pctrank2000 | +0.017 | +0.015 | 0.017 |
| kurt_returns_20_pctrank500 | -0.015 | -0.015 | 0.015 |
| stochastic_d_100_pctrank500 | -0.008 | -0.015 | 0.015 |
| linreg_r2_100_pctrank500 | -0.014 | -0.012 | 0.014 |
| avg_gap_20_pctrank2000 | -0.009 | -0.014 | 0.014 |
| stochastic_k_100_pctrank500 | -0.007 | -0.014 | 0.014 |
| donchian_position_100_pctrank500 | -0.007 | -0.014 | 0.014 |
| williams_r_100_pctrank500 | -0.007 | -0.014 | 0.014 |
| zscore_vs_ma_100 | -0.002 | -0.014 | 0.014 |
| vol_of_vol_100 | -0.007 | -0.014 | 0.014 |
| adx_slope_100 | -0.007 | +0.014 | 0.014 |
| mfi_20_pctrank500 | -0.013 | -0.004 | 0.013 |
| adx_slope_50_pctrank500 | -0.013 | +0.011 | 0.013 |
| cci_100 | -0.002 | -0.013 | 0.013 |
| vol_of_vol_100_pctrank500 | -0.008 | +0.013 | 0.013 |
| vol_of_vol_100_pctrank2000 | -0.013 | -0.013 | 0.013 |
| avg_gap_50_pctrank500 | -0.007 | -0.013 | 0.013 |
| rsi_10_pctrank2000 | -0.003 | -0.013 | 0.013 |
| mfi_100_pctrank2000 | -0.013 | -0.003 | 0.013 |
| ma_slope_100_pctrank2000 | +0.012 | -0.002 | 0.012 |
| roc_100_pctrank500 | -0.012 | +0.000 | 0.012 |
| adx_slope_100_pctrank2000 | -0.006 | +0.012 | 0.012 |
| avg_gap_100 | -0.012 | -0.009 | 0.012 |
| aroon_down_100 | -0.012 | -0.010 | 0.012 |
| rsi_100_pctrank500 | -0.012 | -0.005 | 0.012 |
| ma_slope_50 | -0.012 | +0.001 | 0.012 |
| avg_gap_10_pctrank2000 | -0.005 | -0.011 | 0.011 |
| ma_slope_50_pctrank500 | -0.011 | -0.005 | 0.011 |
| mfi_100 | -0.011 | -0.001 | 0.011 |
| skew_returns_50_pctrank500 | +0.002 | +0.011 | 0.011 |
| mfi_20 | -0.011 | -0.003 | 0.011 |
| adx_slope_100_pctrank500 | -0.003 | +0.011 | 0.011 |
| stochastic_d_100_pctrank2000 | -0.001 | -0.011 | 0.011 |
| dist_from_high_100_pctrank500 | +0.011 | +0.001 | 0.011 |
| skew_returns_10 | -0.001 | +0.010 | 0.010 |
| aroon_down_10_pctrank500 | +0.010 | +0.001 | 0.010 |
| stochastic_k_100_pctrank2000 | -0.001 | -0.010 | 0.010 |
| williams_r_100_pctrank2000 | -0.001 | -0.010 | 0.010 |
| donchian_position_100_pctrank2000 | -0.001 | -0.010 | 0.010 |
| avg_gap_20 | -0.010 | +0.002 | 0.010 |

（另有97个因子|IC|<0.01，判定为无区分力，未列入候选池，完整名单也在CSV里，标记为未通过阈值）

## TOP10因子分层分析（1天(24根H1)未来ER均值，按因子五分位）

### bb_width_50

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.248 | 21602 |
| 1 | 0.241 | 21602 |
| 2 | 0.227 | 21602 |
| 3 | 0.220 | 21602 |
| 4 | 0.206 | 21602 |

### bb_width_50_pctrank500

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.250 | 21523 |
| 1 | 0.236 | 21634 |
| 2 | 0.226 | 21497 |
| 3 | 0.222 | 21525 |
| 4 | 0.208 | 21332 |

### bb_width_50_pctrank2000

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.250 | 21207 |
| 1 | 0.236 | 21254 |
| 2 | 0.228 | 21176 |
| 3 | 0.221 | 21178 |
| 4 | 0.207 | 21196 |

### bb_width_20

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.247 | 21608 |
| 1 | 0.240 | 21608 |
| 2 | 0.229 | 21608 |
| 3 | 0.222 | 21608 |
| 4 | 0.204 | 21608 |

### bb_width_20_pctrank2000

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.249 | 21244 |
| 1 | 0.236 | 21210 |
| 2 | 0.229 | 21200 |
| 3 | 0.224 | 21208 |
| 4 | 0.205 | 21179 |

### adx_20

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.247 | 21605 |
| 1 | 0.237 | 21604 |
| 2 | 0.229 | 21604 |
| 3 | 0.222 | 21604 |
| 4 | 0.207 | 21604 |

### choppiness_index_50

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.209 | 21602 |
| 1 | 0.224 | 21602 |
| 2 | 0.224 | 21602 |
| 3 | 0.235 | 21602 |
| 4 | 0.250 | 21602 |

### bb_width_20_pctrank500

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.248 | 21584 |
| 1 | 0.236 | 21611 |
| 2 | 0.229 | 21454 |
| 3 | 0.221 | 21447 |
| 4 | 0.207 | 21445 |

### choppiness_index_50_pctrank2000

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.210 | 21246 |
| 1 | 0.223 | 21198 |
| 2 | 0.224 | 21194 |
| 3 | 0.237 | 21207 |
| 4 | 0.249 | 21166 |

### adx_20_pctrank2000

| 五分位(0=最低) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.247 | 21256 |
| 1 | 0.236 | 21157 |
| 2 | 0.228 | 21235 |
| 3 | 0.225 | 21175 |
| 4 | 0.208 | 21199 |

## 时段(session) / 星期 与 1天(24根H1)未来ER 的关系（非数值因子，用分组均值代替相关系数；session划分未经独立UTC核实，结论暂视为初步）

### session

| session | 未来ER均值 |
|---|---|
| asia | 0.227 |
| late | 0.228 |
| ny_overlap | 0.228 |
| ny_only | 0.229 |
| london_open | 0.230 |

### day_of_week

| day_of_week | 未来ER均值 |
|---|---|
| 6 | 0.219 |
| 1 | 0.224 |
| 0 | 0.225 |
| 2 | 0.226 |
| 5 | 0.226 |
| 3 | 0.232 |
| 4 | 0.237 |

## 系统性两两组合搜索（从15个“每个指标家族里IC最强的代表因子”中两两配对，各自取20%“安全”分位，看同时成立时未来ER相对基准的降幅，按降幅排序取前10；这是穷举而不是像v1~v3那样手选一对）

代表因子: bb_width_50, adx_20, choppiness_index_50, keltner_width_100_pctrank2000, atr_100_pctrank2000, h4_bb_width_50, parkinson_vol_100_pctrank2000, h4_choppiness_index_20, garman_klass_vol_100_pctrank2000, efficiency_ratio_20, realized_vol_100_pctrank2000, autocorr_returns_100, variance_ratio_2_100, linreg_r2_20, dist_from_low_50_pctrank2000

| 因子A | 因子B | 样本占比 | 未来ER均值 | 相对基准降幅 |
|---|---|---|---|---|
| bb_width_50 | efficiency_ratio_20 | 6.3% | 0.188 | +17.7% |
| bb_width_50 | linreg_r2_20 | 5.4% | 0.188 | +17.5% |
| choppiness_index_50 | atr_100_pctrank2000 | 4.7% | 0.189 | +17.3% |
| choppiness_index_50 | garman_klass_vol_100_pctrank2000 | 4.9% | 0.191 | +16.6% |
| h4_choppiness_index_20 | efficiency_ratio_20 | 5.7% | 0.192 | +16.0% |
| efficiency_ratio_20 | realized_vol_100_pctrank2000 | 4.4% | 0.192 | +15.9% |
| parkinson_vol_100_pctrank2000 | efficiency_ratio_20 | 4.2% | 0.193 | +15.7% |
| choppiness_index_50 | keltner_width_100_pctrank2000 | 4.7% | 0.193 | +15.4% |
| choppiness_index_50 | parkinson_vol_100_pctrank2000 | 5.2% | 0.194 | +15.1% |
| choppiness_index_50 | efficiency_ratio_20 | 8.3% | 0.194 | +15.1% |

（全样本未来ER基准均值: 0.228；样本数<500的组合已剔除，不然小样本均值不稳定容易排到前面制造假象）

## 多因子合成打分：取IC最强的8个代表因子，每个按“安全方向”转成0~1的历史分位（1=最安全），取平均作为一个综合regime分数，再看它本身的IC和五分位分层

合成因子: bb_width_50, adx_20, choppiness_index_50, keltner_width_100_pctrank2000, atr_100_pctrank2000, h4_bb_width_50, parkinson_vol_100_pctrank2000, h4_choppiness_index_20

- 合成分数 vs 1天(24根H1)未来ER 的Spearman IC: -0.087（对比单因子最强的bb_width_50: -0.087）

| 五分位(0=最危险,4=最安全) | 未来ER均值 | 样本数 |
|---|---|---|
| 0 | 0.251 | 21605 |
| 1 | 0.234 | 21604 |
| 2 | 0.225 | 21604 |
| 3 | 0.226 | 21604 |
| 4 | 0.206 | 21604 |

- 合成打分最高分位（最“安全”20%）相对基准降幅: +9.9%，对比两两组合里最好的一对（降幅+17.7%）——两两组合反而更强，说明简单平均稀释了强因子的信号，不如直接用组合过滤器

## 结论与下一步

- 本轮共构建383个数值因子，286个通过|IC|>=0.01的候选池筛选，97个未通过；候选池整体仍以“波动率/趋势强度类”因子（bb_width、keltner_width、adx、h4_adx、h4_bb_width）为主，说明真正有效的regime信息集中在这一类，扩大搜索范围并没有找出量级更强的新因子（max|IC|依然在0.05~0.10区间），只是把同类信息用更多参数化形式重新表达了一遍——这本身也是一个有用的结论：不必再花力气在方向类(RSI/zscore/ma_slope)或分布形态类(skew/kurt)因子上，它们持续垫底。
- |IC|数值虽然普遍不高，但样本量巨大（10万+根H1），且五分位分层单调、跨H1/H4双周期一致，说明这是稳定的统计规律而非噪音；不过|IC|~0.05~0.10对应的可解释方差不到1%，单独使用任何一个因子都不构成可交易的强信号，必须像候选池里已验证的“组合过滤器”那样叠加使用。
- 候选池CSV会传给阶段3，用于在真实马丁资金曲线回测里做特征选择/组合，而不是直接把静态相关性当结论。
- MFI（资金流量指标）偏弱（max|IC|约0.02~0.03），符合预期：我们的volume是tick数量代理而非真实成交量，量价类指标在这份数据上先天打折扣，不建议作为主力因子。
- 百分位排名的价值分窗口而定：对20/50根这种较短窗口，pctrank版本和原始值IC几乎一样（说明短窗口本身已经是局部相对值，金价长期涨幅带来的尺度漂移影响不大）；但对100根这种较长窗口，pctrank版本明显强于原始值（如atr_100、keltner_width_100，max|IC|从~0.055~0.061提升到~0.076），说明长窗口的原始指标确实受金价从860到5300+的尺度漂移污染，百分位排名修正了这个问题——这是本轮扩大范围里少数几个“方法改进直接带来更强因子”的例子，值得在阶段3优先使用这些pctrank_2000版本而非同名原始版本。
- 仍然只用了价格衍生的技术类因子（含H1自身+H4更高周期），没有引入跨市场/宏观数据（本地目前只有XAUUSD自身行情）；如果后续要加美元指数/美债收益率/VIX等跨市场因子，需要额外的数据源。
- v4新增的Choppiness Index表现符合预期地强（0.081，跻身候选池前列），且频繁出现在系统性搜索出的最佳组合里，证明“专门为这个问题设计的指标”确实比通用技术指标更有效，这比v1~v3的泛化搜索更有针对性。注意它的IC符号和bb_width/adx相反（当前越“choppy”→未来ER越高），这不是矛盾：结合两者看，故事是regime会交替——当前波动率已经放大/趋势已经很强时，未来更可能“歇一歇”变震荡(bb_width/adx的发现)；当前处于窄幅盘整时，未来更可能变成突破趋势(choppiness_index的发现)。两个独立构造的指标从不同角度印证了同一个“波动率/趋势会均值回归”的市场现象，互相印证比单独看更可信。
- 系统性两两组合搜索（穷举15个代表因子的组合，而非手选一对）找到了比v1~v3手选组合更好的结果：bb_width_50+efficiency_ratio_20能把未来ER压低约17.7%（v1~v3手选的bb_width+adx组合约15%），说明系统性搜索确实有必要，手选容易漏掉更优组合。
- 但8因子平均合成打分并不比两两组合更好——合成分数整体IC(-0.087)和单用bb_width_50几乎一样，最高安全分位的ER降幅(9.9%)反而不如最优两两组合(17.7%)，说明简单平均会把强因子的信号稀释掉，阶段3不建议用“一堆因子取平均”的合成分数，应该用穷举验证过的两因子(或阶段3回测里可以再试三因子)AND过滤器。

