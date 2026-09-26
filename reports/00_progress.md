# 项目进度总览

XAUUSD量化马丁格尔策略：数据清洗 → 因子挖掘 → 回测 → 策略成型 → 多agent审核 → 策略运行 → 复盘。

约定：每个阶段对应一个编号脚本（`scripts/NN_xxx.py`）+ 一份编号报告（`reports/NN_xxx_report.md`），
报告记录该阶段的输入、方法、结论和遗留问题，脚本可重复运行复现该阶段的产出。本文件是索引，
每个阶段完成/更新后同步维护。

| 阶段 | 状态 | 脚本 | 报告 | 一句话结论 |
|---|---|---|---|---|
| 1. 数据清洗 | ✅ 完成 | `scripts/01_build_clean_dataset.py` | `reports/01_data_quality_report.md` | 635万条M1数据，0坏点、0未解释缺口，已产出M1~D1多周期parquet |
| 2. 因子挖掘 | ✅ 完成(v4，新因子类型+全面百分位化+系统性组合搜索) | `scripts/02_factor_mining.py` | `reports/02_factor_mining_report.md` + `reports/02_factor_candidate_pool.csv` | 383个因子中286个通过\|IC\|≥0.01；新增Choppiness Index/Aroon/Parkinson-GK波动率/linreg_r²/avg_gap，其中Choppiness Index表现强(\|IC\|~0.08)且符号与bb_width/adx相反，两者互相印证"波动率/趋势会均值回归"；穷举15个代表因子的两两组合，最优对(bb_width_50+efficiency_ratio_20)把未来ER压低17.7%，好于v1~v3手选组合(~15%)；8因子平均合成打分反而不如两两组合(9.9%<17.7%，简单平均稀释信号)，阶段3不建议用平均合成分数；session/星期几乎无区分力 |
| 2b. 因子分类筛选(PBO+OOS Sharpe) | ✅ 完成 | `scripts/02b_factor_screening.py` | `reports/02b_factor_screening_report.md` + `reports/02b_factor_screening_results.csv` | 38个因子家族按7类(均值回归/动量/波动率/趋势强度/价格行为/量能/更高周期)分类筛选，PBO≤0.5且walk-forward OOS Sharpe>0的双重标准下**只有12个通过**；和纯IC筛选结论有明显分歧——v1~v4里最强的efficiency_ratio/realized_vol/atr/keltner_width等全部未通过，反而是MFI、streak_length、macd_hist这类IC弱的通过了；发现donchian_position/stochastic_k/williams_r是同一信号的仿射变换(结果完全相同)，阶段3应只留一个 |
| 3. 回测 | 未开始 | - | - | - |
| 4. 策略成型 | 未开始 | - | - | - |
| 5. 多agent审核 | 未开始 | - | - | - |
| 6. 策略运行 | 未开始 | - | - | - |
| 7. 复盘 | 未开始 | - | - | - |

## 关键假设/待核实事项（跨阶段持续跟踪）

- 时间戳假定为UTC（沿用原有导出脚本的默认假设），尚未用独立的实时UTC行情源交叉核实，
  session/小时因子上线前应补一次核实。
- 数据源：2009-2016为Dukascopy(`_DUKAREAL`)，2017年至今为Dukascopy导出，volume字段为
  tick成交量代理，非真实成交量，量价类因子需谨慎解读。
- 目标运行环境：MT4/MT5经纪商账户，标准杠杆(1:100~1:500)，回测的保证金/爆仓/隔夜利息
  模型需按此假设设计（阶段3会明确参数来源，暂用可配置的通用参数）。
- 阶段2产出的候选regime过滤器（bb_width_24与adx_14同时处于各自20年历史高位20%分位）
  待阶段3在真实回测（而非静态相关性）里验证是否真的能降低马丁网格的爆仓概率，
  且要检查触发频率（历史上约10%的时间满足条件）是否会让策略常年空仓。
- 阶段2b用更严格的PBO+walk-forward OOS Sharpe筛选后，只有12个因子家族通过，且和阶段2
  纯IC筛选的结论有明显分歧——阶段3选因子应以阶段2b的12个通过项为准，阶段2的候选池
  （尤其是efficiency_ratio、realized_vol、atr等在2b里未通过的）仅供参考，不要直接用。
