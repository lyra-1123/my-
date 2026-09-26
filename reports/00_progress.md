# 项目进度总览

XAUUSD量化马丁格尔策略：数据清洗 → 因子挖掘 → 回测 → 策略成型 → 多agent审核 → 策略运行 → 复盘。

约定：每个阶段对应一个编号脚本（`scripts/NN_xxx.py`）+ 一份编号报告（`reports/NN_xxx_report.md`），
报告记录该阶段的输入、方法、结论和遗留问题，脚本可重复运行复现该阶段的产出。本文件是索引，
每个阶段完成/更新后同步维护。

| 阶段 | 状态 | 脚本 | 报告 | 一句话结论 |
|---|---|---|---|---|
| 1. 数据清洗 | ✅ 完成 | `scripts/01_build_clean_dataset.py` | `reports/01_data_quality_report.md` | 635万条M1数据，0坏点、0未解释缺口，已产出M1~D1多周期parquet |
| 2. 因子挖掘 | ✅ 完成(v4，新因子类型+全面百分位化+系统性组合搜索) | `scripts/02_factor_mining.py` | `reports/02_factor_mining_report.md` + `reports/02_factor_candidate_pool.csv` | 383个因子中286个通过\|IC\|≥0.01；新增Choppiness Index/Aroon/Parkinson-GK波动率/linreg_r²/avg_gap，其中Choppiness Index表现强(\|IC\|~0.08)且符号与bb_width/adx相反，两者互相印证"波动率/趋势会均值回归"；穷举15个代表因子的两两组合，最优对(bb_width_50+efficiency_ratio_20)把未来ER压低17.7%，好于v1~v3手选组合(~15%)；8因子平均合成打分反而不如两两组合(9.9%<17.7%，简单平均稀释信号)，阶段3不建议用平均合成分数；session/星期几乎无区分力 |
| 2b. 因子分类筛选(IC+IR+PBO+OOS Sharpe) | ✅ 完成(v6) | `scripts/02b_factor_screening.py` | `reports/02b_factor_screening_report.md` + `reports/02b_factor_screening_results.csv` | v1→v5见历史(12→2→11→0→0/38，v4/v5的IC+IR联合筛选让4h/8h的通过名单完全不重叠)；**v6只用IR≥0.3做门槛(IC/PBO/OOS Sharpe降级为参考指标)后，4h通过名单变成8h通过名单的子集，"稳健核心"回升到4个：atr、autocorr_returns、bb_width、choppiness_index**；但其中只有`autocorr_returns`和`choppiness_index`在4h和8h上OOS Sharpe同时为正(atr/bb_width的8h OOS Sharpe为负，IR稳定不代表能赚钱)——**最终推荐的单因子候选是这2个**。阶段3计划两条线并行：单因子用这2个，同时把阶段2 v4验证过的组合过滤器(bb_width_50+adx_20，压低未来ER15~18%)接入真实马丁回测做对照 |
| 3. 回测(基线，无过滤器) | ✅ 完成 | `scripts/03_baseline_backtest.py` | `reports/03_baseline_backtest_report.md` | 长仓ATR网格马丁(初始0.01手/2倍加仓/最多8层/1xATR(14)间距和止盈/$10000初始资金/1:200杠杆)：权益从$10000稳定涨到峰值$104,961(2012-10-09)，随后半年内回撤95%到$4,787，最终被2013年4月中旬黄金历史级暴跌一根H1 bar打出-$105,418强平，账户**破产**(2013-04-15)，此后不再交易。验证了马丁格尔的核心风险：稳定盈利可以持续数年，但尾部风险一次性摧毁全部收益。这是阶段3b(接入11个regime因子做入场过滤)的对照组基准 |
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
- 阶段2b最终结论（v6）：单因子regime过滤器，能同时满足"4h/8h跨horizon稳定(IR)"+"OOS
  Sharpe为正"的只有`autocorr_returns`和`choppiness_index`，阶段3应以这2个为单因子候选，
  同时并行测试阶段2 v4的组合过滤器（bb_width_50+adx_20）作为对照，不要只押注单因子路线。
- 因子库已做过全量look-ahead审计（31个H1+4个H4指标逐行检查）：全部只用backward
  `.rolling()`/`.shift(正数)`/`.diff()`/causal`.ewm()`，没有发现任何未来函数。唯一用到
  未来数据的是`labels.py`的前向标签和验证用的测试床收益，两者都明确只用于评估、不作为
  任何决策输入。
