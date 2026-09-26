# 03_回测引擎

第 5 章"回测层：防御过拟合"的通用框架。只有通过了第 4 章验证三件套的候选信号才送进来。

| 文件 | 作用 |
|---|---|
| `cost_model.py` | 交易成本（点差 + 双边滑点）。**目前是占位值**：点差 $0.30 + 2×$0.05，接实盘前换成真实券商数据 |
| `backtest_engine.py` | 信号 → 成交：信号bar收盘价进场，SL/TP/HOLD出场（同bar先判SL，保守），持仓期间忽略新信号 |
| `wf_runner.py` | Walk-Forward 窗口：按实际成交频率反推 OOS 天数（≥300笔/窗口），窗口不重叠，连续≥6个通过 |
| `cscv_pbo.py` | CSCV/PBO（变体家族内 IS 最优者 OOS 落到中位数以下的概率）+ DSR（多重检验折扣） |
| `regime_split.py` | 多 Regime：ATR 波动率高/中/低，ADX 趋势/震荡 |
| `chapter5_pipeline.py` | 串起全部检验 + 指南 5.3.4 的 5 项综合判定（PASS / WARN / FAIL） |
| `validate_ch5_<假设>.py` | 每个候选假设一个入口脚本 |

## 新假设怎么接入

1. 信号脚本拆成 `build_features(df)`（只算一次指标）+ `signal_from_features(feat, 参数...)`（返回 {-1,0,1}）
2. 新建 `validate_ch5_<假设名>.py`：
   - 候选 = 第 4 章通过的那组参数
   - 变体家族 = 候选周围"研究员自然会试"的参数邻域（PBO 和 WF 选择法用）
   - `N_TRIALS_PRIOR` = 本项目之前在全量数据上正式验证过的策略定义数（**只增不减**，每验证一个新假设都要加）
3. 调 `run_chapter5_validation()` + `print_chapter5_report()`

## 已知简化（v1）

- 出场用对称 SL=TP=目标值，是第 6 章风控之前的临时替代
- DSR 用逐笔 Sharpe，基准 `sqrt(2·ln N)/sqrt(T)`，与指南 5.3.2 的 `sqrt(log N)` 不同（单位一致性，见 `cscv_pbo.calc_dsr` 注释）
- PBO 的逐bar收益矩阵把每笔收益记在出场bar上，只适合持仓短的低频信号
