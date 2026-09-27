# 03_回测引擎

第 5 章"回测层：防御过拟合"的通用框架。只有通过了第 4 章验证三件套的候选信号才送进来。

| 文件 | 作用 |
|---|---|
| `cost_model.py` | 交易成本（点差 + 双边滑点）。**目前是占位值**：点差 $0.30 + 2×$0.05，接实盘前换成真实券商数据 |
| `lib/goldq/exits.py` | **出场规则唯一实现**（第4、5章共用）：止损{结构, ATR移动, ATR固定} × 止盈{R倍数, 不设, 指标反转, 分批}，时间/缺口出场，跳空按开盘价成交 |
| `backtest_engine.py` | 信号 → 成交：逐笔调用 `simulate_trade()`，持仓期间忽略新信号，扣成本 |
| `wf_runner.py` | Walk-Forward 窗口：按实际成交频率反推 OOS 天数（≥300笔/窗口），窗口不重叠，连续≥6个通过 |
| `cscv_pbo.py` | CSCV/PBO（变体家族内 IS 最优者 OOS 落到中位数以下的概率）+ DSR（多重检验折扣） |
| `regime_split.py` | 多 Regime：ATR 波动率高/中/低，ADX 趋势/震荡 |
| `chapter5_pipeline.py` | 串起全部检验 + 指南 5.3.4 的 5 项综合判定（PASS / WARN / FAIL） |
| `validate_ch5_<假设>.py` | 每个候选假设一个入口脚本 |

## 新假设怎么接入

1. 信号脚本拆成 `build_features(df)`（只算一次指标）+ `signal_from_features(feat, 参数...)`（返回 {-1,0,1}）
2. 第4章先跑方向性筛选（`signal_validation.evaluate_exit_direction`，参考
   `04_策略研究/validate_exits_atr_momentum_v2.py`）：同一批信号bar上信号方向 vs 随机方向，
   方向边际 t≥2 且成本后R>0 才进第5章
3. 新建 `validate_ch5_<假设名>.py`：
   - `market = prepare_market(feat, exit_long, exit_short)`（用指标反转出场才需要后两个）
   - 候选 = 第 4 章通过的那组（信号参数 + 出场规则）
   - 变体家族 = `Variant(name, signal, exit_rule)` 列表，候选周围"研究员自然会试"的邻域
   - `N_TRIALS_PRIOR` = 本家族之前已正式验证过的策略定义数（不含本家族），DSR 用 prior + 家族大小。
     **只增不减**，当前累计见 `00_方案/changelog.md`
4. 调 `run_chapter5_validation()` + `print_chapter5_report()`

## 已知简化（v1）

- 仓位固定 1 单位，没有按风险定仓（第 6 章）；分批出场按 1 单位内的比例计
- 同一根 bar 内止损与止盈/分批都触及时一律按止损记（M5 内部先后未知，偏保守）
- DSR 用逐笔 Sharpe，基准 `sqrt(2·ln N)/sqrt(T)`，与指南 5.3.2 的 `sqrt(log N)` 不同（单位一致性，见 `cscv_pbo.calc_dsr` 注释）
- PBO 的逐bar收益矩阵把每笔收益记在出场bar上，只适合持仓短的低频信号
