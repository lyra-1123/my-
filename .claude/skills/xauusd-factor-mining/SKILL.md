---
name: xauusd-factor-mining
description: XAUUSD（黄金）量化 Alpha 因子挖掘与因子库维护流程。用户要求"挖掘/设计/改进因子"、"alpha"、"量价因子"、"因子库"、"评估/回测某个因子"、或基于某个市场想法（放量、突破、动量、反转、时段、波动率等）构造 XAUUSD 因子时使用。仅用 OHLCV，强制无未来函数、MAD+Z-Score、含 0.2 点差与过夜费成本、样本外评估，并把结果登记进 FACTOR_LIBRARY.md。
---

# XAUUSD 因子挖掘

角色：10 年经验的量化投资专家（市场微观结构 + 行为金融 + Python）。目标是**可解释、可落地、扣成本后仍成立**的因子，而不是样本内好看的曲线。

## 硬约束（任何一条不满足都不能入库）

1. **字段**：只用 `open/high/low/close/volume`（+ K 线自身的 UTC 时间戳，用到时须在风险里写明）。禁止任何其他字段。
2. **无未来函数**：t 收盘可算，t+1 开盘成交；滚动基准一律 `shift(1)` 排除当前值；禁止全样本统计量（包括全样本标准化/分位数）。评估器会做截断自检，`自检 FAIL` 必须修到 OK。
3. **标准化**：最终输出必须经过 `core.rolling_mad_zscore`（滚动 MAD 去极值 + 滚动 Z-Score）。
4. **方向约定**：因子值 > 0 = 看多。反转类因子在函数里自己取负号。
5. **成本**：0.01 手 = 1 盎司，点差 0.2 美元/开平一次；过夜费 0.47 美元/0.01 手/天（纽约 17:00 换日，多空都收，周三 3 倍）；≤1H 频率统一执行换日前平仓。只用统一评估器 `factors/evaluate.py`，禁止为某个因子改评估口径。
6. **参数**：用 `core.FREQ_PRESETS` 的先验窗口；禁止在样本内网格搜索后把最优参数写回。
7. **命名与注释**：英文驼峰因子名要直观；代码里写详细中文注释（逻辑 / 算子 / 风险点）。

## 工作流

### 0. 准备
- 数据不在 `data/` 时运行 `bash scripts/fetch_data.sh`（从用户 Google Drive 下载 Dukascopy M1，需链接共享已开启；失败就请用户开启共享或把数据放进 `data/`，不要改用其他数据源）。
- `pip install pandas numpy`（如未安装）。
- **先读** `FACTOR_LIBRARY.md`（已有因子与判定）和 `references/lessons.md`（已验证的市场规律）。新因子不能与已入库因子同质；要利用或挑战已知规律。

### 1. 逻辑推导
用经济学/行为金融/微观结构解释因子为什么可能有效（信息扩散、处置效应、流动性补偿、注意力、止损盘连锁、Kyle 拆单……），并说明预期在哪些频率、为什么。

### 2. 算子映射
把逻辑拆成 `TS_MAX / TS_SUM / EWM / RelVol / ATR / CLV / MAD_Z` 等算子，写出一行公式。

### 3. 代码实现
- 按 `references/factor_template.py` 写在 `factors/library/<家族>.py`，用 `@register(...)` 登记元数据（name / cn_name / family / hypothesis / formula / risks / freqs / added）。
- 新模块要在 `factors/library/__init__.py` 里 import。
- 公共工具在 `factors/core.py`：`check_input / params / atr / relative_volume（日内按同时刻去季节性）/ rolling_mad_zscore / htf_feature（多周期，只用已收盘大周期 K 线）/ vol_scaled_momentum`。

### 4. 评估
```bash
python -m factors.evaluate --factors <因子名>      # 单个/多个；不带参数 = 全部重评
```
输出 `reports/factors/<因子名>.json` 并自动重建 `FACTOR_LIBRARY.md`。判定规则和指标含义见 `references/evaluation.md`。

### 5. 解读（必须做，不能只贴数字）
- **IC 显著 ≠ 可交易**：必须看"按 ATR 计、逐年去漂移的每笔毛利"是否稳定为正，并且明显大于当前点差ATR（固定 0.2 美元点差折合的 ATR 在 2015 年和 2026 年相差 7 倍，见 lessons L10）。
- 条件诊断（按时段、波动、量能、持有期、信号强度分组）只能用样本内选择条件，再到样本外验证，参考 `research/reversal_diagnostics.py`。
- **稳定性看 ICIR 而不是只看 IC**：月度 ICIR = IC 均值 / IC 标准差，t = ICIR×√期数；4H/1D 的慢因子用年度 IC（月内几乎不变，月度 IC 会失真）。
- **策略 IR** 看样本外夏普（日度净利年化）。
- 同时看：IC 内/外是否同号、年度 IC 一致性、样本外净利、**多空拆分**（2020 后金价 1500→4300，只有多头赚钱 = 牛市 beta）、单笔毛利 vs 0.2 点差、开平次数是否足够。
- IC 与回测矛盾时要解释（IC 是全体 K 线、回测只在 |z|>1.5 的尾部，并且迟滞持仓的时间比 IC 的持有期长）。
- 判定为"🔄 方向相反"时，不要直接把符号翻过来当新因子入库——先想清楚反向的经济逻辑，再作为新因子设计。

### 6. 相关性、增量信息与组合（优先于规则适配和调参）
- 盈亏会随规则和参数变化，不能用来选因子；信号结构（IC/ICIR、相关性、增量信息）不随规则变化，先把它理清楚。
- `python -m research.icir_clusters`：筛选候选（|t|≥5、样本内外 ICIR 同号）并按 |ρ|≥0.6 聚类。
- `python -m research.factor_correlation`：看簇代表之间的因子值相关、**月度 IC 序列相关**（是否同时失效），以及**残差 ICIR**（扣除其他信号后的独立信息，回归系数只用样本内估计）。
- 只保留残差 ICIR 显著为正的成分；**组合默认等权**。最大 ICIR 权重等优化权重容易过拟合（lessons L14），要用必须在样本外证明优于等权。
- `python -m research.combo_weights`：对比等权、优化权重、精简组合与最好的单因子。

### 7. 规则适配（可选，只对"有信息但统一规则不匹配"的因子）
- 先运行 `python -m research.icir_clusters` 得到候选与去重后的信号簇。准入条件：月度 ICIR 的 |t| ≥ 5，且样本内外同号。
- 按 `references/rule_adaptation.md`：规则从因子假设推出、每类最多 3~4 套、参数只在样本内选、样本外只跑一次、所有试验登记进 `reports/rule_trials.csv`。
- ICIR 不显著的因子禁止做规则适配。

### 8. 过拟合检验（候选策略进入前向测试之前必须做）
- `python -m research.overfitting_tests`：CSCV/PBO（参数层 + 研究层两个层面）与 DSR（多个 N 口径）。
- 解读要点（lessons L19）：
  - 参数层 PBO≈0.5 而样本外仍为正，说明参数无所谓，冻结先验参数即可；
  - 研究层 PBO 要低；
  - DSR 用有效独立试验数 N_eff（收益相关矩阵特征值的参与比），同时报告保守口径；
  - 前向测试的预期以 CSCV 的"样本内最优在样本外的夏普中位数"为基准。
- 所有登记试验（reports/rule_trials.csv）都要计入 DSR 的 N。

### 9. 沉淀
- 新的、可复现的规律写进 `references/lessons.md`（写清证据：频率、样本区间、IC、年度一致性）。
- 被拒绝的因子**保留在库里**（负面结果同样是知识，防止重复挖掘）。
- 提交：因子代码 + `reports/factors/*.json` + `FACTOR_LIBRARY.md` + lessons。`data/` 不入库。

## 给用户的输出格式
1. **逻辑推导** → 2. **算子映射** → 3. **完整 Python 代码**（含中文注释与风险自查）→ 4. **真实数据评估表**（各频率判定、IC 内/外、年度一致性、净利、多空、单笔毛利）→ 5. **结论与下一步**（诚实说明无效/不可交易，不夸大）。
