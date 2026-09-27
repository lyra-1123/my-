# 模拟盘（前向测试）

## 目录
| 文件 | 作用 |
|---|---|
| `paper/specs.py` | 策略规格（冻结）：因子组合、参数、频率、进出场、仓位、前向起始日、证据 |
| `paper/engine.py` | 引擎：读数据（丢弃未走完的 K 线）→ 算信号 → 模拟成交 → 账本；成交与成本模型与研究评估完全一致 |
| `paper/onboard.py` | 准入与登记：证据检查、与在跑策略的相关性和组合贡献；`--register` 写入 prereg.json（行为指纹 + 预期区间 + 评估标准） |
| `paper/run.py` | 日常运行：校验指纹，输出下一根 K 线开盘的操作，更新前向账本 |
| `paper/report.py` | 报告：对照登记时写死的区间给出状态，汇总组合，统计实际成交的滑点 → `paper/state/REPORT.md` |
| `paper/selftest.py` | 自检：与研究回测对账、指纹敏感性、未走完 K 线、前向流程 |
| `paper/state/<id>/` | `prereg.json`（登记后不可改）、`signal.json`、`trades.csv`、`daily.csv`、`fills_manual.csv`（手工记录实际成交） |

## 日常流程
1. 更新数据：在本机运行 Dukascopy 导出脚本，把最新的 `DAT_ASCII_XAUUSD_M1_2026.csv` 放进 `data/`（或上传到 Google Drive 后运行 `bash scripts/fetch_data.sh`）。
2. `python -m paper.run` → 看每个策略"下一根开盘的操作"，并更新前向账本。
3. 如果实际下单了，把成交记到 `paper/state/<id>/fills_manual.csv`。
4. 每周 `python -m paper.report`，提交 `paper/state/`（账本随仓库保存）。

两种用法：
- **模型前向测试**（当前方式）：每天或每周运行一次即可。引擎会从 forward_start 起按规则重放新数据，得到"如果严格按规则执行"的账本。它检验的是策略本身，不需要实时盯盘。
- **实时执行**：30MIN 策略每半小时都可能产生操作，需要每根 K 线收盘后运行一次，并要求数据实时更新。按天导出 Dukascopy 数据做不到这一点，需要接入 MT5 等实时数据源（尚未实现）。

## 评估标准（每个策略登记时写死，见 prereg.json）
- 失败：前向累计 R 低于同长度窗口（63/126/252 个交易日）回测分布的 5% 分位 → 停止该策略并复查。
- 警告：低于中位数，或前向最大回撤超过回测最大回撤的 1.5 倍。
- 预期：以 CSCV 样本外夏普中位数（TT30-EW-v1 为 0.41）为基准，而不是全样本回测值。
- 252 个交易日之前不对"有效"下结论。

## 新因子 / 新策略加入模拟盘
1. 因子库评估（`python -m factors.evaluate`）、ICIR 与相关性（`research/icir_clusters.py`、`research/factor_correlation.py`）、稳健性（参数平原、walk-forward、成本）、过拟合检验（`research/overfitting_tests.py`：研究层 PBO、DSR）。结果写进 spec 的 `evidence`（至少包含 `cscv_oos_sharpe_median`、`research_PBO`）。
2. 在 `paper/specs.py` 追加规格（status="candidate"，id 带版本号，forward_start 设为下一个交易日）。
3. `python -m paper.selftest`，然后 `python -m paper.onboard <id>`：要求与所有在跑策略的日度 R 相关系数 < 0.5，且加入后组合夏普不下降。
4. 通过后 `python -m paper.onboard <id> --register`，把 status 改为 "active"。
5. **不能修改在跑策略**：需要改动时新增版本（-v2）重新登记，旧版本改为 "retired"；它的账本保留。行为指纹会阻止无意中的改动（例如改了 `factors/core.py` 的公共函数）。
