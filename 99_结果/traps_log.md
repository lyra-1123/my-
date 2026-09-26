# Trap 沉淀日志

按指南第 8 章"复盘双轨制"维护：这里是 md 记录，重复出现 3 次以上的 Trap 应再沉淀为
`~/.claude/skills/` 或项目内 Skill（本仓库暂未建立，等实际踩坑后再建）。

## 数据自检结论（2026-09-26，真实全量数据 2008-12-31~2026-09-25，635万行）

跑完 `build_dataset.py` + `validate_data.py` + `inspect_year.py` 人工抽查后的结论：
**数据整体健康，可以进入第 4 章（先写 Hypothesis First）**。细节：

- 重复时间戳：0
- 日均 bars：排除圣诞/元旦固定假期后，无异常低量交易日
- 缺口：2831 处 >30 分钟缺口里 86% 集中在 UTC 22-23 点，是 broker 每日固定 rollover 停牌，非数据丢失；
  >3 小时的大缺口共 86 处，大部分对应美股节假日（MLK/总统日/阵亡将士纪念日/劳动节/独立日），符合预期
- **遗留点 1**：`2014-08-25`（缺口 53 小时）、`2014-03-03`（缺口 50 小时）两处大缺口不对应已知假期，
  原因未知（可能是 2014 年 Dukascopy 历史数据源本身的缺口）。不影响现在，等回测/训练窗口
  真的覆盖到 2014 年那两段时间时再决定是跳过还是补数据。
- **遗留点 2**：2009 年（`DAT_ASCII_XAUUSD_M1_2009_DUKAREAL.csv`）成交量分布的 peak hour 是 UTC 0 点，
  和其他年份的 13-14 点差异较大，一度怀疑时区错位。用 `inspect_year.py` 人工核查后判断更可能是
  **数据本身稀疏**（2009 年 60.5% 的行 volume=0，全年总成交量只有 2010 年的三分之一，
  容易被几笔大单主导统计），而非系统性时区偏移：抽样价格（2009-07 ~$940、2010-07 ~$1200）
  与黄金实际历史价位吻合，且两年 bar 总数量级一致，说明日期本身没有错位。
  建议：早期年份（2009 及更早，如果之后再往前补数据）在回测里适当降权或标注"低流动性期"，
  不需要现在返工数据管道。

## 已知/继承的 Trap（来自项目初始化审查）

### Trap-001（本项目实证，2026-09-26）：mt5_export_direct.py 时区偏移未校准

- **现象**：`mt5_export_direct.py` 直接把 MT5 `copy_rates_range` 返回的时间戳当作 UTC 写入
  Dukascopy 格式 CSV，但 MT5 返回的实际是 broker 服务器时间（不同 broker 偏移不同，常见 UTC+0/+2/+3）。
  同目录下的 `dukascopy_export_2017_now.py` / `dukascopy_export_historical.py`
  （来自 `dukascopy_python.fetch()`）返回的才是真 UTC。
- **风险**：如果两类文件被 `build_dataset.py` 无差别合并，会出现同一时间点被记录成两个不同的
  K 线时间戳（错位若干小时），Walk-Forward / 信号回测会静默产生前视偏差或时段错配。
- **防御**：
  1. `07_配置参数/data_paths.yaml` 的 `mt5_utc_offset_hours` 默认给的是占位值 0，
     **禁止在未核实前直接使用**。
  2. 上线前必须跑 `mt5_fill_volume.py` 里的 `verify_overlap()`，用真实重叠时间段的收盘价
     比对来确认偏移量。
  3. `validate_data.py` 的 `check_hourly_distribution_by_year()` 会做跨年小时分布 sanity check，
     发现可疑年份需人工抽查。
- **状态**：待用户在本地环境用真实 MT5 账户核实后关闭。
  注：2026-09-26 的全量 build 里，18 个文件全部来自 Dukascopy（`_DUKAREAL` 或原生导出），
  没有 `_MT5REAL` / `_with_volume` 文件参与合并，所以当前数据集不受此 Trap 影响；
  这条只在未来真的跑 `mt5_export_direct.py` / `mt5_fill_volume.py` 补数据时才会触发。

### 永久铁律（继承自指南，尚未在本项目触发，先记录用于警觉）

- **SSoT 原则**：全系统只允许一个"K 线权威源"函数入口——本项目是
  `lib/goldq/datastore.py::fetch_bars()`。任何新脚本一律通过它读数据，
  不允许在 02/03/04 目录下直接 `pd.read_csv` 原始文件。
- **localhost IPv6 陷阱**：如果未来搭建本地采集器 HTTP API，一律用 `127.0.0.1`，不用 `localhost`。
- **Bot 修改前必须 git 快照**：任何 `.py` 修改前 `git status` 确认干净 + 视情况打 tag/开分支。
- **部署审计**：Bot/采集器重启后必须核对 `ps -p <PID> -o lstart=` 晚于最新 commit 时间，
  不能只看 commit 时间就假设线上跑的是新代码。
- **配置分离**：非敏感配置进 `07_配置参数/*.yaml`（跟踪 git），敏感凭证只进 `.env`（不跟踪 git）。
