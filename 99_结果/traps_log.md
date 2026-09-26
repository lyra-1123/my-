# Trap 沉淀日志

按指南第 8 章"复盘双轨制"维护：这里是 md 记录，重复出现 3 次以上的 Trap 应再沉淀为
`~/.claude/skills/` 或项目内 Skill（本仓库暂未建立，等实际踩坑后再建）。

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

### 永久铁律（继承自指南，尚未在本项目触发，先记录用于警觉）

- **SSoT 原则**：全系统只允许一个"K 线权威源"函数入口——本项目是
  `lib/goldq/datastore.py::fetch_bars()`。任何新脚本一律通过它读数据，
  不允许在 02/03/04 目录下直接 `pd.read_csv` 原始文件。
- **localhost IPv6 陷阱**：如果未来搭建本地采集器 HTTP API，一律用 `127.0.0.1`，不用 `localhost`。
- **Bot 修改前必须 git 快照**：任何 `.py` 修改前 `git status` 确认干净 + 视情况打 tag/开分支。
- **部署审计**：Bot/采集器重启后必须核对 `ps -p <PID> -o lstart=` 晚于最新 commit 时间，
  不能只看 commit 时间就假设线上跑的是新代码。
- **配置分离**：非敏感配置进 `07_配置参数/*.yaml`（跟踪 git），敏感凭证只进 `.env`（不跟踪 git）。
