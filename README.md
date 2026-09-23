# 2026 华为杯 F 题实验仓库

本仓库用于统一测试问题一的四套候选方案，并在选出唯一优胜方案后继续完成问题二至问题四。原始赛题数据不提交到 Git；代码通过环境变量 `HUAWEI_F_DATA_ROOT` 读取本机的 `real_attachments` 目录。

## 问题一候选方案

| 编号 | 数据质量 Q | 配比到 Loss 的模型 |
|---|---|---|
| S1 | CRITIC 稳健综合评价 | 稀疏二阶 Scheffé 混料回归 |
| S2 | 稳健潜变量评价 | ILR + 多任务 Elastic Net |
| S3 | 稳健排序聚合 | 非线性 XGBoost |
| S4 | 贝叶斯层次质量模型 | 贝叶斯混料回归 |

最终只选择一套方案。选择规则在运行实验前固定，详见 [`docs/q1_selection_protocol.md`](docs/q1_selection_protocol.md)。

## 目录结构

```text
configs/                 实验配置与固定随机种子
docs/                    建模、评选和封版协议
scripts/                 数据检查与实验入口
src/huawei_f/            可复用源码
tests/                   自动化测试
outputs/                 本地运行结果，不提交大文件
reports/figures/         论文图片输出位置
reports/tables/          论文表格输出位置
```

## 本地准备

支持Python 3.9—3.13。若需严格复现实次正式结果，请使用Python 3.9.13与`requirements-lock-experiment.txt`；日常开发可使用Python 3.12。Windows PowerShell示例：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev]"
$env:HUAWEI_F_DATA_ROOT = "D:\path\to\F题\real_attachments"
```

检查四个方案的注册信息和数据契约：

```powershell
python scripts/run_q1_benchmark.py --list-schemes
python scripts/run_q1_benchmark.py --validate-data
pytest
```

运行四方案嵌套交叉验证并选择唯一冠军：

```powershell
python scripts/run_q1_full_experiment.py
```

完整运行依次生成质量评分稳定性、嵌套交叉验证、唯一冠军、A6/A7独立验证、60M/1B排序迁移以及可复用的冻结模型文件。

本次正式实验已经选定并冻结S3，参数见 [`configs/q1_winner.toml`](configs/q1_winner.toml)，选择证据见 [`docs/q1_winner_decision.md`](docs/q1_winner_decision.md)，完整轻量结果见 [`results/q1/`](results/q1/)。

## 实验原则

- 四个方案共用相同数据划分、随机种子和评价指标。
- A4/A5 用于训练和交叉验证；A6/A7 仅用于最终独立验证。
- A8-A11 用于跨规模迁移检验；A12-A15 只用于外推稳健性分析。
- 先按交叉验证结果选择唯一冠军，再冻结问题一的指标体系。
- 后续问题允许估计新加入的规模律参数，但不随意修改问题一的质量定义、领域映射和配比表示。
