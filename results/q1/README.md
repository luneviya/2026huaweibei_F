# 问题一正式实验结果

本目录保存2026-09-23运行的四方案正式对比结果。唯一冠军为S3：稳健排序聚合质量评价与多输出XGBoost。

核心文件：

- `scheme_comparison.csv`：四方案统一指标对比与一倍标准误差判定；
- `winner_pairwise_cv_tests.csv`：冠军与其余方案的同折配对检验；
- `q_observed_domain_scores.csv`：7个质量域的Q；
- `q_mapped_17_domain_scores.csv`：映射到17个训练域的Q；
- `winner_holdout_per_task.csv`：A6/A7上13个Loss任务的独立验证；
- `winner_cross_scale_rank_transfer.csv`：60M与1B排序迁移；
- `winner.json`：冠军参数和冻结字段；
- `cv_fold_metrics.csv`、`inner_tuning_results.csv`：逐折结果与内层调参记录。
- `environment_versions.json`：本次正式运行的解释器与依赖版本。

大体积可复用模型文件保存在本地 `outputs/q1_benchmark_20260923/`，可通过 `scripts/run_q1_full_experiment.py`重新生成，避免将环境相关的二进制对象写入Git历史。
