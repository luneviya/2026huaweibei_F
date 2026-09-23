# 数据目录说明

原始赛题数据不进入 Git 仓库。请设置环境变量 `HUAWEI_F_DATA_ROOT`，使其指向本机的 `real_attachments` 目录。

程序预期至少存在：

```text
real_attachments/
└── A_data_value/
    ├── domain_mapping_guide.csv
    ├── regmix_domain_summary.csv
    ├── slimpajama_quality_signal_sample.jsonl.xz
    └── regmix_tables/
        ├── train_mixture_1m.csv
        ├── train_pile_loss_1m.csv
        ├── test_mixture_1m.csv
        └── test_pile_loss_1m.csv
```
