# 这份代码包怎么用

这是论文配套的私有仓库版本 v0.1.0，整理日期为 2026 年 10 月 25 日。

仓库：https://github.com/hanabi2233/nightroad-dark-structure

权重附件：https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0

目前仅获授权的仓库用户能够访问，不代表已经向公众开放，也不代表论文已经录用。

- `src/frozen/`：与封存记录一致的原始科学代码，不要直接改它来适配路径。
- `tools/`：为发布整理的新入口；先阅读 `docs/REPRODUCING.md`。
- `metadata/`：论文附件2已有的划分、参数、日程和哈希记录。
- `results/`：已有实验的数值结果和论文图表对应关系。
- 模型文件在 Release v0.1.0 的单独附件 `SIVP_MODEL_ASSETS.zip` 中。

先运行 `python -B tools/verify_release.py`，只检查文件完整性，不会开始训练。

本轮整理没有重训、没有运行数据集推理、没有重新计算科学结果。因此不能把“包内检查通过”当作“独立复现了全部实验”。新入口与原始科研代码的区别写在 `docs/PROVENANCE.md`。

作者信息和自有代码/权重的使用许可仍需作者确认。第三方数据仍从原提供者获取。
