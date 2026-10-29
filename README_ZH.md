# 这份代码包怎么用

这是论文配套的公开仓库。v0.1.0 整理于 2026 年 10 月 25 日，仓库及 Release 附件于 2026 年 10 月 29 日向公众开放。

仓库：https://github.com/hanabi2233/nightroad-dark-structure

权重附件：https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0

仓库及上述权重附件无需仓库邀请即可访问。

- 作者（按署名顺序）：Jie Xu（徐洁）、Lanxiang Wei（魏蓝翔）、Zhifeng Wang（汪志锋）。
- 单位：School of Intelligent Manufacturing and Control Engineering, Shanghai Polytechnic University, Shanghai, China。
- 通讯作者：Lanxiang Wei（魏蓝翔），[20251513219@sspu.edu.cn](mailto:20251513219@sspu.edu.cn)。
- 论文状态：已于 2026 年 10 月 28 日投稿至 Signal, Image and Video Processing，尚未录用或发表。

v0.1.0 标签及下载归档保留 10 月 25 日的原始快照；当前访问方式和作者信息以本 README 及更新后的 Release 说明为准，历史记录见 `docs/PROVENANCE.md`。

- `src/frozen/`：与封存记录一致的原始科学代码，不要直接改它来适配路径。
- `tools/`：为发布整理的新入口；先阅读 `docs/REPRODUCING.md`。
- `metadata/`：论文附件2已有的划分、参数、日程和哈希记录。
- `results/`：已有实验的数值结果和论文图表对应关系。
- 模型文件在 Release v0.1.0 的单独附件 `SIVP_MODEL_ASSETS.zip` 中。

先运行 `python -B tools/verify_release.py`，只检查文件完整性，不会开始训练。

本轮整理没有重训、没有运行数据集推理、没有重新计算科学结果。因此不能把“包内检查通过”当作“独立复现了全部实验”。新入口与原始科研代码的区别写在 `docs/PROVENANCE.md`。

第三方组件保留各自原有许可证；作者原创部分及受控模型权重目前未另行指定使用许可证。如需使用许可，请联系通讯作者。第三方数据仍从原提供者按其条款获取。论文发表引用及 DOI 尚未生成。
