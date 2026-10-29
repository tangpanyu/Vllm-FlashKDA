# 文档导航

本仓库集中保存 vLLM FlashKDA 的源码、调优笔记、早期实验和本地 NCU 报告。迁移时工作树为 `experiment/vsplit-final@0c8a2d0`，并有 `fwd_kernel2.cuh`、`fwd_launch.cu` 的未提交修改；这些修改原样保留。

下文“当前分支优化记录”和实验复盘中的“当前”指它们写作时的 `feature/k2-vsplit@3fcf41b`，不是迁移时的实验工作树。阅读代码行为时请先核对文档标明的 commit。

| 文档 | 用途 | 什么时候看 |
|---|---|---|
| [FlashKDA v1: A Deep Dive](20260420-flashkda-v1-deep-dive.md) | 原始 FlashKDA v1 的公开设计说明，介绍 chunk size、两段 kernel、数值精度和基础优化 | 第一次了解 FlashKDA，或需要理解后续优化的基线时 |
| [当前分支相对 FlashKDA 的优化记录](current-branch-vs-flashkda-optimizations.md) | `feature/k2-vsplit@3fcf41b` 的实现事实、代码位置、数据布局约束和维护清单 | 理解 plain V-split 分支的基线，或准备修改 K1/K2 时 |
| [K2 V-split、Cluster 与寄存器实验复盘](k2-vsplit-experiment-retrospective.md) | 两代 V-split 性能分析、cluster/5-warp/state transpose 等正负实验，以及未合并的后续结果 | 追溯为什么选择当前方案，或继续做性能实验时 |
| [H20 benchmark](../BENCHMARK_H20.md) / [GB200 benchmark](../BENCHMARK_GB200.md) | 固定环境生成的公开性能快照 | 查看 FlashKDA 相对 FLA backend 的基准结果时 |
| [vLLM V-split / Cluster](vsplit-cluster.md) | resident-state 基线的实验设计和运行入口 | 维护当前实验代码时 |
| [NCU 热点分析](ncu-vllm-flashkda-hotspot-analysis.md) | H100 报告与后续优化候选 | 定位现有实验的性能问题时 |
| [NCU 使用指南](ncu-kernel-profiling-guide.md) / [报告索引](../ncu-reports/INDEX.md) | 采集、指标解释、23 份本地报告和文件校验 | 采集或复查 profiling 证据时 |
| [KDA 算法简化版](<KDA Kernel 简化版.md>) / [FlashKDA Kernel 讲解](<FlashKDA Kernel 讲解.md>) | 算法推导、旧版 kernel 主线和异步流水图 | 从算法开始复习时 |
| [旧版代码走读](kernel_walkthrough.md) / [旧版 Cluster Multicast](k2-vsplit-cluster-multicast.md) | 对应 `experiments/flashkda-v1`，不对应根目录新版 kernel | 追溯早期实现时 |
| [早期优化计划](FlashKDA优化计划.md) / [CP 笔记](CP之中.md) | 保留原始设想和背景 | 查阅历史思路时 |
| [原始 benchmark 输出](archive/benchmark-session-notes.md) | 旧目录 `pr.md` 中的 H100/B300 结果 | 核对实验复盘中的原始测量值时 |
| [实验资产](../experiments/README.md) / [迁移与打包](workspace-portability.md) | 源码快照、历史恢复、打包范围和来源清单 | 搬机器或继续实验时 |

## 阅读顺序

只想维护当前分支：直接读“当前分支相对 FlashKDA 的优化记录”。

想完整理解演进过程：

1. 先读 v1 Deep Dive，理解原始两 kernel 设计；
2. 再读当前分支优化记录，理解已经合入的变化；
3. 最后按需查实验复盘。实验复盘中的 cluster、5-warp、bankfix 和 state transpose 改写都不是当前 `feature/k2-vsplit@3fcf41b` 的既有能力。

## 维护规则

- 当前代码行为、默认 dispatch 或数据布局发生变化时，更新“当前分支优化记录”的版本边界和对应章节。
- 新的性能尝试统一追加到实验复盘，并写清代码基线、GPU、正确性检查和证据等级。
- 已被实验复盘吸收的单次实验报告不再单独保留，避免结论随时间互相冲突。
