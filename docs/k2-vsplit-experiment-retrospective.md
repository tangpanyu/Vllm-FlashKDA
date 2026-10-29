# FlashKDA K2 V-split、Cluster 与寄存器实验复盘

> 迁移标注（2026-10-29）：正文的“当前分支”沿用写作时的 `feature/k2-vsplit@3fcf41b`。本次汇总没有改变实验结论或将旧版代码覆盖到根目录；迁移后的源码、报告和原始输出入口见 [实验资产](../experiments/README.md)。

> 文档定位：这是实验历史与负结果的统一归档，不是当前代码说明。当前 `feature/k2-vsplit@3fcf41b` 的事实边界见 [当前分支相对 FlashKDA 的优化记录](current-branch-vs-flashkda-optimizations.md)，文档总览见 [文档导航](README.md)。

这份文档记录两轮结论，必须按时间理解：

1. **进入当前分支的收敛结果**：保留 6-warp plain V-split，后在 `3fcf41b` 改为保守的自动 dispatch；第一轮 cluster、5-warp fused-IO 和 state transpose 改写没有合入。
2. **未合并的后续实验**：per-warp cluster release 消除了旧 cluster 的 CTA-wide barrier 热点，使 C5/C6 在部分低并行度 workload 上超过 plain split；但 5/6-warp 同时使用不同 register ceiling，尚不能作为纯 warp 数消融，也没有进入当前分支。

本文重点记录 6-warp V-split/cluster、5-warp V-split/cluster、state 寄存器削减、RTX 5060 Ti 上的 MIO stall、H100 上的 L2 tag/request 瓶颈，以及为什么提高理论 occupancy 后仍不保证产生更多 eligible warp。第 13 节收录后续最终对比，第 14 节给出可直接复述的版本。

> **版本边界：本文跨越两代 kernel。** 第 4～5 节的 RTX 5060 Ti MIO 和 H100 L2 tag/request 结论来自较早的 **FlashKDA 基线**；第 6～12 节的第一轮 cluster、5-warp、state 寄存器改写和当前分支选择，来自已经集成 **Flash-Flash-KDA** 优化的 `vllm-project/FlashKDA` 基线；第 13 节又来自更晚的本地实验分支。各代代码的 TMA 数量、state 生命周期、stage、寄存器和 shared-memory footprint 均不同，不能把前一代的 NCU 数字直接当成后一代 kernel 的瓶颈数据。

| 实验代际 | 代码基线 | 对应实验 | 结论用途 |
|---|---|---|---|
| 第一代 | 较早的 FlashKDA V-split | RTX 5060 Ti MIO、H100 Long Scoreboard/L2 tag | 解释原始 V-split 为什么同时出现低计算与低 DRAM 吞吐 |
| 第二代 | `vllm-project/FlashKDA dev@b5d1101`，已包含 Flash-Flash-KDA 集成 | 6-warp plain/cluster、5-warp plain/cluster、2-stage、state 寄存器消融、per-warp release、bankfix | 选择当前分支方案，并记录后续未合并 variant 的性能边界 |

## 1. 一页结论

1. V-split 的核心价值是增加 K2 的 CTA 数，解决 TP 后 local head 少、长 prefill CTA 不足的问题；它不是提高单 CTA 算术强度的优化。
2. 当 $N\times H$ 较小时，plain V-split 在 H100/B300 的代表 workload 上可提升约 20%～36%；当原始 K2 已经有足够 CTA 时，V-split 会因计算宽度减半、公共输入重复和额外调度成本而回退。
3. Cluster multicast 确实减少了公共 workspace 的 TMA/L2/DRAM 流量。第一轮每 chunk 的 128-thread 同步抵消了收益；后续 per-warp release 删除该热点后，cluster 在部分低并行度 case 超过 plain split，但成对调度和较低的单 CTA 计算密度仍限制适用范围。
4. 第一轮 5-warp 实验将独立 load/store warp 合并为 fused-IO warp，理论 CTA 上限可从 2 提高到 3，但 active warp 没有自动变成 eligible warp。后续 per-warp release 实验中 C6 在低并行度 H12 更快、C5 在部分高-head shape 更稳；由于二者 register ceiling 不同，这只能比较完整实现，不能归因于 warp 数本身。
5. 当前 kernel 并没有长期保存两份 state。常驻的是 C-layout `resident_state`，B-layout 只是 Phase 1 当前 block 的临时 fragment。强行削减临时寄存器会减少编译器的跨 iteration register renaming 和 ILP，性能反而下降。
6. 在第一代 FlashKDA V-split 上，RTX 5060 Ti 与 H100 都曾出现“计算吞吐和 DRAM 吞吐同时不高”，但原因不同：5060 Ti 主要表现为低 TLP、MIO/依赖导致 scheduler 饥饿；H100 的代表报告则主要卡在 TMA transaction barrier 和 L2 tag/request 路径，而不是 HBM 字节带宽打满。这一结论用于解释早期 kernel，不直接归因于后来的 Flash-Flash-KDA 基线。

当前分支的工程决策：保留实现简单、收益边界清楚的 6-warp plain V-split，并只在 `2 * H * N <= SM count` 时自动启用。Cluster、5-warp、bankfix 和 state transpose 仍只是实验资产；后续结果见第 13 节。

## 2. Kernel 与实验变量

K2 的基础问题规模为：

```text
CHUNK = 16
D     = 128
```

原始 K2 的 state/output V 维为 128。V-split 取 `VD=64`，把同一个 `(sequence, head)` 拆成两个 CTA：

```text
CTA v_idx=0 -> V[0:64]
CTA v_idx=1 -> V[64:128]
```

两条 CTA 的 `V`、state 和 `O` 不同，但以下 workspace 数据与 V-slice 无关：

```text
k_decayed, q_decayed, k_restored, g_total, INV, Mqk
```

实验过的五条路径如下。

| 路径 | V tile | CTA 数 | Warp 组成 | 公共 workspace |
|---|---:|---:|---|---|
| original | 128 | $N\times H$ | 4 MMA + load + store | 每 CTA 加载一次 |
| plain V-split 6-warp | 64 | $2N\times H$ | 4 MMA + load + store | 两个 slice 各加载一次 |
| cluster V-split 6-warp | 64 | $2N\times H$ | 4 MMA + load + store | rank 0 multicast 给两个 CTA |
| cluster V-split 5-warp | 64 | $2N\times H$ | 4 MMA + fused load/store | rank 0 multicast给两个 CTA |
| plain V-split 5-warp | 64 | $2N\times H$ | 4 MMA + fused load/store | 两个 slice各加载一次 |

plain V-split 的收益和代价可以粗略写成：

$$
\text{收益}=\text{更多 CTA}+\text{更小的 state/shared footprint}
$$

$$
\text{代价}=\text{公共输入重复}+\text{每 CTA 的 MMA N 维减半}+\text{更多控制工作}
$$

cluster 只试图消掉“公共输入重复”中的 global/L2 部分，它不会恢复原始 CTA 内 $N=128$ 的数据复用，也不会消除两个 V-slice 各自的 MMA、state、V、O 和 pipeline 控制。

## 3. 实验方法与证据边界

VLLM 版本的正式 benchmark 使用：

- H100 80GB HBM3，132 SM；B300 SXM6 AC，148 SM；
- FP32 initial/final state；
- 每条 variant 单独构建 CUDA Graph；
- 30 次 warmup、200 次 iteration、6 次交错 repeat；
- 每轮计时前读取 256 MiB，构造 cold-L2 条件；
- speedup 定义为 `original latency / variant latency`；
- output 与 final state 做 bitwise exact 检查。

命令为：

```bash
modal run modal_vllm_flashkda.py --action bench \
  --mode cases \
  --warmup 30 \
  --iterations 200 \
  --repeats 6 \
  --l2-scrub-mib 256
```

文中的 NCU 报告来自不同阶段的二进制。必须区分以下两组：

1. `FlashKDA_h100_k2_full_compare.ncu-rep` 是较早的 FlashKDA 版本，V-split 为 2 input stages，state 尚未采用后来的完整寄存器常驻实现。
2. `Vllm-FlashKDA_*.ncu-rep` 基于 vLLM/Flash-Flash-KDA 优化后的版本，3 input stages，包含 raw workspace bulk copy 和 resident state。

因此，本文只在同一份报告或同一次 benchmark 内做直接数值比较。跨报告的数据用于解释机制，不用于声称某一项改动单独贡献了多少加速。

### 3.1 证据审计（2026-08-07）

本文中的数字按可复核程度分为三档：

| 等级 | 数据 | 当前证据状态 |
|---|---|---|
| A：可直接重导 | RTX 5060 Ti K2、第一代 H100 original/V-split、第二代 H100 6-warp/5-warp，以及最终 C5/S5 NCU 指标 | 第 15 节列出的关键 `.ncu-rep` 均仍存在，并已重新导入核对；最终 C5/S5 报告由 NCU 2026.1.1 生成 |
| B：有完整 benchmark 文本 | H100/B300 的 original、plain split、6-warp cluster 延迟与 correctness | 原始终端结果保存在 [benchmark-session-notes.md](archive/benchmark-session-notes.md)；不是从 NCU duration 推算 |
| C：历史实验记录 | 2-stage、5-warp 完整 12-case/大 batch、state 强依赖消融 | 实验分支、表格及部分 Modal run ID 仍在，但没有为每次运行保留独立 `.ncu-rep` 或完整本地日志；精确延迟应视为当时运行快照 |

机制结论主要由 A/B 级证据支撑。C 级数据适合说明“试过什么、结果方向如何”，如果以后要在论文、正式报告或新的 PR 中引用精确百分比，应从相应实验分支重新构建并复跑。

## 4. 第一代 FlashKDA / RTX 5060 Ti：MIO stall 与 scheduler 饥饿

### 4.1 观察

SM120 上被测 V-split K2 为：

```text
CHUNK=16, D=128, VD=64
InputStages=3, OutputStages=2
Threads=192
Grid=(2,1,32)，共 64 CTA
```

代表性 NCU 指标：

| 指标 | 数值 |
|---|---:|
| Tensor pipeline utilization | 22.23% |
| Memory throughput | 132.78 GB/s |
| Mem Busy | 35.82% |
| Mem Pipes Busy | 8.69% |
| Issue Slots Busy | 7.40% |
| Executed IPC elapsed | 0.29 |
| No Eligible | 91.55% |
| Active Warps / Scheduler | 1.51 |
| Eligible Warps / Scheduler | 0.09 |
| Registers / Thread | 58，硬件按 64 分配 |
| Shared Memory / CTA | 69.632 KB，含 driver allocation |
| Shared-memory CTA limit | 1 CTA/SM |

现存报告的 Warp State raw ratio 及其在全部 state ratio 中的归一化占比为：

| Stall | cycles / issued instruction | 归一化占比 |
|---|---:|---:|
| sleeping | 8.949 | 50.1% |
| MIO throttle | 2.986 | 16.7% |
| fixed-latency wait | 1.827 | 10.2% |
| long scoreboard | 0.911 | 5.1% |
| math-pipe throttle | 0.839 | 4.7% |
| short scoreboard | 0.801 | 4.5% |
| barrier | 0.104 | 0.6% |

表中未列出 `selected` 和占比较小的其他 state，所以占比之和不为 100%。`sleeping` 很高不等于一半时间都可优化，因为 load/store 专用 warp 的正常等待也被统计进去。真正重要的是 `No Eligible=91.55%`：scheduler 大多数周期找不到一条可以发射的 warp。

### 4.2 为什么计算和 DRAM 都低

这是 latency/issue starvation，而不是某条峰值管线打满：

$$
\text{1 CTA/SM 的低 TLP}
+\text{紧密的 shared/MMA 依赖}
+\text{MIO queue 压力}
\Longrightarrow
\text{No Eligible 很高}
$$

Phase 6 的少量 MMA 周围有 `LDSM`、`STSM`、FP32 FMA、BF16 pack 和寄存器搬运。warp 发出 shared 指令后很快消费结果，独立指令不够；另一 CTA 又无法驻留，最终 Tensor Core 和 DRAM 都吃不满。

### 4.3 MIO 不等于 bank conflict

`mio_throttle` 表示 MIO 指令队列不能接受更多请求。它可能来自 shared load/store、matrix shared 指令、特殊函数或相关控制流，不能单独证明 bank conflict。

现存报告中逐条 `LDSM/STSM` 的 `Wavefronts Excessive` 为 0，但 raw shared bank-conflict counter 非零：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum=196,622`，其中 store 为 195,700、load 为 223。总数不等于 load/store 两项简单相加，说明 counter 口径不能混用；这里仅把它视作 conflict 信号。两类观察不矛盾：

- `Excessive = actual wavefronts - ideal wavefronts`，检查相对理想访问是否产生额外 wavefront；
- bank-conflict counter 检查同一 shared request 内的 bank 序列化。

`LDSM/STSM .4` 本来就需要多个 wavefront，所以不能用下面的值直接宣称“4-way layout conflict”：

$$
\text{Source N-Way}=\frac{\text{Wavefronts}}{\text{Instructions Executed}}
$$

结论是：存在 store-side conflict 计数，但没有证据证明它是 MIO stall 的唯一或主要来源。更强的证据仍是低 TLP、`No Eligible` 和紧密的 load-use 链。

### 4.4 为什么增加 prefetch 没用

Prefetch 改变发射时间，不减少指令或流量。例如 8 个 block：

```text
PREFETCH=1：初始 1 次 + 循环补载 7 次 = 8 次
PREFETCH=2：初始 2 次 + 循环补载 6 次 = 8 次
```

更早发 `LDSM` 会增加 fragment live range，也可能更早堵住 MIO queue。只有当 `short_scoreboard` 下降且 `mio_throttle` 不上升时，prefetch 才真的覆盖了 latency；本实验没有观察到这类收益。

## 5. 第一代 FlashKDA / H100：Long Scoreboard 背后的 L2 tag/request 瓶颈

### 5.1 不是 HBM bandwidth 打满

较早的 H100 V-split 报告配置为：

```text
CHUNK=16, D=128, VD=64
InputStages=2, OutputStages=2
Threads=192
Grid=(2,8,96)，共 1536 CTA
Registers / Thread Allocated=80
Shared Memory / CTA=53.632 KB
```

Warp State 指标：

| 指标 | original | V-split |
|---|---:|---:|
| Warp cycles / issued instruction | 约 8.1 | 约 17.1 |
| Long Scoreboard cycles / issue | 3.00 | 9.32 |
| Long Scoreboard 占 inter-issue 周期 | 37.1% | 54.5% |

Source/SASS 中，V-split 的 Long Scoreboard not-issued samples 共 71,420 个，其中 55,991 个，即 78.4%，落在：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT ...
@!P0 BRA ...
```

它对应 `load_pipeline.consumer_wait(load_read)`。标红的 `BRA` 不是根因；consumer 正在轮询 transaction barrier，等待当前 TMA input stage 完成。

### 5.2 顶层 L2 Throughput 为什么接近满载

H100 V-split 的 L2/DRAM 子指标：

| 子指标 | `% of peak sustained elapsed` | 含义 |
|---|---:|---|
| `lts__t_tag_requests` | 93.58% | L2 tag/request 查询路径 |
| `lts__xbar2lts_cycles_active` | 64.63% | XBAR 到 L2 的输入路径 |
| `lts__t_sectors` | 33.47% | tag 侧 requested sectors |
| `lts__d_sectors` | 36.13% | L2 data 侧 sectors |
| `lts__lts2xbar_cycles_active` | 35.33% | L2 返回 SM/XBAR 的路径 |
| `lts__d_sectors_fill_device` | 26.06% | HBM miss fill sectors |
| `gpu__dram_throughput` | 40.64% | HBM 字节吞吐 |

GUI 顶层 `L2 Throughput` 接近 94%，是由 tag/request 子路径决定的，不代表 L2 data bandwidth 达到 94%。hit 和 miss 都要做 tag lookup，但只有 miss 才产生 HBM fill，所以 L2 tag 高、DRAM 只有 40.6% 完全可以同时成立。

报告中的请求粒度也很碎：

```text
L2 requests = 215.976 M
L2 sectors  = 231.725 M
```

$$
\text{Sectors/Request}=\frac{231.725}{215.976}\approx1.073
$$

这不能单独证明 TMA 没有合并，但说明 request 数相对每个 request 携带的 sector 很高。

### 5.3 为什么 occupancy 增加仍在等 Long Scoreboard

| Scheduler 指标 | original | V-split |
|---|---:|---:|
| Active Warps / Scheduler | 2.96 | 5.71 |
| Eligible Warps / Scheduler | 0.488 | 0.511 |
| Issue Active | 36.60% | 33.34% |

V-split 让 active warp 几乎翻倍，但 eligible warp 基本不变。这是整组实验中最重要的一条：

$$
\text{resident/active warp 增加}\not\Rightarrow\text{ready/eligible warp 增加}
$$

多个 consumer warp 可能一起等待相同类型的 TMA stage，新增 CTA 还会增加 L2 request/tag 压力。该报告中：

| 指标 | original | V-split | 变化 |
|---|---:|---:|---:|
| L2 requests | 147.9 M | 216.0 M | +46.0% |
| L2 sectors | 161.8 M | 231.7 M | +43.2% |
| NCU kernel duration | 1.167 ms | 1.687 ms | +44.6% |

因此这份报告支持“新增 request/tag 工作使 TMA barrier latency 暴露”，不支持“HBM bandwidth 已满”。

## 6. 第二代 Flash-Flash-KDA 基线：第一轮 6-warp cluster 实验

### 6.1 设计

6-warp cluster 保留四个 MMA warp、独立 load warp 和独立 store warp。两个 V-slice CTA 组成 `clusterDim=(2,1,1)`，rank 0 通过：

```text
cp.async.bulk.shared::cluster.global
    .mbarrier::complete_tx::bytes.multicast::cluster
```

把六块公共 workspace 的原始 swizzle 字节图 multicast 到两个 CTA 的 shared memory。`V`、beta、state 和 `O` 仍由每个 CTA 自己处理。

正确同步至少需要：

1. 第一笔 multicast 前 cluster sync，保证两个 CTA 的 transaction barrier 已初始化；
2. FP32 state scratch 与 input stage 共用 union 时，state 转换完成后再 cluster sync；
3. 每 chunk 在 `consumer_release()` 前做 128-thread `NamedBarrier`，确保四个 MMA warp 都不再读取该 stage；
4. kernel 退出前 cluster sync，防止 peer 仍访问 DSM/barrier 时某个 CTA 先退出。

这些同步不能为了性能直接删除。缺失时可能是数值竞态、illegal access 或偶发 hang；若两个 CTA 的 barrier phase/arrival 次数不对称，则会形成确定性死锁。

### 6.2 H100 benchmark

VLLM 优化基线上的代表结果，单位为微秒：

| Case | original | plain split 6-warp | cluster 6-warp | plain speedup | cluster speedup |
|---|---:|---:|---:|---:|---:|
| `pure_n1_t8192_h12` | 645.02 | 474.53 | 490.75 | 1.359x | 1.314x |
| `pure_n4_t2048_h12` | 225.28 | 177.86 | 182.30 | 1.267x | 1.236x |
| `pure_n1_t8192_h96` | 1036.77 | 1012.96 | 1050.85 | 1.024x | 0.987x |
| `pure_n4_t2048_h96` | 972.26 | 912.16 | 946.56 | 1.066x | 1.027x |
| `mixed_d31_p8161_h12` | 669.28 | 499.42 | 513.28 | 1.340x | 1.304x |
| `mixed_d31_p8161_h96` | 1417.09 | 1188.77 | 1247.52 | 1.192x | 1.136x |

12 个 workload 全部 bitwise exact，但 cluster 相对 plain split 通常慢约 2.5%～5%。B300 也呈现相同方向：低 head 时 cluster 只能大致追平 plain split，$H=96$ 时通常再慢约 4%～6%。

### 6.3 为什么减少了 TMA 仍不快

Cluster 的优化目标确实实现了：在后来的 H100 NCU 中，`pure_n1_t8192_h96` 的 plain split 与 5-warp cluster 对比为：

| 指标 | plain split | cluster |
|---|---:|---:|
| DRAM throughput | 74.05% | 53.51% |
| L2 throughput | 79.99% | 62.15% |

公共请求减少了，但 cluster 仍更慢。这说明 TMA/L2 流量不是唯一成本。原因包括：

- vLLM 基线已经采用 raw bulk workspace load、resident state 和较少的 TMA，multicast 的边际收益比旧 FlashKDA 小；
- cluster 每 chunk 多一个 128-thread barrier，并有启动/退出 sync；
- 两个 CTA 成对调度，降低了 placement 和尾部调度自由度；
- multicast 只减少 global/L2 请求，不恢复原始 $N=128$ CTA 的计算密度；
- 两个 slice 仍分别执行 state、V/O 和 pipeline 控制。

旧 FlashKDA 版本上，cluster 在 CTA 已足够多时曾明显追回 plain split 的损失，例如 H100 `B=4,T=2048,H=96`：original 为 1.417 ms、plain split 为 1.674 ms、cluster 为 1.435 ms。它说明 multicast 的机制是有效的，但即使在那里，cluster 也只是接近 original，并未形成稳定新最优；换到已经减少大量 TMA 的 vLLM 基线后，收益进一步缩小。

## 7. 第二代 Flash-Flash-KDA 基线：第一轮 5-warp cluster 实验

### 7.1 设计与同步陷阱

5-warp cluster 保留四个 MMA warp，把 load 和 store 控制合并到第五个 warp。实际发 TMA 的仍只有 elected lane；少一个 warp 的实验目的不是增加搬运线程，而是减少 CTA thread 数并尝试提高 residency。这里的 160/192 是线程数，不是寄存器数；后续实验还人为修改过 whole-kernel `__maxnreg__`，因此不能把观察到的资源或性能变化全部归因于少一个 warp。

fused-IO 循环为：

```text
循环外：预取 min(t_tiles, InputStages) 个 input stage

循环内：
1. 等待 O[t] ready
2. 发射 input[t + InputStages]
3. 发射 O[t] 的 TMA store
4. 等待/释放 output stage
```

`load_pipeline.producer_tail()` 必须在 output ring 全部排空后执行。若 fused warp 先在 producer tail 等 input empty barrier，而 MMA warp 又因 output ring 满等待同一个 fused warp 写回，就会形成循环等待。这是早期实现可能卡死的原因之一。

### 7.2 资源与 NCU

对齐的 H100 NCU，case 为 `pure_n1_t8192_h96`：

| 指标 | cluster 6-warp | cluster 5-warp |
|---|---:|---:|
| Block size | 192 | 160 |
| Allocated registers / thread | 128 | 120 |
| Shared memory / CTA | 69.632 KB | 69.632 KB |
| Shared-memory config | 167.936 KB | 233.472 KB |
| Register CTA limit | 2 | 3 |
| Shared-memory CTA limit | 2 | 3 |
| Spill | 0 | 0 |
| NCU duration | 621.952 us | 617.632 us |
| Active warps / SM | 9.20 | 7.67 |
| Eligible warps / scheduler | 0.533 | 0.522 |
| Issue active | 40.67% | 39.82% |
| Tensor throughput | 20.87% | 20.86% |
| DRAM throughput | 53.15% | 53.51% |
| L2 throughput | 61.70% | 62.15% |
| Barrier stall / issue active | 0.149 | 0.241 |

表中的 128/120 是 `launch__registers_per_thread_allocated`。对应的编译器/NCU `Registers Per Thread` 分别是 122/120；前者按硬件分配粒度向上取整，二者不应混为同一个寄存器计数。

还要注意，这组历史 NCU 不是严格的 register-policy 对照。后续 5/6-warp 集成代码曾分别使用 128 和 112 的显式 `__maxnreg__` ceiling，以便在 H100 的寄存器账面上容纳 3 CTA。`__maxnreg__` 不代表 LOAD/STORE warp 天然需要这些寄存器，也不会按 warp role 自动分配；它会同时约束四个 MMA warp。因此本节表格只能还原当时两个完整构建的实际资源，不能证明“192-thread CTA 天然需要更多寄存器”或“160-thread CTA 天然具有寄存器优势”。

这组数据纠正了一个早期判断：不能只用旧 6-warp 的 167.936 KB carveout 推断 5-warp 仍只能驻留 2 CTA。新报告实际选择了 233.472 KB shared 配置，理论 CTA 上限确实是 3。

但理论上限提高后，Tensor、issue、eligible warp 和 memory throughput 几乎不变，kernel 只快约 0.7%。这个 case 的 grid 只有 192 CTA，而 H100 有 132 SM，不能在全卡持续填满 3 CTA/SM；同时 cluster 成对调度，实际可利用并发还受 cluster placement 限制。

相对 6-warp cluster 的 12-case benchmark，5-warp cluster 的归一化平均改善约 2.8%，但收益高度不均匀：`pure_n2_t4096_h96` 改善约 22.8%，`pure_n4_t2048_h96` 改善约 6.1%，其余大部分 case 约为 -0.6%～+2.8%。它改善了 cluster 自身，却仍没有稳定超过 6-warp plain split。

### 7.3 2-stage 补充实验

还试过把 input pipeline 从 3 stage 降到 2 stage。5-warp cluster 的 12 个 workload 全部退化，归一化平均约 -2.5%；plain split 平均约 -6.2%，部分 specialization 还出现新 spill。

这不能简单解释为“stage 越少越差”。fused-IO warp 每轮发新 input 后还要等待当前 output store，input refill 与 output drain 形成新的串行边；3-stage 尚能缓冲，2-stage 更容易把这条边暴露到关键路径。

## 8. 第二代 Flash-Flash-KDA 基线：第一轮 5-warp plain split

为了区分“少一个 warp”和“cluster multicast”各自的影响，`experiment/split-5warp-ablation` 让 plain 与 cluster 都使用五个 warp。

编译资源：

| 版本 | Registers / thread | Barrier | Spill | Register 允许的 CTA/SM |
|---|---:|---:|---:|---:|
| 5-warp non-cluster | 114 | 1 | 0 | 3 |
| 5-warp cluster | 118 | 9 | 0 | 3 |

H100、固定 `T=2048,H=96` 的大 batch 结果：

| Case | original | 5-warp non-cluster | 5-warp cluster | cluster 对 non-cluster |
|---|---:|---:|---:|---:|
| `B=8` | 1753.71 us | 1830.75 us | 1780.64 us | cluster 快 2.8% |
| `B=16` | 3460.32 us | 3583.38 us | 3524.80 us | cluster 快 1.7% |
| `B=32` | 6913.33 us | 7152.37 us | 7180.58 us | non-cluster 快 0.4% |

全部 bitwise exact。相对先前 6-warp non-cluster，5-warp non-cluster 在 B8 基本持平、B16 约快 1%、B32 约快 1.7%。因此不能说 5-warp 在所有场景都更慢；它在足够大的 pure-prefill grid 上确实兑现了一点 residency 收益。

问题是收益不稳定。完整 H100 workload 中，5-warp plain split 在 mixed decode/prefill、`H=96` 的三组 case 上平均退化约 13.1%，前两组约退化 16.5%。FP32 state 的 shared-memory 转换会把 8×8 atom 分摊给 CTA 全部线程，线程从 192 减到 160 后，每个参与 warp 承担更多固定转换工作；短 decode CTA 对这部分固定成本尤其敏感。

因此 5-warp non-cluster 是“某些大 pure-prefill case 小幅更好”，不是可以替换 6-warp plain split 的稳健方案。

## 9. 第二代基线：为什么第一轮 5-warp ± cluster 没有稳定打过 6-warp plain V-split

这是整组实验最值得面试讲的部分。

### 9.1 提高的是 TLP，不是单 CTA 的 ILP

3 CTA/SM 增加的是 thread-level parallelism。它不会自动增加单个 warp 内可同时发射的独立指令，也不会改变 `MOVM -> MMA`、`LDSM -> MMA`、TMA barrier 等依赖。

如果三个 CTA 在相似阶段等待同一种资源，新增 active warp 仍不是 eligible warp：

$$
\text{更多 resident CTA}
+\text{相关的 memory/barrier wait}
\Longrightarrow
\text{active 增加，eligible 不增加}
$$

H100 早期报告里 active warp 几乎翻倍而 eligible 仅从 0.488 到 0.511，就是这一现象的直接证据。

### 9.2 异步 TMA 不代表搬运“没有成本”

TMA 指令由少数 lane 发射，数据搬运不占 CUDA core，但仍消耗：

- TMA request queue；
- L2 tag/request 和 data path；
- shared-memory 写入；
- mbarrier transaction；
- pipeline stage；
- producer/consumer 的控制指令和等待。

所以“load/store warp 只发指令”不等于它们之间可以任意合并。5-warp fused-IO 把 input refill 和 output drain 放进同一条控制流，形成 6-warp 版本不存在的串行边。

### 9.3 V-split 降低每 CTA 的计算密度

原始 CTA 对同一组 K/Q/workspace 做 $N=128$ 的输出；V-split 的每个 CTA 只做 $N=64$。plain split 还让两个 CTA 重复公共 workspace 请求。即使 cluster 消除这部分 global load，它也没有把两半计算重新合回一个更宽的 MMA tile。

当 $N\times H$ 已足够填满 SM，original 的大 tile 复用更有价值；split 多出来的 CTA 只增加 wave 数和控制成本。这就是 H100/B300 上 `H=12` 常获益，而 `H=96` 边界明显变差的原因。

### 9.4 Cluster 节省流量，但增加同步和 placement 成本

5-warp NCU 中 cluster 相对 plain split 把 DRAM/L2 throughput 从 74.05%/79.99% 降到 53.51%/62.15%，证明 multicast 做到了它声称的事；但 barrier stall 上升、issue/Tensor 吞吐没有提高，最终 latency 仍更差。

这说明优化不能只看“减少了多少字节”，而要看减少的资源是否真在关键路径上：

$$
\text{净收益}
=\text{tail/residency 收益}
+\text{公共流量收益}
-\text{计算密度损失}
-\text{同步/placement 成本}
-\text{新增依赖}
$$

### 9.5 5-warp 落在一个尴尬区间

- 小 grid：理论 3 CTA/SM 填不满，plain 6-warp 已足够；少一个 warp 没有兑现 occupancy。
- 大 grid：3 CTA/SM 能带来约 1%～2% 的 pure-prefill 收益，但 original 已有足够 CTA，并保留更好的 $N=128$ 计算密度。
- mixed workload：少一个 warp 增加 FP32 state 转换的每线程固定工作，5-warp 明显回退。
- cluster：节省公共流量，但 barrier 和成对调度抵消收益。

所以最终观察不是“5-warp 完全无效”，而是“适用区间太窄，不足以承担额外实现和 review 复杂度”。

## 10. 第二代基线：State 寄存器削减实验

### 10.1 先纠正假设

当前 VLLM K2 把 `[V,K]` state 以 MMA C-fragment 的形式跨 phase、跨 chunk 常驻：

```text
resident_state (C-layout)
  -> Phase 1: 当前 16x16 block MOVM_T 为临时 B fragment
  -> K@S 和 Q@S 消费临时 B fragment
  -> 临时 B fragment 失效
  -> Phase 6: C-layout resident_state 原地更新
```

因此不存在“完整 C-layout state + 完整转置 B-layout state”长期并存。每次只为当前 16×16 block 产生四个 32-bit 临时 B 寄存器。

### 10.2 尝试与结果

| 方案 | 资源变化 | 性能结果 | 结论 |
|---|---|---|---|
| 显式临时 B + `__syncwarp()` | 112 regs，无 spill | 快速筛查退化 | sync 不能命令 ptxas 结束 live range |
| resident state 原地 C→B→C | 112→110 regs | 退化 | MOVM 工作量翻倍 |
| 严格对齐旧 `fwd_kernel2_1.cuh` 的 Phase 1 顺序 | plain stack 8 B→0 B，112 regs 不变 | plain 平均 -2.7%，最大 -6.5%；cluster 平均 -2.5% | 少一对静态 LDL/STL 不代表关键路径更短 |
| 用真实 RAW 依赖禁止跨 block 提前转置 | 静态 live regs 107→101，stack 32 B→8 B | plain 平均 -7.6%，cluster 平均 -14.0% | register renaming 原本在提供 ILP |

完整 SASS 对比：

| Specialization | 实现 | Registers | Stack | `LDL` | `STL` | `MOVM` | `HMMA` |
|---|---|---:|---:|---:|---:|---:|---:|
| plain split | baseline | 112 | 8 B | 1 | 1 | 40 | 52 |
| plain split | 对齐旧 `2_1` | 112 | 0 B | 0 | 0 | 40 | 52 |
| cluster split | baseline | 112 | 8 B | 1 | 1 | 40 | 52 |
| cluster split | 对齐旧 `2_1` | 112 | 40 B | 18 | 11 | 40 | 52 |

baseline 和改写版都会把展开循环中不同 iteration 的临时 B 结果 rename 到不同物理寄存器。源码只有一个 fragment 名称，不代表 SASS 只能用一组物理寄存器。

强制顺序版形成：

```text
上一轮 HMMA -> LOP3 dependency -> MOVM -> 下一轮 HMMA
```

它确实减少了静态 live register，但也删除了编译器原本用于隐藏 latency 的 ready instruction。即使 kernel 总体表现为 memory latency bound，人为增加 compute dependency 也不是免费的；它会延长关键路径并降低覆盖同一 memory latency 的能力。

### 10.3 Spill 的正确解读

- `__maxnreg__` 是每线程寄存器上限，不是把整个 SM 的寄存器自动按 warp 重新分配。
- 减少一个控制 warp不会自动缩短 MMA warp 中 state、accumulator、pipeline state 和临时 fragment 的 live range。
- whole-kernel 出现一对 `LDL/STL`，不能在没有 source correlation 时直接认定为 state B-fragment spill。
- 评估寄存器改写必须同时看 registers/thread、stack frame、动态 local-memory 请求、SASS 调度和最终 latency；只看 ptxas 的数字容易反向优化。

最终保留原有 state C→临时 B 的转置及 MMA/prefetch 顺序。

## 11. 第一代 FlashKDA 的两份 NCU 为什么看起来不同

| 现象 | RTX 5060 Ti 报告 | H100 报告 |
|---|---|---|
| CTA residency | shared 限制 1 CTA/SM | V-split 可有更多 active warp |
| Scheduler | No Eligible 91.55%，eligible 0.09/scheduler | active 2.96→5.71，但 eligible 0.488→0.511 |
| 主要 stall 表象 | MIO throttle + fixed wait + 低 TLP | Long Scoreboard，主要停在 TMA barrier |
| Memory 子系统 | shared/MIO 密度高 | L2 tag requests 93.58%，DRAM 40.64% |
| 核心解释 | 没有足够独立 warp/指令覆盖 shared/MMA latency | 更多 warp在等相关的 TMA/L2 request 依赖 |

这一节比较的两份报告都属于第一代 FlashKDA V-split。共同点不是“都 bandwidth bound”，而是 scheduler 没有持续得到 ready instruction。区别在于 ready instruction 被哪条依赖链阻断。它们不能用来声称第二代 Flash-Flash-KDA kernel 仍由同一组指标主导；第二代各 variant 的判断以第 6～10 节的同基线 benchmark/NCU 为准。

面试时不要只说“memory-bound”。更准确的说法是：

> RTX 5060 Ti 的代表 kernel 是低 occupancy 下的 issue starvation，MIO/shared 依赖更突出；H100 的代表 kernel 有更多 resident warp，但这些 warp 的等待高度相关，TMA transaction barrier 和 L2 tag/request 路径成为暴露出来的瓶颈。两者都没有打满 HBM，因此属于 latency/issue 问题，不是峰值字节带宽问题。

## 12. 当前分支的收敛决策

保留：

- 6-warp plain V-split；
- `VD=64`，两个独立 CTA；
- 先以显式 opt-in 引入，后在 `3fcf41b` 改为保守自动 dispatch；
- 针对 TP 后 local heads 较少、长 prefill 主导的 shape 使用；
- 原有 resident state 和编译器调度。

不保留：

- 6-warp cluster 产品路径；
- 5-warp cluster；
- 5-warp plain split；
- 2-stage fused-IO；
- state C/B 原地变换、额外 sync 或强制寄存器依赖。

原因不是这些方案完全无效，而是当时的收益不稳定、适用区间窄，且增加了同步正确性、架构选择和 code review 成本。后续 per-warp release 修复了 cluster 的主要同步热点，但结果来自未合并实验分支，不改变当前代码的事实边界。

## 13. 未合并的后续实验：bankfix 与 per-warp cluster release

当前分支收敛后，又做了一轮 H100/B300 集成对比。这一轮回答三个问题：packed shared-state access 是否值得保留、去掉每 chunk 的 CTA-wide barrier 后 cluster 是否受益，以及 5/6-warp 的完整实现各自适合哪些 shape。

### 13.1 对比版本与证据边界

| 缩写 | V tile | CTA threads | `__maxnreg__` ceiling | Shared state conversion | Input-stage release |
|---|---:|---:|---:|---|---|
| O0 | 128 | 192 | 168 | 两次 scalar access | 无 cluster |
| O1 | 128 | 192 | 168 | packed `float2` / `uint32_t` | 无 cluster |
| S5 | 64 | 160 | 128 | packed | plain split |
| C5 | 64 | 160 | 128 | packed | per-warp cluster release |
| S6 | 64 | 192 | 112 | packed | plain split |
| C6 | 64 | 192 | 112 | packed | per-warp cluster release |

5-warp 和 6-warp 都固定使用四个 MMA consumer warp。5-warp 让 warp 4 同时管理 input/output pipeline；6-warp 让 warp 4/5 分别管理 input/output。必须注意，本轮同时改变了 I/O warp 拓扑和 register ceiling：S5/C5 使用 128，S6/C6 使用 112。因此数据只能说明“当前完整实现谁快”，不能证明 warp 数本身的因果关系。

测试覆盖 H100 80GB HBM3 和 B300 SXM6，使用 BF16 Q/K/V/G、FP32 initial/final state、CUDA Graph、256 MiB L2 scrub、30 次 warmup、200 次 iteration、6 次 repeat。两套集成构建均通过 22 个定向测试，所有 benchmark 的 output 和 final state 与 original bitwise exact。

### 13.2 Per-warp release 修复了什么

第一轮 cluster 把四个 MMA warp 的 input-stage release 压缩成一个代理 arrival，因此每个 chunk 都需要 128-thread `NamedBarrier`。NCU 中 `BAR.SYNC.DEFER_BLOCKING` 后的 `MEMBAR.ALL.CTA` 约有 18,950 个 barrier not-issued samples，占全部 not-issued samples 约 8%。

新协议删除 CTA-wide barrier：四个 MMA warp 各由 elected lane 向 cluster 中两个目标 CTA 的 empty barrier arrive。每个目标 stage 的 expected arrivals 为：

$$
4\text{ MMA warps per source CTA}\times2\text{ source CTAs}=8.
$$

Producer 收齐 8 次 arrival 后才能回收 stage，因此生命周期仍安全，较快的 MMA warp也不再每个 chunk 等待最慢 warp。修复后，cluster 在多组低并行度 workload 上超过 plain split，说明第一轮 cluster 的负结论有一部分来自同步协议，而不是 multicast 机制本身。

### 13.3 最终性能摘要

下表是相对 O0 的 geometric-mean speedup：

| GPU / workload | O1 | S5 | C5 | S6 | C6 | 最优路径 |
|---|---:|---:|---:|---:|---:|---|
| H100，默认 12 case | 1.012x | 1.203x | **1.253x** | 1.226x | 1.246x | C5 |
| H100，H12 子集 | 1.011x | 1.324x | 1.324x | 1.322x | **1.373x** | C6 |
| H100，H96 子集 | 1.013x | 1.092x | **1.185x** | 1.136x | 1.132x | C5 |
| B300，默认 12 case | 1.008x | 1.111x | **1.121x** | 1.084x | 1.117x | C5 |
| B300，H12 子集 | 1.010x | 1.256x | 1.274x | 1.255x | **1.292x** | C6 |
| B300，H96 子集 | **1.006x** | 0.983x | 0.986x | 0.936x | 0.965x | original/O1 |
| H100，大 batch H96 | **0.980x** | 0.948x | 0.971x | 0.934x | 0.929x | O0 |
| B300，大 batch H96 | **1.016x** | 0.952x | 0.949x | 0.883x | 0.888x | O1 |

低并行度 H12 上 C6 最稳；H100 部分高-head、但仍处于 V-split 获益区间的 shape 中 C5 更稳；B300 H96 和任意 CTA 已充分覆盖 GPU 的大 batch 应回到 original。Cluster 可以减少公共 TMA/L2 流量，但仍无法恢复原始 `V=128` CTA 的计算密度。

<details>
<summary>H100 默认 workload 原始延迟（微秒）</summary>

| Case | O0 | O1 | S5 | C5 | S6 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| pure 1x8192, H12 | 646.30 | 638.98 | 479.42 | 480.74 | 481.34 | **460.66** |
| pure 2x4096, H12 | 366.40 | 360.35 | 278.02 | 278.82 | 279.81 | **266.08** |
| pure 4x2048, H12 | 225.89 | 220.99 | 178.46 | 178.91 | 178.50 | **172.80** |
| pure 1x8192, H96 | 1048.32 | 1032.56 | 1012.61 | 1008.80 | 1024.64 | **1006.11** |
| pure 2x4096, H96 | 931.84 | 919.55 | 894.66 | **868.48** | 1033.09 | 1039.66 |
| pure 4x2048, H96 | 983.92 | 960.70 | 905.89 | **882.75** | 916.29 | 923.65 |
| mixed d7/p8185, H12 | 655.78 | 652.40 | 487.65 | 487.17 | 488.35 | **473.70** |
| mixed d15/p8177, H12 | 660.99 | 656.32 | 493.31 | 493.73 | 493.47 | **476.86** |
| mixed d31/p8161, H12 | 670.30 | 666.27 | 503.84 | 500.67 | 501.54 | **485.82** |
| mixed d7/p8185, H96 | 1383.81 | 1373.92 | 1220.77 | **1029.73** | 1038.94 | 1031.90 |
| mixed d15/p8177, H96 | 1385.38 | 1375.65 | 1225.38 | **1035.20** | 1049.79 | 1039.10 |
| mixed d31/p8161, H96 | 1415.20 | 1398.62 | 1251.42 | **1139.87** | 1148.59 | 1198.05 |

</details>

<details>
<summary>B300 默认 workload 原始延迟（微秒）</summary>

| Case | O0 | O1 | S5 | C5 | S6 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| pure 1x8192, H12 | 561.12 | 554.91 | 442.30 | **431.90** | 442.24 | 432.00 |
| pure 2x4096, H12 | 309.31 | 304.93 | 247.58 | 243.49 | 247.68 | **241.54** |
| pure 4x2048, H12 | 182.27 | 180.06 | 149.71 | **147.33** | 149.38 | **147.33** |
| pure 1x8192, H96 | 786.43 | **778.30** | 820.06 | 811.71 | 812.43 | 794.53 |
| pure 2x4096, H96 | 688.06 | 682.11 | 637.95 | **636.74** | 812.70 | 722.59 |
| pure 4x2048, H96 | 612.29 | **602.18** | 638.70 | 640.83 | 685.95 | 681.86 |
| mixed d7/p8185, H12 | 571.33 | 567.10 | 448.61 | 446.11 | 450.43 | **436.03** |
| mixed d15/p8177, H12 | 574.40 | 570.18 | 452.42 | 448.58 | 454.56 | **438.11** |
| mixed d31/p8161, H12 | 581.66 | 577.63 | 461.60 | 455.58 | 460.70 | **442.24** |
| mixed d7/p8185, H96 | 820.19 | **819.01** | 845.63 | 843.55 | 841.60 | 830.34 |
| mixed d15/p8177, H96 | **845.89** | 846.94 | 874.53 | 871.20 | 870.27 | 857.52 |
| mixed d31/p8161, H96 | **892.96** | 892.99 | 922.40 | 917.28 | 920.45 | 912.35 |

</details>

<details>
<summary>大 batch 原始延迟（微秒）</summary>

| GPU / Case | O0 | O1 | S5 | C5 | S6 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| H100, B8/T2048/H96 | **1712.48** | 1745.18 | 1819.14 | 1765.34 | 1819.97 | 1832.94 |
| H100, B16/T2048/H96 | **3381.54** | 3467.28 | 3553.25 | 3479.47 | 3616.80 | 3633.57 |
| H100, B32/T2048/H96 | **6762.03** | 6869.74 | 7109.52 | 6973.25 | 7305.97 | 7327.23 |
| B300, B8/T2048/H96 | 1161.18 | **1146.75** | 1201.02 | 1206.18 | 1308.80 | 1302.53 |
| B300, B16/T2048/H96 | 2192.48 | **2165.66** | 2309.06 | 2311.04 | 2499.55 | 2487.46 |
| B300, B32/T2048/H96 | 4404.96 | **4302.72** | 4687.78 | 4706.24 | 4982.06 | 4941.47 |

</details>

### 13.4 最终 C5 的 H100 NCU：收益与成本落在哪里

2026-08-17 在 `experiment/vsplit-5warp-final@0c8a2d0` 上补跑了最终版 C5，并用同一二进制内的 S5 作对照。Profile shape 为 `N=16, T=8192, H=96`，共 131072 tokens；K2 grid 为 `(192,16,1)`，共 3072 CTA，block 为 160 threads。NCU 采用 application replay，完整 metric set 需要 39 passes。

这是一组刻意放大的、grid 已完全覆盖 H100 的 K2-only profile，用于定位 steady-state 成本；它不是第 13.3 节默认 benchmark case。两条路径分别采集，且使用 `--cache-control none --clock-control none`，所以 duration 只作为方向性证据，不能代替锁频、交错重复的端到端 benchmark。

| 指标 | C5 final | S5 final | C5 相对 S5 |
|---|---:|---:|---:|
| K2 duration | 6.305 ms | 7.634 ms | -17.4% |
| DRAM read | 14.220 GB | 17.430 GB | -18.4% |
| DRAM write | 3.336 GB | 3.337 GB | 基本相同 |
| L2 requests | 699.98 M | 816.88 M | -14.3% |
| L2 sectors | 1053.85 M | 1442.46 M | -26.9% |
| Long Scoreboard not-issued samples | 57,058 | 223,746 | -74.5% |
| MIO not-issued samples | 20,200 | 35,389 | -42.9% |
| `No Eligible` | 45.76% | 56.29% | -10.53 pp |
| Eligible warps / scheduler | 1.01 | 0.70 | +44.3% |
| Tensor pipe active | 33.40% | 27.10% | +6.30 pp |
| Registers / thread | 116 | 118 | -2 |
| Achieved active warps / SM | 14.6 | 14.6 | 相同 |

这说明最终 per-warp release 已经兑现了 multicast 的主要目标：公共 workspace 流量下降，TMA consumer polling 对应的 Long Scoreboard 大幅减少，scheduler 有更多 ready warp。C5 与 S5 都是 3 CTA/SM、约 14.6 active warps/SM；在这个大 grid case 中，没有看到 cluster 通过降低 residency 支付成本。

最终 C5 仍有三类可见代价：

1. **Cluster fence/release 控制**：C5 新增 3,620 个 membar not-issued samples；barrier samples 为 3,831，对比 S5 的 3,446，增加 11.2%。热点包括 `FENCE.VIEW.ASYNC.S`、per-warp cluster arrive、阶段 phase check 和末尾 cluster sync。
2. **更多控制指令**：总执行指令增加约 1.0%，branch 指令增加约 9.0%。这是 per-warp/cluster 生命周期管理的直接开销，但远小于被消掉的 Long Scoreboard 等待。
3. **Shared bank conflict 略增**：总 conflict 从 48.66 M 增至 64.07 M，增加 31.7%；shared-load/store conflict 分别增加 22.5%/10.8%。不过两条路径都存在 state conversion store warning，C5 的总 shared LSU wavefront 只增加约 1.4%，而且最终仍更快，所以它是次要优化点，不是当前主瓶颈。

最终判断：旧版 cluster 的主要问题确实是 CTA-wide release 协议；最终 C5 已经基本消除这项成本。对这个 fully-resident H100 K2 workload，multicast 节省的 L2/DRAM 请求和 TMA 等待明显大于新增的 fence、barrier 与控制开销。仍需注意，cluster 成对 placement 和 tail wave 成本更可能在小 grid 或不规则 shape 暴露，本报告不能排除那部分成本。

### 13.5 当前 5-warp delayed output release：B300 补测与 H100 NCU

在上一轮 C5 的 per-warp input release 基础上，当前工作树又把 output stage 的回收延迟到下一个 stage 已经完成写回之后：第一次 `tma_store_wait<0>()` 只等待当前写回事务，下一轮开始前再执行 `store_pipeline.consumer_release(out_release)`。这样不会把 output stage 提前交给下一轮 TMA，同时保持计算 warp 不必等待无关的 input load。该版本仍是实验代码，尚未合入产品 dispatch。

本轮统一使用 BF16 Q/K/V/G、FP32 state、CUDA Graph、冷 L2（256 MiB scrub），并检查 split/cluster 的 output 和 final state 与 original bitwise exact。B300 默认 12 case 和三组大 batch 使用 30 warmup、200 iterations、6 repeats；H100 的 `N=1,T=8192,H=96` 使用同一正式协议，其余 H100 大 batch 是先前的短时方向性补测（5 warmup、20 iterations、2 repeats）。与上一节的 C5/C6 不同，这里只比较当前 5-warp delayed-release 版本的 `split` 和 `cluster`，没有重新改变 register ceiling。

| GPU / workload | Original (us) | 5-warp split (us) | 5-warp cluster (us) | Split speedup | Cluster speedup | Exact |
|---|---:|---:|---:|---:|---:|:---:|
| H100, pure 1x8192 H96 | 1035.55 | 1029.15 | 1019.14 | 1.006x | 1.016x | yes |
| H100, pure 4x2048 H96 | 958.00 | 901.36 | 877.54 | 1.063x | 1.092x | yes |
| H100, pure 8x8192 H96 | 6639.09 | 7104.61 | 6775.81 | 0.935x | 0.980x | yes |
| B300, pure 1x8192 H12 | 555.14 | 434.37 | 438.40 | 1.278x | 1.266x | yes |
| B300, pure 1x8192 H96 | 780.42 | 819.30 | 815.26 | 0.953x | 0.957x | yes |
| B300, pure 4x2048 H96 | 604.29 | 647.26 | 651.36 | 0.934x | 0.928x | yes |
| B300, pure 8x2048 H96 | 1146.98 | 1212.54 | 1222.62 | 0.946x | 0.938x | yes |
| B300, pure 16x2048 H96 | 2201.63 | 2371.50 | 2367.58 | 0.928x | 0.930x | yes |
| B300, pure 32x2048 H96 | 4402.24 | 4836.29 | 4815.92 | 0.910x | 0.914x | yes |

H100 的 `N=4,T=2048,H=96` 是本轮大 batch 中唯一明显超过 original 的点；扩大到 `N=8,T=8192,H=96` 后，cluster 仍比 split 快 4.6%，但比 original 慢 2.1%。B300 的正式大 batch 三组均落后 original：`B=8` 为 0.938x，`B=16` 为 0.930x，`B=32` 为 0.914x。也就是说，延迟 output release 能回收一部分 split/cluster 的尾部等待，却不能改变“大 batch 已经有足够 CTA 时，重复 workspace 流量和半宽 MMA 是主成本”的结论。表中 H100 的 `B=4`、`B=8,T=8192` 使用较短的 warmup/iteration 作为方向性补测；B300 默认 12 case 和三组大 batch 使用上一段的正式协议。

按默认 12 case 的 geometric mean，当前 B300 delayed-release 为 split 1.107x、cluster 1.098x；上一轮旧 C5 的记录是 1.111x、1.121x。因此这次 output release 改动没有让 5-warp cluster 超过旧 C5，反而略低约 2%。在 B300 三组大 batch 上，当前 cluster 的 geometric mean 约 0.927x，旧 C5 约 0.949x，也没有扭转大 batch 的负收益。

当前 H100 NCU 文件已下载到 [`ncu-reports/vllm-flashkda/h100-delayed-release-n16-t8192-h96-full.ncu-rep`](../ncu-reports/vllm-flashkda/h100-delayed-release-n16-t8192-h96-full.ncu-rep)。它是当前 delayed-release cluster、`N=16,T=8192,H=96`、单次 kernel 的完整 39-pass profile。关键读数为：kernel duration 6.438 ms，DRAM read/write 14.219/3.317 GB，L2 hit 44.22%，L2 throughput 87.81%，achieved active warps/SM 14.61，active/eligible warps per scheduler 3.65/1.00，`No Eligible` 45.92%。因此仍然是 memory/依赖等待主导，增加 resident warp 数本身不会自动转化成 issue throughput。

B300 的 NCU 在同一 `N=16,T=8192,H=96` 单 case 上复试仍在 `Failed to prepare kernel` / `Unknown error on device 0` 退出，没有生成可用 report；这属于 profiler 权限或当前 B300 NCU 支持问题，不能把失败当成 kernel 性能数据。

### 13.6 各修改的最终判断

- **Per-warp cluster release**：明确有效。它移除了 cluster 特有的 CTA-wide 同步热点，并使 cluster 在多个低并行度 workload 上超过 plain split；仍未合入当前分支。
- **Packed shared-state conversion**：是合理的局部 bank-access 改写，但 O1 的端到端变化约为 -2%～+1.6%，接近模板代码生成和测量噪声量级，不宜无条件默认。
- **5/6-warp**：H12 plain split 几乎持平；cluster 中 C6 在 H12 更快，C5 在部分 H96 更稳。由于 112/128 register ceiling 混入实验，不能宣称某个 warp 拓扑天然更优。
- **Shape dispatch**：无可靠 heuristic 时保留 original 默认；只有 local heads 少、grid 明显不足时才考虑 V-split。CTA 已充足时，任何 split/cluster 变体都不应替换 original。

后续若要产品化 C5/C6，必须先完成相同 register policy 的消融，分别核对 registers、allocated registers、spill、achieved residency 和 SASS；不能直接把本轮完整实现的胜负固化成 5/6-warp 因果结论。

## 14. 面试复述版本

### 14.1 90 秒主线

> 这个工作跨了两代实现。最早在 FlashKDA 上做 V-split 时，我用 NCU 定位过两类现象：RTX 5060 Ti 主要是低 TLP 下的 MIO/issue starvation，H100 则暴露出 TMA barrier 和 L2 tag/request latency；这两组数据属于早期 kernel。之后我把 V-split 移植到已经集成 Flash-Flash-KDA 的 vLLM 基线，再完成最终方案比较。
>
> 在第二代基线上，原始 kernel 每个 head 只有一个 CTA，在 TP 把 local head 数切小后，长 prefill 的 CTA 数不足。我把 V 维从 128 切成两个 64-column CTA，低并行度场景在 H100/B300 上能快 20%～36%。代价是每个 CTA 的 MMA N 维减半，而且 K/Q 等公共 workspace 被重复读取，所以 head 多、SM 已经打满时会回退。
>
> 我又做了 cluster multicast，让两个 V-slice CTA 只加载一次公共 workspace。第一轮 NCU 确认 DRAM/L2 压力下降，但每 chunk 的 CTA-wide barrier 抵消了收益；把 release 改成四个 MMA warp 分别 arrive 后，cluster 才在多个低并行度 case 超过 plain split。5-warp fused-IO 能把理论 residency 从 2 CTA/SM 提高到 3，但更多 active warp 没有稳定变成 eligible warp。后续 C6 在 H12 更快、C5 在部分 H96 更稳，不过二者还混入了不同 register ceiling，不能作为纯 warp 数结论。
>
> 我还检查了 state 寄存器压力，发现 kernel 本来只常驻 C-layout state，转置后的 B fragment 是临时值。强制复用寄存器虽然减少静态 live set，却破坏 ptxas 的 register renaming 和 ILP，性能下降 8%～14%。当前分支最终保留最简单的 6-warp plain V-split，只在 CTA 不足的 shape 上自动启用；cluster、5-warp、bankfix 和 state 改写仍是未合并实验。

### 14.2 常见追问

**问：计算和 DRAM 吞吐都低，kernel 为什么还会慢？**

答：吞吐低只排除了峰值 compute/bandwidth bound，不排除 latency bound。先看 scheduler；本例 `No Eligible` 很高或 active warp 增加但 eligible 不变，说明 warp 被依赖卡住，硬件没有 ready instruction 可发。

**问：Long Scoreboard 很高，为什么 DRAM 只有 40%？**

答：这是第一代 FlashKDA 的 H100 报告。Long Scoreboard 表示依赖未就绪，不表示 HBM 字节带宽已满。这里主要等待 TMA barrier，上游 L2 tag/request 已接近峰值，而 L2 data 和 HBM 只有约 35%～41%。请求处理和相关 latency 足以卡 consumer，无需打满 HBM；不能把这组数值直接套到后来的 Flash-Flash-KDA kernel。

**问：Cluster 已经减少 TMA，为什么不快？**

答：第一轮不快，是因为它减少公共 global/L2 请求的同时增加了 CTA-wide barrier、cluster placement 和控制成本，也没有恢复 split 后损失的 CTA 内复用。Per-warp release 删除主要同步热点后，cluster 在部分低并行度 case 已能超过 plain split；CTA 充分时仍会输给 original。

**问：为什么 3 CTA/SM 没明显快于 2 CTA/SM？**

答：理论 occupancy 是资源允许的上限，不保证 grid 足够，也不保证新增 warp 可发射。多个 CTA 若在同一 TMA/L2/barrier 阶段相关等待，active warp 会增多，eligible warp和 issue throughput不变。

**问：为什么减少寄存器反而变慢？**

答：编译器用额外物理寄存器做跨 iteration renaming，换取 shared load、MOVM 和 MMA 的 ILP。强制复用寄存器会新增 RAW 依赖。寄存器少只有在它真正提高 occupancy 且不破坏关键路径时才可能有收益。

**问：MIO throttle 就是 bank conflict 吗？**

答：不是。MIO 是更广的指令队列压力；bank conflict 要看 raw shared bank-conflict counter。`Excessive wavefront=0` 也不能证明 bank conflict 为 0，因为它检查的是 actual 相对 ideal 的额外 wavefront，是另一件事。

**问：最终为什么选择 plain 6-warp？**

答：这是当前分支的收敛选择：它在 CTA 不足时有明确收益，同步协议最简单，适合先合入并用保守 heuristic 自动 dispatch。后续 cluster 虽因 per-warp release 有了更好结果，但 5/6-warp 数据混入不同 register ceiling，尚不足以安全替换当前路径。

## 15. 实验资产

本地/远端分支：

```text
feature/k2-vsplit                最终 plain V-split PR 路径，提交 1fb1d9c
feature/vsplit-pr                早期 plain V-split 前身，提交 6446ec1
c @ 715e464                      早期 V-split + cluster 工作树
experiment/cluster-5warp-io      5-warp cluster，本地分支头 36eaf09，核心提交 4bdcb45
experiment/split-5warp-ablation  5-warp non-cluster 对照，提交 b905efc
experiment/vsplit-5warp-final    bankfix + per-warp release，提交 0c8a2d0
experiment/vsplit-6warp-per-warp-release
                                6-warp per-warp release，提交 aad7669
```

最终对比的 Modal Volume：

```text
vllm-flashkda-final5-bankfix
vllm-flashkda-final6-bankfix
vllm-flashkda-perwarp            O0 控制
vllm-flashkda-perwarp6           无 bankfix 的 6-warp 控制
```

State transpose 实验的 Modal run：

```text
ap-hbQ8wE0IJKjW6jPiF2Mhut       对齐版本全平台安装
ap-e7ov4TyfH5FvITcJOdlFCS       correctness
ap-WmGyWiT6EVn4lChIFNFYaV       H100 benchmark
ap-fi13oZr2dotwE3CiXf6Vkt       恢复基线版本
ap-brwtptlWQ4d08XdSWE9ixk       强依赖版本全平台安装
ap-xLlJqZ0uRGAjmM1o0xSBXC       强依赖 correctness
ap-346YPIGlmv5f4NDwDyISaX       强依赖 H100 benchmark
```

原始 NCU 指南与两台 GPU 的完整分析：

```text
docs/ncu-kernel-profiling-guide.md
```

关键报告：

```text
ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep
ncu-reports/flashkda/h100__original-vsplit__b8-t2048-h96__full.ncu-rep
ncu-reports/vllm-flashkda/h100__warp6-original-split-cluster__n1-t8192-h96__full.ncu-rep
ncu-reports/vllm-flashkda/h100__original-warp6split-warp5cluster__n1-t8192-h96__pre-perwarp-full.ncu-rep
ncu-reports/vllm-flashkda/h100__original-warp6split-warp5cluster__n4-t2048-h12__pre-perwarp-full.ncu-rep
ncu-reports/vllm-flashkda/h100__warp5cluster-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep
ncu-reports/vllm-flashkda/h100__warp5split-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep
```

当前工作到此暂停。若未来重启，优先做 shape-aware dispatch 或 WGMMA/UMMA 的全新数据流设计，而不是继续微调现有 cluster/warp 数。
