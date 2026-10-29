# vLLM FlashKDA NCU 热点分析

## 范围

本报告分析 `ncu-reports/vllm-flashkda` 中 H100 的完整 NCU 报告，重点使用：

- `h100__original__n16-t8192-h96__cache-all-full.ncu-rep`
- `h100__warp5cluster-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__warp5split-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__baseline-current-toolchain__n16-t8192-h96__cache-none-full.ncu-rep`

这些报告不是当前 `OutputStages=3` 的新 NCU 采样；它们用于定位当前 kernel 的瓶颈。最近 B300 的 `OutputStages=3` bench 只用于验证端到端结果。

## 资源和驻留

NCU 对最终 5-warp cluster kernel 的记录是：

| 指标 | 数值 |
|---|---:|
| registers/thread | 116（硬件分配 120） |
| shared memory / CTA | 69.632 KiB |
| shared-memory config | 233.472 KiB |
| register occupancy limit | 3 CTA/SM |
| shared-memory occupancy limit | 3 CTA/SM |
| barrier occupancy limit | 32 CTA/SM |

因此 C5 的 3 CTA/SM 是真实可驻留配置，不是只由理论线程数推出来的。历史 NCU 中的 original 行显示 168 registers/thread、99.456 KiB shared、register limit 2 CTA/SM；该报告记录的 block size 是 192 threads。当前源码的 original K2 launch 已是 256 threads，在同样 168 registers/thread 假设下，寄存器账面为 `256×168=43,008`，两个 CTA 需要 86,016，超过 64K，因此当前 original K2 不能同时驻留两个 CTA。历史报告与当前代码不能直接混用。

`OutputStages=3` 只增加一个 `16×64` BF16 output tile，即 2,048 B。按当前 69,632 B/CTA 计算，新 CTA 约为 71,680 B，3 CTA 需要 215,040 B，仍低于 B300/H100 的 233,472 B 配置，因此不会改变这一级 occupancy。

## 主要瓶颈

### 1. 长 scoreboard 是第一主因，但不是单一的“某一条 load”

5-warp cluster 报告的 warp sampling 原始计数为：

| Stall reason | Samples |
|---|---:|
| long scoreboard | 438,061 |
| wait | 621,812 |
| short scoreboard | 296,014 |
| dispatch stall | 167,330 |
| not selected | 345,630 |
| math pipe throttle | 139,183 |
| MIO throttle | 101,627 |
| barrier | 27,941 |

long scoreboard 确实是主要等待来源，但它表示 warp 在等较长延迟的内存/异步事务依赖；source view 把 stall 归到消费者 PC，而不是必然归到发起事务的那条 C++ 语句。因此 912 行高采样不能直接证明 910 行 `cute::transform` 是根因。

### 2. `cute::transform(identity{})` 不是有效的内存隐藏点

这段代码生成的主要数据流是：

```cpp
copy(..., tCrAi_kr_view);                 // shared -> fragment registers
cute::transform(tCrAi_kr, ring_A_kr[i], identity{}); // register -> register
ring_g0[i] = g_total(...);                // shared -> register
```

`transform(identity{})` 只搬运已经在寄存器中的 fragment，不发起 TMA、global load 或 shared load。截图中 910 行约 0.11%，912 行约 4.35%，也支持“912 的消费者等待高于 910 的寄存器搬运”这一判断。直接改写 identity transform 没有足够收益依据。

### 3. `g_total` 读取是有证据的候选点

`g_total` 来自 input stage 的 shared-memory TMA load。它位于 pipeline buffer 中，读取时必须依赖对应 input stage 已经完成。若 source view 在 912 行对应的是 barrier phase check 或 shared load，那么这部分等待来自 input TMA/mbarrier 的完成时序，而不是浮点计算。

但当前报告没有把 912 行唯一映射到一条 SASS 指令；必须在 source view 中展开该行的 SASS dependency，区分 `LD.shared`、`SYNCS.PHASECHECK` 和地址计算。不能仅凭 C++ 行号判断。

### 4. NCU 中的 `barrier.h:425` 是一次性的 initial-state wait

后续对 source view 的核对显示，截图中选中的 `barrier.h:425` 对应 CUTLASS `ClusterTransactionBarrier::wait(0)` 的 inline PTX。当前 `fwd_kernel2.cuh` 中这个对象只有两处 `.wait(0)` 调用，分别位于 BF16/FP32 initial-state TMA load 之后（约 359、392 行）；主循环没有再次调用 `state_acc_tma_barrier.wait(0)`。因此该 source stall 是每个 CTA 的一次性前置等待，不是每个 recurrence tile 重复发生的等待，也不是大 batch steady-state 的主要证据。

后续主循环里的等待来自 `load_pipeline.consumer_wait`、`store_pipeline.consumer_wait` 及其对应的 pipeline barrier，必须通过它们各自的 SASS barrier 地址区分，不能把 `barrier.h:425` 的采样泛化到整个 pipeline。

### 5. `__syncthreads()` 是真实的阶段边界，不能直接删除

FP32 state 输出前的同步位于：

```cpp
__syncthreads();
smem_cvt_bf16_to_fp32(...);
```

前面所有 warp 都写 `state_acc`，后面所有线程都读取它做 BF16→FP32 转换。该 barrier 的占比不如 long scoreboard 主导，但它保护的是跨 warp 的真实 RAW 依赖。`fence_view_async_shared()` 只能保证可见性，不能替代等待；`store_pipeline.producer_commit()` 也只发布 output pipeline 状态，不能发布 `state_acc` 的 CTA 内部完成。

删除同步或仅替换成普通 fence 会引入数据竞争，排除该方案。

## 值得实施的修改

### A. 先做 input-stage / `g_total` 依赖的定点验证

这是当前最有证据支持的方向。对 912 行对应的 SASS 做 source dependency 展开，确认它等待的是 input TMA barrier 还是 shared load。若确认是 input stage 完成晚：

1. 检查 `g_total` 是否能与同一 input stage 的其他小字段合并为一次 TMA 事务；
2. 检查该 stage 是否在 `consumer_wait` 之后才读取，避免过早读取造成 scoreboard 暴露；
3. 保持至少一个 tile 的预取距离，比较 `PREFETCH=1` 与 `PREFETCH=2`，同时确认寄存器不会把 occupancy 从 3 CTA 降下来。

这是可验证的 latency-hiding 改动，不能只凭减少 C++ 行数判断收益。

### B. 保留 delayed output release，和 `OutputStages=3` 分开做消融

当前延迟释放方案已经通过正确性测试。它解决的是 output shared stage 被过早重用的问题，和 input `g_total` 等待属于不同依赖。应分别比较：

| 版本 | output stage | release |
|---|---:|---|
| A | 2 | 原逻辑 |
| B | 2 | delayed |
| C | 3 | 原逻辑 |
| D | 3 | delayed |

B300 默认 bench 已显示 `OutputStages=3` 没有带来稳定端到端收益，所以它目前只能作为低成本保留项，不能当作主优化方向。

### C. 把 FP32 state 转换和最后一次 output store 重叠

这是唯一值得尝试的 barrier 方向。目标不是删除 `__syncthreads()`，而是让 barrier 之前有更多独立工作：

1. 提交最后 output tile 后，先让 store warp 发起可重叠的 TMA store；
2. 其他 warp 在确认最后 `state_acc` 写入完成后进入转换；
3. 用专门的 mbarrier 发布转换完成，再由 state store warp 等待。

实现前必须确认 state/output shared buffer 不重叠、转换读写范围完整，并用 correctness 检查尾 tile。若转换本身已经短于最后 output store，这个改动不会收益；但它有明确的重叠目标，值得做一次小规模实验。

## 明确排除的方案

- 删除 `__syncthreads()`：违反 `state_acc` 的跨 warp 读写依赖。
- 只把 `cute::transform(..., identity{})` 改成手写循环：仍是寄存器搬运，不改变内存事务或等待来源。
- 继续增加 output stage：当前 O3 已在 B300 默认 bench 中验证没有稳定收益，O4 只会增加 shared footprint，缺少证据支持。
- 仅降低 `__maxnreg__`：当前 C5 已真实保持 3 CTA/SM；继续压寄存器可能损失跨迭代 ILP，不能把 occupancy 数字当成性能收益。

## 结论

当前 kernel 的主要矛盾是长延迟内存/异步事务没有被足够的独立工作覆盖，而不是 910 行寄存器 transform 或单独一个 CTA barrier。最值得继续的顺序是：

1. 对 912 行做 SASS dependency 定位，确认 `g_total` 等待的具体生产者；
2. 在不降低 3 CTA/SM 的前提下试 `PREFETCH=2` 或调整 input stage 消费时序；
3. 再试 FP32 state 转换与最后 output store 的重叠；
4. `OutputStages=3` 保留作低成本配置，但不作为主要优化结论。

## 2026-09-09 新报告复核

文件：`ncu-reports/vllm-flashkda/h100-delayed-release-n16-t8192-h96-full.ncu-rep`。

该报告的 kernel 是 `(160 threads, grid=(192,16,1))`，资源仍为 120 registers/thread、69.632 KiB shared、3 CTA/SM。原始采样计数为：`wait=157,850`、`long_scoreboard=107,146`、`short_scoreboard=74,767`、`not_selected=84,527`、`barrier=6,509`；warpsampling 计数为 `wait=630,102`、`long_scoreboard=427,631`、`not_selected=337,210`。这些计数不是同一维度的百分比，不能直接相加或把 `wait` 解释成一次 mbarrier 的事务等待。

在 source/SASS CSV 中，912 行附近对应的关键指令为：

```text
0x2b23a97dbb30  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R82+URZ+0x10bc0], R6
  all=857, not-issued=83, stall_wait=266, stall_long_sb=0
0x2b23a97dbb50  @!P0 BRA 0x2b23a97ddf70
  all=48632, not-issued=25633, stall_wait=0, stall_long_sb=47337
```

这组数据直接说明：`TRYWAIT` 指令本身只有 266 个 `stall_wait`，真正的大量等待归因在后面的条件 branch，且是 47,337 个 `long_scoreboard`。`stall_wait` 的大总量主要来自其它固定延迟依赖（报告中最高的 `FFMA`、`F2FP.PACK_AB`、`LDS/LDSM`、`HMMA` 等指令），不是 mbarrier 事务等待本身。

同一报告中：

```text
WARPSYNC.ALL                  all=27442, not-issued=8520, barrier=0
BAR.SYNC.DEFER_BLOCKING       all=9738,  not-issued=3455, barrier=792
```

因此附近出现 `WARPSYNC` 不能说明 `__syncthreads()` 可以替换成 warp sync；两者是不同的机器同步指令，且 block-level `BAR.SYNC` 的 barrier stall 只有 792。只有当参与读写的线程集合确定为单个 warp 时才可以改用 `__syncwarp()`。
