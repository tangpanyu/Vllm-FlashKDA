# Nsight Compute CUDA Kernel 性能分析指南

> 路径标注（2026-10-29）：本文旧版 `tests/bench_vsplit.py` 示例从 `experiments/flashkda-v1/` 运行；其 `ncu-reports` 软链接指向根目录的统一报告库。vLLM 新版入口为根目录的 `benchmarks/bench_vsplit.py`，具体命令见 [V-split / Cluster](vsplit-cluster.md)。

这是一份纯 Nsight Compute（NCU）使用指南。目标是解决三个问题：

1. 用什么命令只抓目标 kernel。
2. 报告输出的每一组数据怎样读。
3. 怎样从汇总指标追到 CUDA 源码和真实 SASS。

文中的 K2 示例来自 RTX 5060 Ti（SM120）和 H100（SM90）上的真实报告，
分析工具为 Nsight Compute 2025.4。不同 GPU 上可采集的指标名称和硬件峰值可能
变化，但分析顺序不变；跨 GPU 比较时还必须控制 grid、stage 和 shared-memory
配置。

官方资料对应关系：

- 命令行选项：[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- 指标、stall 和内存表：[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- Source 页面和 section 文件：[Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html)

## 1. 先建立 NCU 的整体模型

一次 NCU 分析包含四层：

```text
运行目标程序
    ↓ kernel/launch 过滤
采集 set、section 或单独 metric
    ↓ 生成 .ncu-rep
查看 Details / Raw / Source
    ↓
从汇总瓶颈定位到具体 SASS 指令
```

### 1.1 Set、Section 和 Metric

三者关系是：

```text
Set
 ├── Section
 │    ├── Metric
 │    ├── Metric
 │    └── Rule
 └── Section
```

- `set` 是一套预定义采集方案，例如 `basic`、`full`。
- `section` 是一类性能问题，例如 `LaunchStats`、`SchedulerStats`。
- `metric` 是具体硬件计数器或派生值。
- `rule` 根据采集结果给出诊断提示，但 rule 只能作为线索，不能代替指标交叉验证。

常用查询命令：

```bash
NCU_BIN=/usr/local/NVIDIA-Nsight-Compute-2025.4/ncu

"$NCU_BIN" --list-sets
"$NCU_BIN" --list-sections
"$NCU_BIN" --section SchedulerStats --list-metrics
"$NCU_BIN" --query-metrics-mode all \
  --query-metrics-collection profiling \
  | less
```

### 1.2 采集和读取是两个步骤

采集：

```bash
"$NCU_BIN" \
  --section LaunchStats \
  --force-overwrite \
  -o report_name \
  -- command arg1 arg2
```

会生成：

```text
report_name.ncu-rep
```

读取已有报告不会重新运行程序：

```bash
"$NCU_BIN" --import report_name.ncu-rep --page details
```

使用 GUI 打开同一份报告：

```bash
ncu-ui report_name.ncu-rep
```

不要把被 NCU 包住的程序自己打印出来的 benchmark 时间当成真实性能。NCU 为了收集计数器可能 replay 同一个 kernel 很多次。

## 2. 最常用的五条采集命令

下面用 FlashKDA V-split K2 演示。替换最后的目标命令即可用于其他项目。

### 2.1 第一步：用最小报告验证过滤器

```bash
"$NCU_BIN" \
  --section LaunchStats \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__launchstats \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

终端至少应该出现：

```text
==PROF== Profiling "...flash_kda_fwd_recurrence..."
==PROF== Report: ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__launchstats.ncu-rep
```

立刻读取它：

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__launchstats.ncu-rep \
  --page details
```

先核对：

- Kernel Name 是否正确。
- Grid Size、Block Size 是否符合预期。
- Registers Per Thread 是否合理。
- Shared Memory Per Block 是否符合模板配置。

这一步不对，后面的指标一律没有意义。

### 2.2 第二步：看整体吞吐和 occupancy

```bash
"$NCU_BIN" \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o kda_overview \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

它主要回答：

- kernel 多久。
- Compute、Memory、L1、L2、DRAM 哪个接近峰值。
- 一个 SM 能驻留多少 CTA 和 warp。
- occupancy 被 shared memory、register、warp 还是 barrier 限制。

### 2.3 第三步：吞吐都低时看 scheduler

```bash
"$NCU_BIN" \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o kda_scheduler \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

优先看：

```text
Issue Slots Busy
Issued Warp Per Scheduler
Active Warps Per Scheduler
Eligible Warps Per Scheduler
No Eligible
Warp Cycles Per Issued Instruction
```

### 2.4 第四步：看内存层级和 shared memory

```bash
"$NCU_BIN" \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o kda_memory \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

这里回答：

- DRAM 是否真满了。
- 流量主要命中 L2，还是去了 DRAM。
- global load/store 是否合并。
- shared load/store 的 request、wavefront 和 bank conflict 是否异常。
- memory pipeline 是忙，还是根本没有足够请求。

### 2.5 第五步：追到源码和 SASS

```bash
"$NCU_BIN" \
  --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables \
  --import-source yes \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o kda_source \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

查看 CUDA 源码关联：

```bash
"$NCU_BIN" \
  --import kda_source.ncu-rep \
  --page source \
  --print-source cuda \
  --resolve-source-file csrc/smxx/fwd_kernel2_1.cuh
```

查看真实 SASS：

```bash
"$NCU_BIN" \
  --import kda_source.ncu-rep \
  --page source \
  --print-source sass
```

导出 CSV，方便筛选指令和逐指令指标：

```bash
"$NCU_BIN" \
  --import kda_source.ncu-rep \
  --page source \
  --print-source sass \
  --csv > kda_source.csv

rg 'LDSM|STSM|HMMA|MMA|FFMA|F2FP' kda_source.csv
```

### 2.6 最后才使用 `--set full`

```bash
"$NCU_BIN" \
  --set full \
  --kernel-name-base mangled \
  -k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*' \
  --launch-count 1 \
  --force-overwrite \
  -o ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full \
  -- python tests/bench_vsplit.py \
     --cases '1,2048,32' \
     --warmup 0 --repeats 1 --inner 1
```

`full` 不是第一步，因为它通常需要很多 replay pass。过滤错一个 kernel，就会浪费大量时间并得到一份无关报告。

## 3. 怎样只抓到想要的 kernel

### 3.1 Kernel name basis

```text
function    简化函数名，通常去掉模板参数
demangled   可读的完整 C++ 名称
mangled     ABI 符号名
```

宽松过滤：

```bash
--kernel-name-base function \
-k 'regex:.*flash_kda_fwd_recurrence.*'
```

严格区分 namespace 或模板实例时，使用 mangled 名：

```bash
--kernel-name-base mangled \
-k 'regex:^_ZN16flash_kda_vsplit25_flash_kda_fwd_recurrence.*'
```

正则一定加单引号，否则 shell 可能展开 `*`。

### 3.2 Launch skip 和 count

```bash
-s 3 -c 1
```

等价于：

```bash
--launch-skip 3 --launch-count 1
```

含义是：在已经满足 kernel filter 的 launch 中跳过前三次，只采第四次。

`--launch-skip-before-match` 则在匹配前对所有 launch 计数，不要和 `--launch-skip` 混用。

### 3.3 抓到函数名但内容为空

依次检查：

1. 是否真的生成了 `.ncu-rep`。
2. 用 CLI 执行 `--import report.ncu-rep --page details`。
3. 是否采集了 section，而不只是启动了 GUI。
4. GUI 是否打开了 Details 或 Source 页面。
5. GUI 和生成报告的 CLI 版本是否兼容。
6. Source 为空时，是否有 `-lineinfo`，采集时是否用了 `--import-source yes`。

## 4. 报告的四个页面分别看什么

### 4.1 Session：先确认报告是谁、在哪台机器上采的

拿到一份别人传来的 `.ncu-rep`，第一步不是看 SOL，而是看 Session：

```bash
"$NCU_BIN" --import report.ncu-rep --page session
```

Session 页面包含三类关键事实：

- `Profiler Command Line`：原始 NCU 命令和被测程序参数；
- `Created`：报告创建时间；
- `Device Attributes`：GPU 名称、SM 数量、compute capability 和显存容量。

只提取最有用的字段：

```bash
"$NCU_BIN" --import report.ncu-rep --page session --csv \
  | rg '^"(Profiler Command Line|Created|display_name|multiprocessor_count|compute_capability_major|compute_capability_minor)"'
```

例如一份旧 warp5 报告可以恢复出：

```text
Profiler Command Line ... --case pure_n16_t8192_h96 --profile-variant cluster ...
display_name          NVIDIA H100 80GB HBM3
multiprocessor_count  132
compute capability    9.0
```

这比文件名可靠。文件可能被复制、改名或放错目录，但 profiler command line 和 device attributes 保存在报告内部。注意 NCU 不会自动记录 Git commit；需要把 commit 写进文件名、采集脚本或旁边的索引文档。

### 4.2 Details

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep \
  --page details
```

这是日常主页面，按 section 显示整理后的表格和 rule 建议。适合回答“瓶颈属于哪一类”。

快速筛选：

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep \
  --page details \
  | rg -i 'Duration|Throughput|Issue Slots|No Eligible|Shared Memory|Registers|Occupancy'
```

### 4.3 Raw

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep \
  --page raw
```

这里显示 metric 全名和原始值。适合：

- 查 GUI 中一个值对应哪个 metric。
- 写脚本比较两个报告。
- 检查派生指标是否与底层计数一致。
- 发现 rule 的汇总口径是否误导。

### 4.4 Source

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep \
  --page source --print-source sass
```

这里把指标归因到具体指令。适合回答：

- 哪条 shared load/store 产生额外 wavefront。
- 哪段代码附近采样到 MIO、scoreboard 或 wait。
- 编译器有没有把源码中的 prefetch 真正提前。
- 两个版本的实际指令顺序是否不同。

注意：stall sampling 标在某条指令上，表示 warp 的 PC 当时停在那里，不保证这条指令就是根因。根因经常是前面生产它输入的 load、MMA 或 barrier。

### 4.5 从报告识别实际 kernel 版本

不要根据输出文件名判断 kernel 版本。先看 Details 顶部的 launch header：

```bash
"$NCU_BIN" --import report.ncu-rep --page details | sed -n '1,30p'
```

NCU 会打印类似：

```text
void _flash_kda_fwd_recurrence<..., 16, 128, 3, 2, 160, ..., long, 64, 1>(...)
    (192, 16, 1)x(160, 1, 1), Cluster Size 2
```

对当前 vLLM FlashKDA K2，最有用的字段是 `NumThreads`、`VD` 和 `UseClusterMulticast`。再与 Details 中的 `Block Size`、`Cluster Size` 交叉验证：

| 实际版本 | `VD` | Block Size | NCU `Cluster Size` | Warp 角色 |
|---|---:|---:|---:|---|
| original | 128 | 192 | 0 | 4 MMA + LOAD + STORE |
| warp6 split | 64 | 192 | 0 | 4 MMA + LOAD + STORE |
| warp6 cluster | 64 | 192 | 2 | 4 MMA + LOAD + STORE |
| warp5 split | 64 | 160 | 0 | 4 MMA + fused LOAD/STORE |
| warp5 cluster | 64 | 160 | 2 | 4 MMA + fused LOAD/STORE |

这里的 160/192 是 CTA thread 数，不是寄存器数。寄存器必须另看 `Registers Per Thread` 和 `launch__registers_per_thread_allocated`。

如果采集命令使用 `--launch-count 3` 或 `--profile-variant all`，一份报告可能依次包含 original、split 和 cluster 三个 launch。GUI 中要先选择目标 kernel，CLI 中则要逐个核对 Details 顶部的 launch header，不能把第一个 kernel 的 Block Size 当成整份报告的共同配置。

一个真实反例是旧文件 `h100_b16_t8192_h96_split5.ncu-rep`。虽然文件名写着 `split5`，报告内部却是：

```text
Block Size 192
VD         64
Cluster    0
```

所以它实际是 warp6 split。真正的 warp5 split 报告是 block 160。出现这种情况通常是扩展没有重新安装、Volume 中仍是旧二进制，或输出文件名先于实现改动写好。

### 4.6 判断 cluster 报告处于同步修改前还是修改后

旧 cluster kernel 在每个 chunk 让四个 MMA warp 执行一次 CTA-wide `NamedBarrier`。其热点 SASS 通常是：

```text
BAR.SYNC.DEFER_BLOCKING 0x8, 0x80
MEMBAR.ALL.CTA
```

其中 `0x80` 对应 128 个 compute threads。查找方法：

```bash
"$NCU_BIN" \
  --import report.ncu-rep \
  --page source --print-source sass --csv \
  | rg -n 'BAR\.SYNC.*0x8|MEMBAR\.ALL\.CTA|SYNCS\.ARRIVE'
```

per-warp release 版本不再在该 chunk 位置执行 128-thread barrier，而是由四个 MMA warp 各自选出一个 lane，对 cluster empty barrier 发布 arrival。判断时必须结合 Source correlation 和执行次数，不能仅凭“报告里还有某条 `MEMBAR.ALL.CTA`”就认定旧同步仍存在，因为 kernel 初始化、state 转换和其他 CTA 同步也可能生成 `MEMBAR.ALL.CTA`。

当前仓库的报告已按实际内容归档到 [`ncu-reports/INDEX.md`](../ncu-reports/INDEX.md)。命名格式为：

```text
<GPU>__<实际 kernel 版本>__<shape>__<采集范围>.ncu-rep
```

其中 `pre-perwarp` 明确表示报告早于 per-warp release。归档时建议同时记录：

1. Git commit；
2. GPU 和 compute capability；
3. workload shape；
4. 实际捕获的 kernel 版本；
5. cache-control 与 replay 模式；
6. 报告包含 `full` 还是有限 sections。

## 5. Metric 名字怎样拆

例如：

```text
smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
dram__bytes.sum.per_second
```

可以拆成：

```text
硬件单元__被测量的量.聚合方式.单位或归一化方式
```

常见单元：

| 前缀 | 大致含义 |
|---|---|
| `sm__` | 整个 SM |
| `smsp__` | SM subpartition / warp scheduler 侧 |
| `l1tex__` | L1、texture、shared memory 数据路径 |
| `lts__` | L2 slice |
| `dram__` | 显存 |
| `gpu__` | GPU 全局派生指标 |
| `launch__` | kernel launch 属性 |

常见后缀：

| 后缀 | 含义 |
|---|---|
| `.sum` | 所有实例求和 |
| `.avg` | 所有实例平均 |
| `.max/.min` | 实例最大/最小值 |
| `.per_second` | 除以运行时间 |
| `.per_cycle_active` | 只对单元 active cycle 归一化 |
| `.per_cycle_elapsed` | 对整个 kernel elapsed cycle 归一化 |
| `.pct_of_peak_sustained_elapsed` | 相对持续峰值百分比 |

不要比较分母不同的百分比。`per_cycle_active` 很高但 `per_cycle_elapsed` 很低，可能只是该单元偶尔工作时很忙，大部分 kernel 时间都没工作。

查询一个 metric 的可用完整名称：

```bash
"$NCU_BIN" \
  --query-metrics-mode all \
  --query-metrics-collection profiling \
  | rg 'mio_throttle'
```

## 6. 固定的报告阅读顺序

每次都按下面顺序读，可以避免被某一个“醒目百分比”带偏。

### 6.1 LaunchStats：先确认工作规模

检查：

```text
Grid Size
Block Size
Registers Per Thread
Dynamic Shared Memory Per Block
Driver Shared Memory Per Block
Waves Per SM
```

问题：

- grid 是否足够填满 GPU。
- 每个 CTA 有多少 warp。
- 一个 SM 理论上能放几个 CTA。
- shared memory 和 register 是否把 residency 压到一个 CTA。

### 6.2 Occupancy：判断 TLP 上限

检查：

```text
Theoretical Occupancy
Achieved Occupancy
Block Limit Shared Mem
Block Limit Registers
Block Limit Warps
Block Limit Barriers
```

最终 resident CTA 数由最严格的限制决定。只减少 register，但如果 `Block Limit Shared Mem=1`，仍然只能放一个 CTA。

### 6.3 SpeedOfLight：只做一级分类

典型组合：

| Compute | Memory | 第一判断 |
|---:|---:|---|
| 高 | 低 | 可能 compute throughput bound |
| 低 | 高 | 可能 memory bandwidth bound |
| 高 | 高 | roofline 附近，需看具体 pipe |
| 低 | 低 | 多半是 latency、依赖、同步或工作量不足 |

Compute 和 Memory 同时低，不能解释为“两边都有优化空间”。更常见的是 warp 发不出指令，因此计算和内存请求都没有形成吞吐。

### 6.4 SchedulerStats：检查能不能发射

最关键关系是：

```text
Active warp：驻留且尚未结束
Eligible warp：下一条指令现在可以发射
Issued warp：本周期真的被选中发射
```

如果出现：

```text
Active Warps Per Scheduler      1.50
Eligible Warps Per Scheduler    0.09
No Eligible                    91.1%
Issue Slots Busy                7.6%
```

说明不是 scheduler 没有 resident warp，而是大多数 resident warp 都被依赖卡住了。

### 6.5 WarpStateStats：等待什么

| Stall | 准确定义方向 | 常见根因 |
|---|---|---|
| `long_scoreboard` | 等待 L1TEX 路径的数据依赖 | global/local load、cache miss |
| `short_scoreboard` | 等待非 L1TEX 的 MIO 数据依赖 | shared load、`LDSM`、特殊函数 |
| `mio_throttle` | MIO 指令队列已满，下一条 MIO 发不进去 | shared 指令太密、特殊函数、动态分支 |
| `lg_throttle` | local/global 指令队列已满 | global/local 指令过密 |
| `math_pipe_throttle` | 目标数学管线没有发射槽 | 某条 math pipe 饱和 |
| `wait` | 等待固定延迟执行依赖 | accumulator 链、类型转换、ALU 链 |
| `barrier` | CTA warp 等待 barrier | 同步过多、warp 负载不均 |
| `membar` | 等待内存顺序 | fence、memory barrier |
| `not_selected` | warp eligible，但本周期选了别的 warp | ready warp 足够，通常不是坏事 |
| `sleeping` | warp 处于 sleep/yield 等状态 | pipeline 等待、专用 producer/consumer warp |

两个容易混淆的概念：

- `short_scoreboard`：已经发出的 MIO 结果还没回来，消费者在等数据。
- `mio_throttle`：新的 MIO 指令还没有发出，因为 MIO queue 满了。

因此，prefetch 可能减少前者，却增加后者。

### 6.6 ComputeWorkloadAnalysis：确认是哪条 pipe

不要只看总 Compute Throughput。继续展开：

```text
Tensor pipe
FP32 pipe
FP16/BF16 pipe
Integer pipe
Uniform/Data movement pipe
```

如果 Tensor pipe 只有 20%，但 `No Eligible` 是 90%，问题不是 Tensor Core 太慢，而是没有持续向 Tensor Core 发出 MMA。

### 6.7 MemoryWorkloadAnalysis：确认是哪一层

依次看：

```text
Mem Busy
Mem Pipes Busy
L1/TEX Throughput
L2 Throughput / Hit Rate
DRAM Throughput
Load/Store sectors
Shared requests/wavefronts
```

DRAM 吞吐低不等于不是 memory latency bound。低 occupancy 或强依赖可能让并发请求太少，既承受了 latency，又永远打不满带宽。

## 7. Shared memory、Excessive 和 bank conflict 怎样区分

Shared Memory 表中最重要的字段是：

```text
Instructions
Requests
Wavefronts
Wavefronts Ideal
Wavefronts Excessive
Bank Conflicts
```

概念关系：

- 一条 shared SASS 指令形成 request。
- request 可能被拆成多个 wavefront 分周期处理。
- 宽向量和 matrix 指令本来就可能需要多个 wavefront。
- `Excessive = Wavefronts - Wavefronts Ideal`，主要描述活跃、未被
  predicated-off 的线程没有充分完成该宽度访问时产生的额外工作。
- 真正的 bank-conflict 数来自
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_*` 硬件计数器。

因此这是两个不同问题：

```text
Wavefronts / Ideal / Excessive
    → 检查 lane 参与度、predication 和 uncoalesced shared access

Bank Conflicts
    → 检查同一 shared request 的 bank 序列化
```

`L1 Conflicts Shared N-Way` 的 Source 公式是：

$$
\text{N-Way}=\frac{\text{Wavefronts}}{\text{Instructions Executed}}
$$

这个名字对普通 `LDS/STS` 比较直观，但对 `LDSM/STSM .4` 必须谨慎：
`.4` matrix 指令本来就会产生多个 wavefront，不能把 `N-Way=4` 直接解释
成 layout 导致的 4-way bank conflict。

### 7.1 本次 SM120 报告的真实例子

逐指令 Source 数据：

```text
Instruction                              N-Way  Excessive  Wavefronts / Ideal
STSM.16.M88.4  [R6+0xbb80], R40             4       0       131072 / 131072
LDSM.16.MT88.4 R8,  [R23+0x6880]            4       0       131072 / 131072
LDSM.16.MT88.4 R36, [R32]                   4       0       131072 / 131072
STSM.16.MT88.4 [R32], R36                   4       0       131072 / 131072
```

整个 kernel：

```text
memory_l1_wavefronts_shared          7421952
memory_l1_wavefronts_shared_ideal    7421952
derived excessive wavefronts               0

l1tex shared bank conflicts total     196622
l1tex shared-store bank conflicts     195700
l1tex shared-load bank conflicts         223
```

以上数值以当前保留的 `ncu-reports/flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep` 经 NCU 2026.1 重新导出的 raw
counter 为准。total 与 load/store 两项不满足简单加和，不能混用不同 counter
口径计算冲突倍数。

`Excessive=0` 只说明这些指令没有产生相对 ideal 的 uncoalesced shared
wavefront，不能据此证明没有 bank conflict。

与此同时，raw hardware counter 的 shared-store bank conflict 数非零。但本机
NCU 2025.4 的 `SharedMemoryConflicts.py` 在计算“平均 N-way store conflict”时，
store request 分母只加入普通 shared-store SASS 数，没有加入
`smsp__inst_executed_op_shared_stsm.sum`；wavefront 分子却包含 STSM。因此报告中
“84.3-way store conflict”的平均倍数不适用于这个大量使用 STSM 的 kernel。

本例目前能得出的严谨结论是：

1. 逐条 `LDSM/STSM` 没有 uncoalesced `Excessive` wavefront。
2. raw bank-conflict counter 给出了冲突信号，但 Source 的 `Excessive` 不能定位它。
3. NCU rule 给出的 84.3-way 平均值受 STSM request 计数口径影响，不可信。
4. 要定位冲突来源，需要隔离某组 shared store 做 A/B，或用 PM Sampling
   观察冲突随时间的位置，再用 microbenchmark 校准该 STSM 指令的基线。

## 8. 怎样开始看 SASS

不需要先熟练掌握 PTX。PTX 是虚拟 ISA，SASS 才是 GPU 实际执行的机器指令。做性能分析时，更高效的学习顺序是：

```text
先理解 CUDA/CuTe 的数据流
    ↓
用 NCU 把源码关联到 10～30 条 SASS
    ↓
只认出 load / store / MMA / ALU / barrier
    ↓
沿寄存器看 producer-consumer
    ↓
最后才用 PTX 解释编译器中间变换
```

### 8.1 先认识少量指令族

| SASS | 作用 |
|---|---|
| `LDG` | global memory load |
| `STG` | global memory store |
| `LDS` / `LDSM` | shared memory load；`LDSM` 常用于 MMA fragment |
| `STS` / `STSM` | shared memory store；`STSM` 常用于 MMA fragment |
| `HMMA` / `MMA` | Tensor Core matrix multiply-accumulate |
| `UTMALDG` / `UTMASTG` | TMA global-to-shared load / shared-to-global store |
| `FFMA` | FP32 fused multiply-add |
| `F2FP` | 浮点格式转换或打包 |
| `MOV` | 寄存器数据移动 |
| `BRA` / `BSSY` | 分支和控制流 |
| `BAR` / `MBAR` | barrier 或异步 pipeline 同步 |
| `SYNCS.PHASECHK` | 检查异步 transaction barrier 的 phase/完成状态 |

先能把代码分成这些类别，就足够做第一轮性能分析。

### 8.2 真实 K2 SASS 片段

下面是本次报告中 Phase 5 尾部和 Phase 6 开头的真实 SASS。地址和寄存器编号会随编译改变，但数据流含义不变。

```text
0x...a5c0  F2FP.BF16.F32.PACK_AB R26, R27, R26
0x...a5d0  F2FP.BF16.F32.PACK_AB R24, R25, R24
0x...a610  STSM.16.M88.4  [R6+0xbb80], R40

0x...a620  LDSM.16.MT88.4 R8,  [R23+0x6880]
0x...a630  LDSM.16.MT88.4 R36, [R32]
0x...a660  LDSM.16.MT88.4 R16, [R23+0x6a80]
0x...a670  LDSM.16.MT88.4 R40, [R32+0x800]

0x...a680  HMMA.16816.F32.BF16 R4, R8,  R20, RZ
0x...a710  HMMA.16816.F32.BF16 R8, R8,  R2,  RZ

0x...a740  FFMA.FTZ R4,  R12, R13, R4
0x...a750  FFMA.FTZ R6,  R24, R14, R6
0x...a790  FFMA.FTZ R8,  R12, R15, R8
0x...a7a0  FFMA.FTZ R10, R24, R25, R10

0x...a810  HMMA.16816.F32.BF16 R12, R16, R20, RZ
0x...a830  F2FP.BF16.F32.PACK_AB R38, R9,  R8
0x...a850  F2FP.BF16.F32.PACK_AB R39, R11, R10
0x...a860  F2FP.BF16.F32.PACK_AB R36, R5,  R4
0x...a870  LDSM.16.MT88.4 R8, [R23+0x6c80]
0x...a880  F2FP.BF16.F32.PACK_AB R37, R7,  R6
0x...a8d0  HMMA.16816.F32.BF16 R16, R16, R2, RZ
0x...a8e0  STSM.16.MT88.4 [R32], R36
0x...a910  LDSM.16.MT88.4 R36, [R32+0x1000]
```

### 8.3 一行一行怎样读

```text
LDSM.16.MT88.4 R8, [R23+0x6880]
```

- `LDSM`：shared memory 到寄存器 fragment。
- `[R23+0x6880]`：shared memory 地址。
- `R8`：目标寄存器组的起点。
- `MT88.4`：带转置语义的 matrix shared-load 变体和四寄存器形式。

```text
STSM.16.MT88.4 [R32], R36
```

- store 的内存地址写在前面。
- 把从 `R36` 开始的 fragment 写回 `[R32]` 对应的 shared tile。
- 这就是 Phase 6 更新后的 state 寄存器到 shared memory。

```text
HMMA.16816.F32.BF16 R4, R8, R20, RZ
```

- `16816` 表示这条 MMA 的逻辑 `M/N/K` 形状族。
- 输入是 BF16，累加结果是 FP32。
- `R4` 是输出 accumulator 寄存器组。
- `R8`、`R20` 是 A/B fragment 寄存器组。
- `RZ` 是零寄存器，说明这一条从零开始累加，而不是读取已有 C accumulator。

```text
FFMA.FTZ R4, R12, R13, R4
```

等价数据流为：

$$
R4 = R12\times R13 + R4
$$

它对应 state scale 与 MMA 结果组合附近的标量 FP32 FMA。

```text
F2FP.BF16.F32.PACK_AB R38, R9, R8
```

它把 FP32 结果转换并打包成 BF16。看到连续 `F2FP` 后跟 `STSM`，通常就是 accumulator epilogue 的寄存器转换与 r2s 写回。

### 8.4 用寄存器 def-use 看依赖

不必一开始知道所有修饰符，只做两件事：

1. 哪条指令写了某个寄存器。
2. 后面哪条指令读取它。

例如：

```text
LDSM ... R8, [shared]          # 定义 R8 fragment
HMMA ... R4, R8, R20, RZ      # 消费 R8，定义 R4 accumulator
FFMA ... R4, ..., ..., R4      # 读取并重写 R4
F2FP ... R36, ..., R4          # 消费 R4，生成 BF16 fragment
STSM ... [shared], R36         # 消费 R36，写回 shared
```

这条链越长，中间又没有另一条独立寄存器链可执行，`wait` 或 scoreboard latency 越难隐藏。

### 8.5 Source 页应该打开哪些列

在 GUI 的 Source 页面切到 SASS，并优先显示：

```text
Address
Instructions Executed
Predicated-On Threads Executed
Warp Stall Sampling (Not-issued Samples)
Address Space / Access Operation / Access Size
L1 Conflicts Shared N-Way
L1 Wavefronts Shared
L1 Wavefronts Shared Ideal
L1 Wavefronts Shared Excessive
```

阅读顺序：

1. 找执行次数最高的指令。
2. 找 stall sample 集中的 PC。
3. 向上查看它依赖的数据是谁产生的。
4. 对 shared 指令比较 actual、ideal、excessive。
5. 对比两个版本的指令顺序和执行次数，不只对比源码。

### 8.6 怎样系统增强 SASS 能力

推荐用下面的渐进练习，不要从完整 K2 的几千条 SASS 开始：

1. 写一个只有一次 global load/store 的 kernel，认出 `LDG/STG`。
2. 写一次 shared round trip，认出 `STS/LDS`。
3. 用现有 `ldmatrix_cycle_bench.cu`，改变一处 layout，只比较 `LDSM` 和 wavefront。
4. 写一个单独 MMA microbenchmark，认出输入 fragment、accumulator 和 `RZ`。
5. 给同一个 kernel 写两种调度顺序，在 Source 页面比较 SASS 是否真的移动。
6. 每次只截取一个源码语句附近 10～30 条 SASS，手工标记 `load → compute → store` 和寄存器 def-use。

纯反汇编命令也能看 SASS：

```bash
cuobjdump --dump-sass ./flash_kda_C*.so | less
```

但它没有 NCU 的执行次数、stall 和内存指标。性能学习阶段优先使用 NCU Source 页面，`cuobjdump` 只用于确认二进制里有哪些指令。

## 9. Prefetch、MIO 和 bank conflict 的关系

Prefetch 是调度优化，不是流量优化。它通常不减少 load 数，只是把 load-use 距离拉长。

当前 Phase 6 有 8 个 16-wide block。假设每块各加载一次 K 和 state：

```text
PREFETCH=1：初始加载 1 次 + 循环补载 7 次 = 8 次
PREFETCH=2：初始加载 2 次 + 循环补载 6 次 = 8 次
```

总 `LDSM` 数不变，但 `PREFETCH=2` 会：

- 更早把多条 `LDSM` 放进 MIO queue。
- 增加 fragment 的寄存器生存期。
- 可能增加寄存器移动和编译器调度约束。
- 在独立计算不足时，完全无法覆盖额外 load-use 距离。

所以：

```text
short_scoreboard 下降 + mio_throttle 不升
    → prefetch 可能有效

mio_throttle 上升 + 时间不变或变慢
    → prefetch 只是更早堵住 MIO queue
```

`mio_throttle` 高不等于 bank conflict。MIO queue 包含 shared 指令、部分特殊
函数和动态控制流相关指令。bank conflict 要看
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_*`，而
`actual wavefront > ideal` 检查的是另一类 uncoalesced/predication 额外工作。

## 10. FlashKDA V-split K2 实例

本次被测实例：

```text
CHUNK=16
D=128
VD=64
InputStages=3
OutputStages=2
Threads=192
Grid=(2, 1, 32)，共 64 CTA
```

代表性指标：

```text
Compute/Tensor pipeline utilization       22.23%
Memory throughput                       132.78 GB/s
Mem Busy                                 35.82%
Max Bandwidth                            33.79%
Mem Pipes Busy                            8.69%
L2 Hit Rate                              77.36%
Issue Slots Busy                          7.40%
Executed IPC elapsed                      0.29
No Eligible                              91.55%
Active Warps / Scheduler                  1.51
Eligible Warps / Scheduler                0.09
Registers / Thread                       58，硬件按 64 分配
Shared Memory / CTA                      69.632 KB（含 driver 部分）
Shared-memory CTA limit                   1
```

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

注意，K2 使用四个 MMA warp、一个 TMA load warp 和一个 TMA store warp。专用 warp 的正常等待会抬高 aggregate `sleeping`，不能把 50.1% 直接解释为可消除的 wall-clock 时间。表中未列出 `selected` 和占比较小的其他 state，因此占比之和不为 100%。

### 10.1 诊断过程

第一步，Compute 和 DRAM 都不高，排除“峰值 Tensor Core bound”和“峰值 DRAM bandwidth bound”。

第二步，`No Eligible=91.55%`，说明 scheduler 绝大多数周期找不到能发射的 warp。

第三步，每个 scheduler 只有 1.5 个 active warp，而且 shared memory 只允许一个 CTA/SM，跨 CTA latency hiding 很弱。

第四步，MIO throttle 较高，但它本身不能证明存在 bank conflict。逐条
`LDSM/STSM` 的 excessive wavefront 为零，只能排除这部分 uncoalesced shared
额外工作；raw bank-conflict counter 仍需通过隔离实验定位。

第五步，Phase 6 每轮只有少量 MMA，却紧邻多次 `LDSM`、`STSM`、FP32 FMA 和 BF16 pack，MIO 密度和寄存器依赖链都很高。

最终结论：

$$
\text{低 TLP/ILP}
+\text{紧密的 MIO/MMA 依赖}
\Longrightarrow
\text{scheduler issue starvation}
$$

这解释了为什么 Compute 和 Memory 吞吐可以同时低，也解释了为什么增加 prefetch 没有效果。

### 10.2 H100 实例：Long Scoreboard 很高，但 DRAM 没有打满

这一节分析 `ncu-reports/flashkda/h100__original-vsplit__b8-t2048-h96__full.ncu-rep` 中的 V-split K2。该报告
对应尚未把 state 全程常驻寄存器的版本，不能用后续源码中的寄存器常驻实现解释
这里的指标。

报告里的实际模板和 launch 配置是：

```text
CHUNK=16
D=128
VD=64
InputStages=2
OutputStages=2
Threads=192
Grid=(2, 8, 96)，共 1536 CTA
Registers / Thread Allocated=80
Shared Memory / CTA=53.632 KB
```

先核对模板参数非常重要。当前工作区中的 stage 数可能已经改变，但 `.ncu-rep`
记录的是采集时二进制里的模板实例，分析时应以报告中的 demangled kernel name 和
LaunchStats 为准。

#### 10.2.1 `cycles per issue` 到底是什么意思

该报告的 Warp State 结果是：

| 指标 | original | V-split |
|---|---:|---:|
| Warp Cycles Per Issued Instruction | 约 8.1 | 约 17.1 |
| Long Scoreboard Cycles Per Issue | 3.00 | 9.32 |
| Long Scoreboard 占 inter-issue 周期 | 37.1% | 54.5% |

`9.32 cycles per issue` 不表示某次 L2 load 的延迟是 9.32 cycle，也不表示整个
SM 每发一条指令就停 9.32 cycle。它表示：对 warp 的执行过程做平均，每成功发射
一条指令，累计有 9.32 个 warp-cycle 因 Long Scoreboard 而无法发射下一条指令。

例如某个等价执行过程最终发射 100 条指令，`9.32 cycles per issue` 可以理解为
大约累计 932 个 Long Scoreboard warp-cycle。一次具体的 TMA transaction 仍可能
有数百 cycle 延迟；这里是所有 warp、循环和已发射指令归一化后的平均值。

对应占比为：

$$
\frac{3.00}{8.1}\approx37.1\%
$$

$$
\frac{9.32}{17.1}\approx54.5\%
$$

因此，V-split 实例不是每条内存指令只慢了 9.32 cycle，而是 warp 的整体
inter-issue 间隔中超过一半被归类为 Long Scoreboard。

#### 10.2.2 先在 Source/SASS 中确认等待位置

H100 V-split 的 Long Scoreboard not-issued samples 共 71420 个。其中 55991 个，
即 78.4%，集中在下面的等待循环：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT ...
@!P0 BRA ...
```

源码对应：

```cpp
load_pipeline.consumer_wait(load_read);
```

以及 CUTLASS pipeline 内部：

```cpp
full_barrier_ptr_[stage].wait(phase);
```

采样落在 `BRA` 上不表示分支本身很慢。`SYNCS.PHASECHK` 先检查 transaction
barrier 的 phase，紧随其后的条件分支根据检查结果决定是否继续等待。这里的实际
含义是：MMA consumer 已到达下一 input stage，但该 stage 的 TMA transaction
还没有完成。

CLI 中可以直接寻找这组指令：

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/h100__original-vsplit__b8-t2048-h96__full.ncu-rep \
  --page source \
  --print-source sass \
  | rg -C 8 'SYNCS.PHASECHK|NANOSLEEP|UTMALDG'
```

注意，stall sampling 一般归因给正在等待的 consumer PC。定位根因时应继续向上
追产生依赖的 TMA load 和对应的 barrier，而不能只修改被标红的 `BRA`。

#### 10.2.3 Cache line、request 和 sector

H100 的 L1/L2 cache line 是 128B，一条 cache line 又被分成四个 32B sector：

```text
地址偏移     0          32         64         96        128
             | sector 0 | sector 1 | sector 2 | sector 3 |
             |------------- 128B cache line --------------|
```

三者的统计含义不同：

- `tag` 唯一标识一条 128B cache line。
- `request` 是发给 L2 的命令；一个 L2 request 访问同一条 cache line 内的
  1 到 4 个 sector。
- `sector` 是实际请求、传输和 hit-rate 统计的 32B 单位。

例如只访问一条 cache line 的 `[32, 64)`，统计为 1 request、1 sector；访问
`[16, 80)` 则是 1 request、3 sectors；一次访问跨越两条 cache line 时至少需要
两个 request/tag lookup。

Sector hit 还要求 tag 和请求的 sector data 都存在。tag 不存在，或者 tag 存在但
请求的 sector data 不存在，都会被统计为 sector miss。

本次 V-split 报告中：

```text
L2 requests = 215.976 M
L2 sectors  = 231.725 M
```

所以：

$$
\text{Sectors/Request}
=\frac{231.725}{215.976}
\approx1.073
$$

即每个 L2 request 平均涉及约 1.07 个 32B sector。这个数本身不能直接证明
TMA access 没有合并，因为 TMA 会把 tile 分解为硬件请求；它只能证明 L2 的
request 数远大于每次 request 涉及的 sector 数。

#### 10.2.4 顶层 L2 Throughput 不等于 L2 data bandwidth

NCU 的 throughput 是复合指标。顶层 throughput 会由多个底层计数构成，并以
最繁忙的组成部分代表整个单元。因此必须在 Raw 页面展开 L2 子指标。

该报告中的 H100 V-split 数据是：

| L2/DRAM 子指标 | `% of peak sustained elapsed` | 含义 |
|---|---:|---|
| `lts__t_tag_requests` | 93.58% | L2 tag/request 查询路径 |
| `lts__xbar2lts_cycles_active` | 64.63% | XBAR 向 L2 输入请求或数据 |
| `lts__t_sectors` | 33.47% | tag 侧处理的 requested sectors |
| `lts__d_sectors` | 36.13% | L2 data 侧处理的 sectors |
| `lts__lts2xbar_cycles_active` | 35.33% | L2 向 XBAR/SM 返回数据 |
| `lts__d_sectors_fill_device` | 26.06% | 从 device memory miss-fill 的 sectors |
| `gpu__dram_throughput` | 40.64% | L2 与 HBM 之间的实际显存吞吐 |

所以 GUI 中接近 100% 的 `L2 Throughput`，实际由
`lts__t_tag_requests=93.58%` 决定。它不能表述成“L2 data 字节带宽已经达到
94%”。L2 data sectors 和 L2-to-XBAR 返回路径只有约 35%。

可以用 Raw 页面快速核对这些 metric：

```bash
"$NCU_BIN" \
  --import ncu-reports/flashkda/h100__original-vsplit__b8-t2048-h96__full.ncu-rep \
  --page raw \
  | rg 'lts__throughput|lts__t_tag_requests|lts__t_sectors|lts__d_sectors|xbar2lts|lts2xbar|gpu__dram_throughput'
```

数据路径应理解为：

```text
TMA/global request
        |
        v
L2 tag lookup                     <- hit 和 miss 都必须经过
        |
        +-- hit ---------------------------+
        |                                  |
        +-- miss -> HBM -> L2 fill --------+
                                           v
                                  TMA 写入 shared
                                           |
                                  mbarrier transaction 完成
                                           |
                                  MMA consumer 继续执行
```

所有访问 HBM 的 GPU 数据都要经过 L2，但反方向不成立：L2 的工作还包括 hit、
tag lookup、write、fill 和内部数据传输。因此 L2 与 DRAM 的百分比不是同一个
分母，也不存在 `L2%=DRAM%` 的守恒关系。

#### 10.2.5 为什么 Long Scoreboard 高而 DRAM 只有 40.6%

Long Scoreboard 描述依赖没有及时就绪，不表示 DRAM 字节带宽已经打满。低 DRAM
吞吐与高 Long Scoreboard 可以同时出现，常见原因包括：

1. outstanding miss 不够多，延迟存在但 memory-level parallelism 不足以打满 HBM。
2. 请求先受上游 request/tag 处理速率限制，HBM 没有收到足够的连续字节流量。
3. 部分请求命中 L2，消耗 L2 工作但不产生 HBM 流量。
4. 多个 consumer warp 等待同一个 TMA stage barrier，active warp 很多但依赖不独立。

本报告尤其符合第 2 和第 4 点：

| Scheduler 指标 | original | V-split |
|---|---:|---:|
| Active Warps / Scheduler | 2.96 | 5.71 |
| Eligible Warps / Scheduler | 0.488 | 0.511 |
| Issue Active | 36.60% | 33.34% |

V-split 的 active warp 几乎翻倍，但 eligible warp 基本不变。增加的 resident warp
没有变成可发射的独立工作，而是大量停在 TMA stage 依赖上。

同时，L2 request 数与 runtime 的变化高度一致：

| 指标 | original | V-split | 变化 |
|---|---:|---:|---:|
| L2 requests | 147.9 M | 216.0 M | +46.0% |
| L2 sectors | 161.8 M | 231.7 M | +43.2% |
| NCU kernel duration | 1.167 ms | 1.687 ms | +44.6% |

这组数据支持的判断是：该 H100 launch 的 L2 request/tag 处理已接近持续峰值；
新增请求使总处理时间增加，最终在 TMA consumer barrier 上暴露为 Long
Scoreboard。它不支持“HBM bandwidth 已满”的判断。

#### 10.2.6 L1 GUI 指标怎样读

L1TEX 同时承载 L1 data cache、texture 和 shared-memory 数据路径。顶层
`L1 Throughput` 可能由 shared `LDSM/STSM`、local/global cache access 或其他
L1TEX 子路径决定，不能直接解释成 global-load L1 cache bandwidth。

GUI 中常见行的含义是：

| GUI 行 | 应怎样理解 |
|---|---|
| `L1 Throughput` | L1TEX 多个子单元组成的顶层 throughput |
| `L1 Throughput (Data)` | L1TEX data path 的活动程度 |
| `L1 Sectors` | 普通 cache sector 请求速率 |
| `L1 Hit Rate` | 被请求 sector 的命中率，不是吞吐率 |
| `L1 Lookup Throughput` | L1 tag lookup 速率 |

“hit rate 高、throughput 低”只表示请求不多，但其中较高比例命中；它不是
“L1 容量太小”的表现。容量不足通常首先表现为 hit rate 下降。TMA 的
global-to-shared 数据路径也不能只靠普通 global LSU load 的 L1 hit-rate 行来解释。

#### 10.2.7 这份 H100 报告能和不能得出什么

能得到的结论：

1. Long Scoreboard 的主要采样位置是 input TMA transaction barrier。
2. 顶层 L2 Throughput 由 tag/request 路径决定，而非 L2 data 字节带宽。
3. DRAM 没有打满不排除 memory dependency latency；本例中 HBM 不是峰值带宽瓶颈。
4. 单纯增加 occupancy 没有增加 eligible warp，多个 warp 的等待具有相关性。

不能得到的结论：

1. 不能把 H100 与 RTX 5060 Ti 的 stall 差异归因于单一架构因素。现有两份报告
   使用了不同 grid、stage 和 shared-memory 配置。
2. 不能根据 `L2 Throughput=93.58%` 声称 L2 data bandwidth 达到 93.58%。
3. 不能把 PC sampling 标红的 `BRA` 当成根因；它只是 transaction wait 的
   consumer/control-flow 位置。
4. 不能用尚未进入该二进制的 state 寄存器常驻修改解释这份报告。

若要比较 GPU 架构，至少需要两组受控实验：一组保持完全相同的
`B/T/H/VD/InputStages`，另一组按 SM 数调整 grid、保持相近的 CTA/SM 和 wave 数。
否则“瓶颈迁移”中会同时混入硬件、并发规模和 kernel 配置三个变量。

## 11. Replay 为什么会改变你观察到的运行

很多硬件计数器不能在一个 pass 同时采集。NCU 会多次执行被测 kernel，并合并结果。

常见 replay 模式包括：

- kernel replay：重复单个 kernel。
- application replay：重复整个应用并匹配 kernel invocation。
- range replay：重复标记范围。

Replay 会带来三个陷阱：

1. profiler 下程序自己测出的时间不可信。
2. cache 初始状态可能与真实 K1→K2 连续执行不同。
3. kernel 如果读取或修改状态，NCU 需要保存和恢复相关内存，采集成本可能很高。

研究单 kernel 指令效率时，默认 kernel replay 通常够用。研究跨 kernel cache reuse 或 producer-consumer 行为时，再考虑 application/range replay 和 NVTX。

不要为了得到更漂亮的 cache 数字随意改 `--cache-control`。首先要明确问题是冷缓存单 kernel，还是生产流水中的热缓存行为。

## 12. 去读官方文档时怎样定位

官方 CLI 文档按“选项”组织，Profiling Guide 按“硬件指标”组织。遇到问题时这样找：

| 你想知道什么 | 官方文档关键词 |
|---|---|
| kernel 怎么过滤 | `Kernel Filtering`, `kernel-name`, `launch-skip` |
| 为什么采集很多遍 | `Replay`, `Kernel Replay`, `Application Replay` |
| metric 名怎么拆 | `Metrics Structure`, `Metric Suffixes` |
| throughput 百分比是什么 | `Throughput Metrics`, `pct_of_peak_sustained_elapsed` |
| warp 为什么不发射 | `Warp Scheduler States`, `Warp Stall Reasons` |
| MIO throttle 是什么 | `mio_throttle_not_issued` |
| shared 表怎么读 | `Memory Workload Analysis`, `Requests`, `Wavefronts` |
| Source 页有什么列 | `Source Counters`, `Source Page` |
| section 里收集什么 | `Section Sets`, `Section Files`, `--list-metrics` |

阅读一条官方 metric 定义时，固定问五个问题：

1. 统计对象是 instruction、thread、warp、cycle、sector 还是 byte？
2. 聚合范围是 `.sum`、`.avg` 还是每个硬件实例？
3. 分母是 active cycle 还是 elapsed cycle？
4. 这是精确硬件计数，还是 PC sampling？
5. 这个值是根因、症状，还是 scheduler 状态？

只要先回答这五个问题，官方文档里的长 metric 名就不会显得混乱。

## 13. 一页命令速查

```bash
# 列出采集集合和 section
"$NCU_BIN" --list-sets
"$NCU_BIN" --list-sections

# 查看某 section 会采哪些 metric
"$NCU_BIN" --section SchedulerStats --list-metrics

# 最小探针
"$NCU_BIN" --section LaunchStats -k 'regex:.*kernel_name.*' \
  --launch-count 1 -o probe -- command

# 总览
"$NCU_BIN" --section LaunchStats --section Occupancy \
  --section SpeedOfLight -k 'regex:.*kernel_name.*' \
  --launch-count 1 -o overview -- command

# Scheduler 与 stall
"$NCU_BIN" --section SchedulerStats --section WarpStateStats \
  --section ComputeWorkloadAnalysis -k 'regex:.*kernel_name.*' \
  --launch-count 1 -o scheduler -- command

# 内存
"$NCU_BIN" --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  -k 'regex:.*kernel_name.*' --launch-count 1 -o memory -- command

# Source/SASS
"$NCU_BIN" --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables --import-source yes \
  -k 'regex:.*kernel_name.*' --launch-count 1 -o source -- command

# 最终完整报告
"$NCU_BIN" --set full -k 'regex:.*kernel_name.*' \
  --launch-count 1 -o full -- command

# 读取报告
"$NCU_BIN" --import full.ncu-rep --page session
"$NCU_BIN" --import full.ncu-rep --page details
"$NCU_BIN" --import full.ncu-rep --page raw
"$NCU_BIN" --import full.ncu-rep --page source --print-source sass
```

最后记住这条诊断顺序：

```text
先用 Session 确认机器、命令和 shape
→ 再确认实际 kernel 版本和 launch
→ 看 occupancy 与工作规模
→ 用 SOL 做一级分类
→ 吞吐都低就看 scheduler
→ No Eligible 高就看 stall
→ stall 追到 Source/SASS
→ shared 的 uncoalesced 与 bank-conflict 指标必须分开看
→ 最后用无 profiler benchmark 验证收益
```
