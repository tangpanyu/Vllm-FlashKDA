# K2 V-split 与 CTA Cluster

本文说明实验性的 K2 V-split 和 CTA cluster 路径，包括它们与 upstream/dev、PR #4 的关系，数据与同步如何组织，以及如何在 B300 上验证是否值得启用。

围绕 `VD=64` resident state 转置、寄存器生命周期和 spill 所做但最终未采用的优化，已合并到 [K2 V-split、Cluster 与寄存器实验复盘](k2-vsplit-experiment-retrospective.md) 中。

## 设计基线

这项改动直接泛化 upstream/dev 中已经合入 PR #4 的 K2，而不是复制旧版 kernel 再单独维护一个 `fwd_kernel2_1.cuh`。

因此，三条 K2 路径共享以下 PR #4 优化：

- K1 将 shared-memory 中已经 swizzle 的物理字节图直接写入 workspace；K2 用 raw bulk copy 将相同字节图恢复到相同 shared-memory layout。
- recurrent state 在整个 chunk 循环中常驻寄存器，只在入口加载一次，并在需要 `final_state` 时于末尾写回。
- 保留 PR #4 调整过的 MMA 发射顺序、`PREFETCH=1`、late output acquire 和 varlen 反向调度；V-split specialization 使用显式寄存器上限控制资源占用。
- 原始 `VD=128` specialization 仍是默认路径；V-split 是同一个 kernel 模板的 `VD=64` specialization。

这里的 raw-byte workspace 是 K1/K2 之间的私有 ABI。`k_decayed`、`q_decayed`、`k_restored`、`g_total`、`INV` 和 `Mqk` 不能在 K2 中当作普通逻辑矩阵重新排布，否则会破坏 K1 已经生成的 swizzle 字节图。

## 三条执行路径

| 路径 | API | K2 CTA 数量 | workspace 加载 |
|---|---|---:|---|
| 原始 K2 | 两个 flag 都为 `False` | `H * N` | 每个 CTA raw bulk load |
| plain V-split | `use_vsplit=True` | `2 * H * N` | 两个 V-slice CTA 各自重复 raw bulk load |
| clustered V-split | `use_vsplit_cluster=True` | `2 * H * N`，每两个 CTA 构成一个 cluster | rank 0 将 raw workspace 字节图 multicast 给两个 CTA |

`use_vsplit_cluster=True` 会隐含 `use_vsplit=True`。默认值保持原始 K2，不会自动改变已有调用的性能特征。

## `VD=64` 的索引与工作划分

当前只支持 $D=128$，V-split 取 $VD=64$，所以每个 `(sequence, head)` 由两个 CTA 共同完成：

```cpp
constexpr int kVSlices = D / VD;  // 2
int linear   = int(blockIdx.x);
int v_idx    = linear % kVSlices;
int head_idx = linear / kVSlices;
int seq_idx  = int(blockIdx.y);
```

launch grid 为：

```cpp
dim3 grid_k2(H * (D / VD), N);
```

因此 `blockIdx.x` 中相邻的两个 CTA 分别处理同一个 head 的 `v_idx=0` 和 `v_idx=1`。cluster 路径使用 `clusterDim=(2,1,1)`，恰好把这两个 CTA 配成一组。

varlen 路径仍保留 upstream 的 sequence 反向映射：vLLM 通常将短 decode 放在前面、长 prefill 放在后面，K2 反转 `blockIdx.y`，让长 CTA 尽早发射。V-split 不改变这一规则。

每个 V-slice 的数据范围如下：

- `V` 和 `O`：从 `[H,T,D]` 的最后一维偏移 `v_idx * VD`，tile 为 `[CHUNK,VD]`。
- state：外部语义是 `[N,H,V,K]`，从 V 维偏移 `v_idx * VD`，每个 CTA 持有 `[VD,D]`。
- `K_gamma`、`Q_gamma`、`K_right`、`G`、`INV` 和 `Mqk`：不依赖 V-slice，仍保持完整 D 或 CHUNK 维度。

四个 MMA warp 平分一个 slice。原始 `VD=128` 时每个 warp 处理两个 16-column block；`VD=64` 时每个 warp 只处理一个 16-column block。state 的寄存器驻留、output accumulator、U fragment 和最终 state spill 都随这个 block 数量一起缩减。

## plain V-split 的加载

plain split 不使用 distributed shared memory。两个 CTA 完全独立：

1. 每个 CTA 用自己的 TMA descriptor 加载 64 列 `V`。
2. 每个 CTA各自加载 `beta`。`beta` 虽然相同，但数据量很小，并且保留本地加载能让 cluster 路径只处理 raw workspace。
3. 每个 CTA都从 global workspace raw bulk load 完整的六块公共数据。
4. 每个 CTA只加载、更新并写回自己的 `[VD,D]` state。
5. 每个 CTA只写出自己的 64 列 `O`；tail tile 的标量写回同样带有 `v_idx * VD` 偏移。

这条路径的收益来自 CTA 数量翻倍和单 CTA shared memory/state 寄存器占用降低；代价是公共 workspace 流量重复，并且单 CTA 的 MMA N 维从 128 降到 64。

## clustered V-split 的 raw workspace multicast

cluster 路径仍让两个 CTA各自加载 `V`、`beta` 和自己的 state。只有以下六块 V-independent workspace 由 cluster rank 0 发起一次加载：

```text
k_decayed, q_decayed, k_restored, g_total, INV, Mqk
```

这里使用非 tensor 的 raw `cp.async.bulk` multicast 内联 PTX，将 workspace 中的物理字节原样送到 cluster 两个 CTA 中相同 shared-memory offset：

```text
cp.async.bulk.shared::cluster.global
    .mbarrier::complete_tx::bytes.multicast::cluster
```

multicast mask 为 `0x3`，对应 cluster rank 0 和 1。rank 1 不重复发出这六条 global load，但远端 multicast 会完成 rank 1 对应 stage 的 transaction barrier。

使用 raw byte copy 而不是根据矩阵坐标重建 tensor map，是为了严格保留 PR #4 的 workspace ABI：global workspace 中已经是 K1 shared-memory layout 的物理字节图，K2 需要把它逐字节恢复到同一种 layout。

## Pipeline 与 barrier

K2 当前使用 3 个 input stage 和 2 个 output stage。cluster 路径的 load pipeline 使用 `Shape<_2,_1>`，plain/original 路径使用 `Shape<_1,_1>`。

cluster 同步需要满足以下约束：

- 两个 CTA 都初始化自己 shared memory 中相同 offset 的 full/empty barrier，然后在第一笔远端 multicast 前执行 cluster sync。
- FP32 initial state 临时占用的 `state_fp32_buf` 与 input/output stages 位于同一个 union。每个 CTA 完成本地 FP32→BF16 转换后还要再执行一次 cluster sync，rank 0 才能开始向 rank 1 的 input stage multicast；只有 `__syncthreads()` 会留下跨 CTA 覆盖窗口。
- 每个 stage 的 transaction byte count 仍包含本地 `V`、本地 `beta` 和完整六块 workspace。rank 1 虽然不发 workspace load，但这些字节由 rank 0 的 multicast 到达，不能从 expected transaction bytes 中减掉。
- 两个 CTA 必须执行完全相同次数的 `producer_acquire`、pipeline state advance、consumer wait/release 和 `producer_tail`。只有六条 workspace load 的发射受 `cluster_rank==0` 控制。
- `Shape<_2,_1>` 让 input stage 的 empty barrier 等待两个 CTA 的消费者释放，防止一个 CTA 尚未读完时另一个 CTA 提前复用该 stage。
- cluster pipeline 把每个 CTA 的 128-thread consumer warpgroup 压缩成一次代理 arrival，但 K2 实际使用的是四个彼此独立的 `mma.sync` warp。每个 chunk 在 `consumer_release()` 前必须先经过 128-thread `NamedBarrier`；否则被选中的代理线程可能先释放 stage，而另外的 warp 仍在执行 `ldmatrix`/MMA。非 cluster 路径由 128 个线程各自 arrive，不额外加入这个 barrier。
- initial/final state 的 TMA barrier 保持 CTA-local；每个 CTA只处理自己的 `[VD,D]` state。
- kernel 退出前再次执行 cluster sync，避免一个 CTA退出后，另一个 CTA仍可能访问它的 distributed shared memory 或远端 barrier。

output pipeline 不跨 CTA。每个 V-slice 独立写出，不需要两个 CTA 合并 accumulator。

### Cluster 的五 warp fused-IO 路径

plain V-split 保留六个 warp：四个 MMA warp、一个 input TMA producer warp 和一个 output TMA consumer warp。cluster V-split 改用五个 warp：四个 MMA warp不变，最后一个 warp同时承担 input producer 和 output consumer。该 warp 只有 elected lane 发射 TMA，所以合并的目的不是增加数据搬运线程，而是去掉一个只做控制的 warp，并把 cluster specialization 的寄存器上限从 112 放宽到 128。

fused-IO warp 先预取 `InputStages` 个 input tile，然后对每个 chunk 执行：

```text
等待 O[t] ready
发射 input[t + InputStages]
发射 O[t] 的 TMA store
释放 output stage
```

等待 `O[t]` ready 已经意味着 MMA warpgroup 消费完 `input[t]`，因此此时可以安全复用相应 input stage。下一块 input 在当前 output 写回之前发射，使两条异步搬运有重叠机会。

`load_pipeline.producer_tail()` 必须放在所有 output stage 排空之后。若 fused warp像独立 load warp 一样先执行 producer tail，它可能等待某个 input empty barrier；与此同时 MMA warp又可能因 output ring 已满而等待 fused warp写回，形成由同一个 warp负责解除的循环等待。

没有把五 warp 路径用于 plain split。全量 H100 实验中，plain split 改为五 warp 后，在 mixed decode/prefill、`H=96` 的三组 case 上平均退化约 13.1%，其中前两组退化约 16.5%。FP32 state 的 shared-memory 转换会把 8×8 atom 分摊给 CTA 内全部线程，192 减到 160 线程会增加每个 warp承担的转换工作；短 decode CTA 对这个固定成本尤其敏感。cluster 路径没有出现这组退化，因此最终只对 cluster 启用 fused IO。

H100 `VD=64` cluster 关键 specialization 编译为 118 registers/thread、0 spill；shared memory 仍约为 68 KiB，未因合并 IO warp 增加。旧六 warp NCU 报告显示实际 shared-memory config 为 167.936 KiB/SM，每个 3-stage CTA 的 allocated shared memory 为 69.632 KiB，因此 shared memory 将该配置限制为 2 CTA/SM；减少一个 warp本身没有提高 resident CTA 数。相对六 warp cluster 基线，按每次 run 内 `original / cluster` speedup 归一化后，12 个 workload 的平均变化约为 +2.8%。提升集中在 CTA 数足够多的纯 prefill：`pure_n2_t4096_h96` 为 +22.8%，`pure_n4_t2048_h96` 为 +6.1%；其余 workload 大多在约 -0.6% 到 +2.8% 之间。最终 12 组 output 和 final state 均与 original bitwise exact。

还单独测试了把 V-split input pipeline 从 3 stage 降到 2 stage。2-stage cluster 仍编译为 118 registers/thread、0 spill；由 layout 大小可得 dynamic shared memory 从 68.608 KiB 降到约 52.608 KiB，加上旧 NCU 中每 CTA 的 1.024 KiB driver allocation 后，167.936 KiB config 在资源上允许 3 CTA/SM。寄存器按 120 registers/thread 的分配粒度计算，三个 160-thread CTA需要 57,600 registers，也能放入 H100 的 65,536 registers/SM。因此该实验很可能发生的是 `2 -> 3 CTA/SM`，此前用设备物理 shared-memory 上限推导成 `3 -> 4` 是错误的；还需要用 2-stage NCU 的 launch metrics确认实际 residency。

尽管潜在 occupancy 提升，五 warp 2-stage 在 12 个 workload 中全部退化，归一化平均下降约 2.5%；plain split 平均下降约 6.2%，部分 specialization 还产生了新的 spill。这里不能简单归因于 stage 减少：当前 fused-IO warp 在每轮发射一个新 input 后执行 `tma_store_wait<0>()`，等待当前 O 的 TMA store完成才能进入下一轮。六 warp版本的独立 load warp不受 store warp这段等待影响，可以在 input stage释放后立即继续 refill。合并 warp因此在 input refill 与前一块 output store之间新增了串行边；3-stage 有更多缓冲，2-stage 更容易暴露。最终暂时恢复 3 input stages，但要判断降低 stage 本身是否有利，必须补做“2-stage + 六 warp cluster”对照，不能把本次五 warp结果当成纯 stage 实验。

最终 H100 测量命令：

```bash
VLLM_FLASHKDA_GPU='H100!' modal run modal_vllm_flashkda.py \
  --action bench --mode cases \
  --warmup 30 --iterations 200 --repeats 6 \
  --l2-scrub-mib 256
```

对应 Modal run：`ap-Uxb5oUy8OGG3G4L0kRouoz`。

### 同步错误、偶发 hang 与确定性死锁

这里需要区分数据竞态和真正的 barrier 死锁。早期 cluster 实现缺少部分同步时，代码不是每次都会死锁，而是进入未定义行为：可能得到错误结果、触发非法内存访问，也可能偶发 hang。当前实现中的同步分别保护以下四个生命周期：

1. 第一笔 multicast 之前的 cluster sync 保证两个 CTA 都已经初始化本地 full/empty barrier。否则 rank 0 可能向 rank 1 尚未初始化的远端 barrier 完成 transaction。
2. FP32 initial state 转换之后的 cluster sync 保证两个 CTA 都不再读取与 pipeline stages 共用 union 的 `state_fp32_buf`。否则 rank 0 的 multicast 可能覆盖 rank 1 尚在读取的 state scratch。这首先是数据竞态，不是必然死锁。
3. 每个 chunk 的 128-thread `NamedBarrier` 保证四个 MMA warp 都完成当前 input stage 的 `ldmatrix`/MMA 后，代理线程才能执行 `consumer_release()`。缺少它会造成 stage 提前复用和数据竞态；通常表现为 mismatch，也可能进一步导致非法访问或 hang。
4. kernel 退出前的 cluster sync 保证一个 CTA 不会在 peer 仍可能访问其 distributed shared memory 或远端 barrier 时结束生命周期。缺少它同样属于未定义行为，可能表现为间歇性 hang。

真正可重复的 cluster 死锁通常来自同步协议不对称：两个 CTA 执行了不同次数或不同 phase 的 `cluster_sync()`、barrier arrive/wait、`producer_acquire()`、`consumer_release()` 或 `producer_tail()`。一侧等待的 arrival 永远不会由另一侧产生，因此无法推进。为避免这种情况，cluster rank 条件只能控制六条 workspace load 的实际发射，不能让两个 CTA 跳过不同的 collective、pipeline state advance 或 barrier 操作。当前两个 V-slice CTA 使用相同的 sequence、tile 和 chunk 数，保证同步控制流一致。

这些同步也会产生性能成本。长序列下，启动、state 转换和退出处的 cluster sync 会被摊薄；每个 chunk 都执行的 128-thread `NamedBarrier` 则会随 chunk 数增长，通常更值得在 profile 中重点检查。不能仅为消除等待直接删除它，除非用另一套能够证明四个 MMA warp 已全部结束当前 stage 读取的正确协议替代。

## Stable API

Python stable API 增加了两个可选 flag：

```python
flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log=A_log,
    dt_bias=dt_bias,
    lower_bound=-5.0,
    initial_state=initial_state,
    final_state=final_state,
    cu_seqlens=cu_seqlens,
    workspace=workspace,
    use_vsplit=False,
    use_vsplit_cluster=False,
)
```

选择方式：

```python
# upstream 原始 K2
use_vsplit = False
use_vsplit_cluster = False

# 两个独立 VD=64 CTA
use_vsplit = True
use_vsplit_cluster = False

# 两个 VD=64 CTA组成 cluster；Python wrapper 自动打开 use_vsplit
use_vsplit_cluster = True
```

两个 flag 也存在于 `torch.ops.flash_kda.fwd` 的 stable schema 中，默认均为 `False`。workspace 大小和格式没有变化，三条路径可以复用同一块 workspace。

## 在 Modal B300 上安装与验证

Modal app 使用独立的 `vllm-flashkda-build` Volume 保存 editable install 和 `flash_kda/_C*.so`。install action 不申请 GPU，显式为 B300 编译 `sm_103a`：

```bash
modal run modal_vllm_flashkda.py --action install
```

编译环境设置为：

```text
CUDA_HOME=/usr/local/cuda-13.2
FLASH_KDA_CUDA_ARCHS=103a
NVCC_THREADS=16
MAX_JOBS=16
```

只有 `setup.py`、`flash_kda/`、`csrc/` 或 `cutlass/include/` 的内容变化时才会重新编译。只修改 tests 或 benchmarks 不会触发 NVCC。

运行 focused exactness 测试：

```bash
modal run modal_vllm_flashkda.py --action test
```

同时运行完整 forward correctness matrix：

```bash
modal run modal_vllm_flashkda.py --action test --full
```

先做低成本 smoke benchmark：

```bash
modal run modal_vllm_flashkda.py --action bench \
  --mode smoke --warmup 2 --iterations 5 --repeats 2
```

运行全部 PR #4 workload：

```bash
modal run modal_vllm_flashkda.py --action bench \
  --mode cases --warmup 30 --iterations 200 --repeats 6 \
  --l2-scrub-mib 256
```

只比较指定 case：

```bash
modal run modal_vllm_flashkda.py --action bench \
  --mode cases \
  --cases 'pure_n1_t8192_h12;mixed_d31_p8161_h96' \
  --warmup 30 --iterations 200 --repeats 6
```

benchmark 会依次执行 original、plain split 和 cluster，并先检查 `O` 与 FP32 final state 是否 bitwise exact。

长序列或疑似竞态可以用 eager 诊断器重复执行。它会先确认 plain split control，再按 token、chunk、head 和 V-slice 汇总 cluster mismatch：

```bash
modal run modal_vllm_flashkda.py --action debug \
  --debug-lengths '64,512,2048,8192' \
  --debug-heads 12 --repeats 10
```

### NCU

先完成 install，再启动使用 bench function 配置的 B300 shell；它会挂载 build、benchmark 和 report Volume：

```bash
modal shell modal_vllm_flashkda.py::bench
```

在远端 shell 中只发射一个 case 的三种 K2 variant，并只抓 recurrence kernel：

```bash
cd /workspace/Vllm-FlashKDA

/usr/local/cuda-13.2/bin/ncu \
  --set full \
  --kernel-name-base function \
  -k 'regex:.*_flash_kda_fwd_recurrence.*' \
  --launch-count 3 \
  --clock-control none \
  --import-source yes \
  --source-folders /workspace/Vllm-FlashKDA \
  --force-overwrite \
  -o /workspace/reports/b300_vsplit_all \
  /usr/local/bin/python /workspace/benchmarks/bench_vsplit.py \
    --mode profile \
    --case mixed_d31_p8161_h96 \
    --profile-variant all \
    --profile-launches 1
```

`--profile-variant all` 的 K2 顺序是 original、split、cluster，所以 `--launch-count 3` 会把三者都采集到同一个报告。若只分析一种路径，将 `--profile-variant` 改为对应名称，并把 `--launch-count` 改为 1。

如果云实例没有开放 NVIDIA performance counters，NCU 会报 `ERR_NVGPUCTRPERM`；这不是 kernel 错误，需要换到开放 counters 的 B300 环境。不要用普通 timing benchmark 的 warmup/repeat 驱动 NCU，否则会重复采集大量 kernel 并显著增加运行时间。

## PR #4 workload

`benchmarks/bench_vsplit.py` 将总 token 数固定为 8192，覆盖两个 head 规模：12 和 96。

纯 prefill/fixed cases：

```text
pure_n1_t8192_h12   pure_n1_t8192_h96
pure_n2_t4096_h12   pure_n2_t4096_h96
pure_n4_t2048_h12   pure_n4_t2048_h96
```

vLLM 风格 mixed-varlen cases 让短 decode 位于前面，最后一个 sequence 是长 prefill：

```text
mixed_d7_p8185_h12    mixed_d7_p8185_h96
mixed_d15_p8177_h12   mixed_d15_p8177_h96
mixed_d31_p8161_h12   mixed_d31_p8161_h96
```

这些 case 用 int32 `cu_seqlens`、FP32 initial/final state，并复用同一块 workspace，贴近 PR #4 关注的 vLLM 调度形态。

## B300 实测结果

2026-08-04 在 `NVIDIA B300 SXM6 AC`（148 SM、CUDA 13.2、PyTorch 2.13.0+cu132）上完成了以下验证：

- focused correctness matrix：22 个用例全部通过，original、plain split 和 cluster 均与 `torch_ref` bitwise exact。覆盖 fixed/varlen、int32/int64 `cu_seqlens`、空 sequence、无 state、BF16/FP32 state 和全部 state in/out 组合。
- stage reuse 回归：$T=64$ 的 4 个 chunk 强制复用 3-stage circular buffer，bitwise exact。
- 长序列压力测试：$T=64,512,2048,8192$，每个长度重复 10 次；plain split 和 cluster 的 output/final state 每次都与 original bitwise exact，且 cluster 各次执行之间没有漂移。
- 正式 benchmark：每条路径独立 CUDA Graph，30 次 warmup、200 次 iteration、6 个交错顺序 repeat；下表给出 original latency 除以 variant latency，数值大于 1 才是加速。

| PR #4 case | cold split | cold cluster | steady split | steady cluster |
|---|---:|---:|---:|---:|
| `pure_n1_t8192_h12` | 1.261x | 1.261x | 1.265x | 1.264x |
| `pure_n2_t4096_h12` | 1.238x | 1.238x | 1.244x | 1.244x |
| `pure_n4_t2048_h12` | 1.217x | 1.202x | 1.204x | 1.189x |
| `mixed_d7_p8185_h12` | 1.268x | 1.257x | 1.266x | 1.257x |
| `mixed_d15_p8177_h12` | 1.263x | 1.258x | 1.260x | 1.255x |
| `mixed_d31_p8161_h12` | 1.264x | 1.253x | 1.263x | 1.251x |
| `pure_n1_t8192_h96` | 0.949x | 0.912x | 0.948x | 0.911x |
| `pure_n2_t4096_h96` | 0.835x | 0.791x | 0.836x | 0.791x |
| `pure_n4_t2048_h96` | 0.883x | 0.838x | 0.884x | 0.839x |
| `mixed_d7_p8185_h96` | 0.966x | 0.921x | 0.966x | 0.920x |
| `mixed_d15_p8177_h96` | 0.965x | 0.922x | 0.964x | 0.921x |
| `mixed_d31_p8161_h96` | 0.964x | 0.915x | 0.963x | 0.913x |

结论不是“V-split 全局更快”，而是边界非常稳定：H=12 的六组 workload 中 plain split 提升 20%–27%，H=96 时回退 3%–17%；cold-L2 与 steady-state 的判断一致。cluster 在低并行度时大致追平 plain split，但没有额外收益，在 H=96 时还会因成对调度和 barrier 开销进一步回退。因此当前两条路径都保持显式 opt-in，不能替换默认 K2；实际集成应优先考虑 plain split，并只对 TP 后 head 较少、长 prefill 主导的 shape 启用。

## 何时值得启用

V-split 主要解决 K2 CTA 数量不足，而不是提高单 CTA 的计算密度。是否启用必须以目标 workload 的端到端 forward latency 为准。

plain split 更可能在以下场景获益：

- `N * H` 很小，原始 K2 无法给 B300 的 SM 提供足够 CTA。
- 少数长 prefill CTA 主导尾部延迟，增加 V 方向并行度比重复公共 workspace 流量更有价值。
- 缩小 state/shared-memory footprint 后，实际 resident CTA 数或尾部利用率改善。

cluster 更可能在以下场景优于 plain split：

- split 确实需要额外 CTA，但重复读取六块公共 workspace 已经成为明显成本。
- 两个 V-slice 的运行时间接近，cluster 成对调度不会造成严重负载不均。
- raw multicast 节省的 global/L2 请求足以覆盖 cluster placement、distributed barrier 和同步开销。

应保留原始 K2 的场景：

- `N * H` 已经足以填满设备，CTA 翻倍不再改善利用率。
- `H=96`、较大 batch 或大量 sequence 已提供足够并行度。
- NCU 显示瓶颈来自 MMA/issue latency，而不是 CTA 不足或公共 workspace traffic。
- cluster 的成对驻留限制降低了调度自由度，或 cluster latency 不低于原始路径。

推荐决策顺序：

1. 先确认 original、split、cluster 在目标 shape 上 bitwise exact。
2. 用 Modal benchmark 的 median latency 比较三条完整 forward 路径，而不是只看 K2 单次偶然值。
3. 对有收益和无收益的代表 case 各采一份 NCU，检查 CTA/SM 分布、active warps、long scoreboard、L2/DRAM traffic 和 shared-memory 指标。
4. 只有当目标 vLLM case 稳定获益时才显式打开 flag；在建立可靠的 shape-based dispatch policy 前，不应全局默认启用。

对 B300 可以用“尾部仍然很长的 CTA 数”作为保守 hint，而不是直接用表面的 $N\times H$：

$$
C_{\mathrm{tail}} = H\cdot
\#\left\{i:\left\lceil\frac{L_i}{16}\right\rceil
\geq \frac{1}{2}\max_j\left\lceil\frac{L_j}{16}\right\rceil\right\}.
$$

当前 12 个 case 支持的经验条件是 `max_len >= 2048` 且 $C_{\mathrm{tail}}\leq49$（约为 148 SM 的三分之一）时选择 plain split。pure H=12 的 $C_{\mathrm{tail}}$ 为 12/24/48，mixed H=12 虽然有许多 decode sequence，但它们一个 chunk 后就退出，尾部仍只有 12 个长 CTA；两类都满足条件。H=96 则不满足。真正接入 vLLM 时应使用调度器已有的 CPU sequence metadata 传递这个 hint，不能为了判断路径同步读取 GPU 上的 `cu_seqlens`。
