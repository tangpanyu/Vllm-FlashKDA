# K2 V-split Cluster Multicast 设计与使用

> 来源标注（2026-10-29）：此文保留早期 FlashKDA 的版本语境，相关源码见 [旧版快照](../experiments/flashkda-v1/csrc/)。根目录的 vLLM 实现及实验边界见 [文档导航](README.md)。

本文说明 `fwd_kernel2_1.cuh` 的双 CTA cluster 版本如何工作、代码改在何处，以及如何验证正确性和性能。

## 1. 目标与适用范围

V-split K2 把 $V=128$ 切成两个 $VD=64$ 的 CTA：

| cluster rank | `blockIdx.x` / `v_idx` | 私有 V/O/state 范围 |
|---:|---:|---:|
| 0 | 0 | $[0,64)$ |
| 1 | 1 | $[64,128)$ |

两个 CTA 的 V、O 和 state 不同，但每个 chunk 使用完全相同的以下数据：

$$
\beta,\quad K_{decay},\quad Q_{decay},\quad K_{right},\quad
G,\quad INV,\quad Mqk
$$

普通 V-split 会让两个 CTA 分别从 global/L2 加载一遍这些公共输入。cluster 版本使用 TMA multicast：公共 tile 只读取一次，同时写入两个 CTA 各自的 shared memory。

当前实现固定要求：

- SM90 及以上；H100 的 `sm_90a` 是主要性能目标；
- $D=128$、$VD=64$；
- cluster shape 为 `(2, 1, 1)`；
- `blockIdx.x` 必须是 `v_idx`，两个相邻 x CTA 构成一个 cluster。

这是实验路径，默认不开启。普通 K2 和普通 V-split 都继续保留，便于做性能 A/B 对比。

## 2. 每个 chunk 共享了多少输入

每个 cluster 的公共输入字节数为：

| 输入 | 每个 chunk 字节数 |
|---|---:|
| beta 对齐 tile | $32\times2=64$ |
| K-decayed | $16\times128\times2=4096$ |
| Q-decayed | $16\times128\times2=4096$ |
| K-restored | $16\times128\times2=4096$ |
| g-total | $128\times4=512$ |
| INV | $16\times16\times2=512$ |
| Mqk | $16\times16\times2=512$ |
| 合计 | $13888$ |

每个 CTA 私有的 V tile 为：

$$
16\times64\times2=2048\ \text{bytes}
$$

因此，忽略 state load 和输出写回，仅看 K2 每个 chunk 的输入读取：

$$
\begin{aligned}
\text{普通 V-split} &= 2\times(13888+2048)=31872,\\
\text{cluster multicast} &= 13888+2\times2048=17984.
\end{aligned}
$$

理论上减少约 $43.6\%$ 的这部分 global/L2 读取。shared-memory 写入量不会减半，因为两个 CTA 最终都必须拥有一份完整公共 tile。

## 3. 为什么由 rank 0 发完整 multicast

这里没有把公共 tile 拆成“rank 0 加一半、rank 1 加一半”。因为两个 CTA 请求的是完全相同的 global 坐标，最简单也最少指令的方式是：

1. 两个 CTA 分别 `producer_acquire()` 自己当前 stage 的 mbarrier；
2. rank 0 加载完整公共 tile，multicast mask 为 `0b11`；
3. TMA 把同一次读取写入 rank 0 和 rank 1 的 shared memory，并完成两个 CTA 对应的 mbarrier transaction bytes；
4. rank 1 只发自己的 V tile TMA；
5. 两个 CTA 都等待“公共 tile + 私有 V”全部到达后开始 MMA。

核心封装位于 `issue_common_tma_tile()`：

```cpp
if constexpr (UseClusterMulticast) {
    constexpr uint16_t kClusterMask = 0x3;
    if (cluster_rank == 0) {
        auto cta_tma = tma_load.get_slice(Int<0>{});
        cute::copy(tma_load.with(barrier, kClusterMask),
                   cta_tma.partition_S(gmem_tile),
                   cta_tma.partition_D(smem_tile));
    }
}
```

TMA multicast 并不是让两个 CTA 直接使用同一块 shared memory。每个 CTA 仍有自己的 shared-memory 地址空间；硬件只是把一次 global 读取广播到两个目标 shared-memory 地址。

## 4. Descriptor 与 pipeline 的改动

### 4.1 公共输入 descriptor

普通 V-split 继续使用 `SM90_TMA_LOAD`。cluster 版本为公共输入额外创建 `SM90_TMA_LOAD_MULTICAST` descriptor：

```cpp
auto tma_load_ws_kd_vsplit_cluster = make_tma_copy(
    SM90_TMA_LOAD_MULTICAST{}, m_ws_kd, K2VSplitTMAMMALayout{});
```

descriptor 保持一个逻辑 issuing slice。目的 CTA 集合不是 descriptor 的 tile 分块，而是在 kernel 发指令时通过 `kClusterMask = 0x3` 指定。

V 不使用 multicast，因为两个 rank 的 `v_idx` 不同。initial/final state 和 O 同样保持 CTA 私有。

### 4.2 cluster-aware load pipeline

普通版本的 load pipeline 使用 `Shape<_1,_1>`，cluster 版本使用 `Shape<_2,_1>`：

```cpp
using LoadClusterShape = std::conditional_t<
    UseClusterMulticast, Shape<_2, _1>, Shape<_1, _1>>;
```

这不改变 shared-memory stage 的容量。它改变的是 `PipelineTmaAsync` 的 empty-barrier 到达拓扑：consumer release 必须正确通知同一行/列上的 cluster CTA，防止某一 rank 提前复用仍被另一 rank 访问的 stage。

每个 CTA 的 barrier 期望字节数仍然是：

$$
13888+2048=15936\ \text{bytes}
$$

对 rank 1 而言，其中 $13888$ bytes 由 rank 0 的远端 multicast 完成，$2048$ bytes 由自己的 V TMA 完成。

pipeline barrier 初始化后增加了：

```cpp
__syncthreads();
cute::cluster_sync();
```

这样 rank 0 首次远端写 mbarrier 之前，rank 1 已完成 barrier 初始化。kernel 退出前也执行 cluster sync，防止 peer 仍可能访问 cluster shared memory 时某个 CTA 提前退出。

## 5. Cluster launch

cluster 不能使用普通的 `kernel<<<...>>>` 启动。launch 侧使用 CUTLASS cluster launch API：

```cpp
dim3 grid_k2(2, N, H);
dim3 block_k2(192);
dim3 cluster_k2(2, 1, 1);

cutlass::launch_kernel_on_cluster(
    {grid_k2, block_k2, cluster_k2, smem_size_k2, stream},
    kernel_ptr,
    /* kernel arguments */);
```

`grid_k2.x=2` 是 CTA 数，不是 cluster 数。由于 `cluster_k2.x=2`，每个 `(seq_idx, head_idx)` 对应一个双 CTA cluster。

## 6. API

Python 包装层新增独立开关：

```python
flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log, dt_bias, lower_bound,
    initial_state=initial_state,
    final_state=final_state,
    cu_seqlens=cu_seqlens,
    use_vsplit_cluster=True,
)
```

`use_vsplit_cluster=True` 会自动启用 V-split。三个可比较路径为：

| 路径 | 参数 |
|---|---|
| 原始 K2 | `use_vsplit=False` |
| 普通 V-split | `use_vsplit=True` |
| cluster V-split | `use_vsplit_cluster=True` |

低于 SM90 的 GPU 会直接报告清晰的参数错误，不会进入 device-side assert。SM120 使用项目内的直接 PTX 兼容实现，因为当前 CUTLASS 会主动禁用 SM120 TMA multicast。`ptxas` 会警告该指令在 SM120 上可能显著变慢；本地 RTX 5060 Ti 只适合验证正确性，不应据此评估 cluster 路径性能。

## 7. 修改文件

| 文件 | 改动 |
|---|---|
| `csrc/smxx/fwd_kernel2_1.cuh` | cluster 模板开关、公共 TMA multicast、cluster pipeline 和同步 |
| `csrc/smxx/utils.cuh` | `make_load_pipeline()` 接受 cluster shape |
| `csrc/smxx/tma_multicast_compat.cuh` | SM120 直接 PTX multicast 兼容层；其他架构复用 CUTLASS 实现 |
| `csrc/smxx/fwd_launch.cu` | multicast descriptor 和 cluster launch |
| `csrc/fwd.h`、`csrc/flash_kda.cpp` | C++ API、SM90+ 检查和 dispatch |
| `flash_kda/__init__.py` | Python API 开关 |
| `tests/test_vsplit_cluster.py` | SM90+ 上的无状态、BF16 state、FP32 varlen state 正确性测试 |
| `tests/bench_vsplit.py` | original / split / cluster 三路 benchmark 和单路径 NCU 模式 |

## 8. H100 编译与测试

H100 必须编译 `sm_90a`，不要在 H100 机器上复用为其他架构编出的 `.so`：

```bash
FLASH_KDA_CUDA_ARCHS=90a NVCC_THREADS=1 MAX_JOBS=1 \
python -m pip install -v -e . --no-build-isolation
```

这里建议先用单编译线程，因为普通 V-split 和 cluster V-split 都会实例化多组 state/varlen 模板，并行 NVCC 容易造成较高的主机内存峰值。

运行 cluster 专项正确性测试：

```bash
pytest tests/test_vsplit_cluster.py -q -s
```

三路性能对比：

```bash
python tests/bench_vsplit.py \
  --cases '1,256,1;1,2048,1;1,2048,4;1,2048,32;4,2048,32' \
  --warmup 6 --repeats 5
```

测试输出包含 `original_ms`、`split_ms`、`cluster_ms` 和两种 speedup。

## 9. Nsight Compute 采集

`bench_vsplit.py --profile` 会跳过正确性比较和其他 variant，只启动指定路径。这样 `--set full` 不会把钱和时间花在无关 kernel 的 replay 上。

先取得真实 Python 可执行文件，不要把 shell alias 传给 NCU：

```bash
PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
```

采普通 V-split：

```bash
sudo -E "$(command -v ncu)" \
  --set full \
  --target-processes all \
  --kernel-name-base function \
  -k 'regex:.*flash_kda_vsplit.*recurrence.*' \
  --force-overwrite \
  -o h100_k2_split_full \
  "$PYTHON_BIN" tests/bench_vsplit.py \
    --profile split --cases '4,2048,32' \
    --warmup 0 --repeats 1 --inner 1
```

采 cluster V-split 时只改两处：

```bash
sudo -E "$(command -v ncu)" \
  --set full \
  --target-processes all \
  --kernel-name-base function \
  -k 'regex:.*flash_kda_vsplit.*recurrence.*' \
  --force-overwrite \
  -o h100_k2_cluster_full \
  "$PYTHON_BIN" tests/bench_vsplit.py \
    --profile cluster --cases '4,2048,32' \
    --warmup 0 --repeats 1 --inner 1
```

对比报告时重点看：

- K2 duration；
- L2 sector/request 数量和 L2 throughput；
- DRAM bytes/throughput；
- TMA 指令数；
- `long_scoreboard`；
- cluster 约束后的 achieved occupancy、active warps 和 waves per SM。

理想现象是公共输入相关的 L2 request/sector 明显下降。若请求下降但 duration 上升，通常说明 cluster 的共调度约束、远端 shared-memory multicast 或较少的可驻留 cluster 抵消了流量收益。不要只凭 DRAM throughput 判断：这次优化首先减少的是重复 L2 请求，公共数据本来就可能被第二个 CTA 从 L2 命中。
