# 当前分支相对 FlashKDA 的优化记录

> 迁移标注（2026-10-29）：本文的“当前分支”固定指 `feature/k2-vsplit@3fcf41b`。迁移时实际工作树为 `experiment/vsplit-final@0c8a2d0` 加未提交修改；本文保留原版本边界，不作为该实验工作树的逐行说明。

> 目的：记录 `feature/k2-vsplit` 当前实现相对原始 FlashKDA 的变化、代码位置、收益来源和隐含约束，避免以后只记得“更快了”，却忘了为什么。
>
> 文档定位：这是判断当前代码行为的事实来源。文档总览见 [文档导航](README.md)；未合并的 cluster、5-warp、bankfix 和 state transpose 实验统一收录在 [K2 V-split 实验复盘](k2-vsplit-experiment-retrospective.md)。

## 0. 对比边界

- 当前版本：`feature/k2-vsplit`，commit `3fcf41b`（`fix(cuda): auto-dispatch K2 V-split`）。
- FlashKDA 基线：commit `fa7eb89`（`relax beta constraint (#2)`）。
- 中间的重要提交：
  - `a3e42bb`：集成 Flash-FlashKDA，包含 K1 缩线程、K1/K2 raw bulk workspace、K2 state 驻留寄存器等核心优化。
  - `1fb1d9c`：加入 K2 V-split。
  - `3fcf41b`：把 V-split 从显式 opt-in 改成自动 dispatch。
- `b5d1101` 是 PyTorch stable ABI 迁移，属于兼容性/集成改造，不算 CUDA kernel 性能优化，本文不展开。

本文以当前固定配置为主：

```cpp
constexpr int CHUNK = 16;
constexpr int D = 128;
constexpr int VD = 64;       // V-split 路径
constexpr int kK1Threads = 128;
constexpr int kK2Threads = 32 * 2 + 128;  // load/store 各 1 warp，MMA 4 warps
```

主要源码入口：

- `csrc/smxx/fwd_kernel1.cuh`：K1 prepare。
- `csrc/smxx/fwd_kernel2.cuh`：K2 recurrence。
- `csrc/smxx/fwd_launch.cu`：TensorMap 构造、workspace 切分、kernel launch。
- `csrc/flash_kda.cpp`：V-split 自动选择。
- `tests/test_vsplit.py`：V-split 路径测试。

## 1. 一页结论

当前分支不是只把 K1 从 256 线程缩成 128 线程，而是做了三组相互配合的改造：

| 区域 | 原始 FlashKDA | 当前分支 | 主要节省 |
|---|---|---|---|
| K1 CTA | 256 threads / 8 warps | 128 threads / 4 warps，以 virtual warp 做两遍 | CTA 线程、warp 调度和固定开销 |
| Q/K normalize | 归一化后整块写回 shared | shared 保留 raw Q/K，只保存每行 inverse norm，decay 时寄存器融合 | 约 8 KiB/CTA 的 Q/K shared 写回及一次同步 |
| 标量函数 | `expf(A_log)`、beta sigmoid 重复算 | CTA/row 级缓存到 shared | 特殊函数指令 |
| K1→K2 workspace | 6 个 tensor TMA store + 6 个 tensor TMA load | 6 个 raw bulk S2G + 6 个 raw bulk G2S | TensorMap 数量、坐标/布局处理和 TMA 发射前端开销 |
| K2 state | 每个 chunk 都经 shared 读取、更新、写回 | 初始时装入寄存器，循环中常驻，结束时才落回 shared | 每 chunk 的大块 shared state 往返和 CTA barrier |
| output pipeline | chunk 开头 acquire output stage | 第一次写 output 前才 acquire | 让等待与 Phase 1--4 重叠 |
| 小 batch/head 并行度 | 每个 `(sequence, head)` 一个 K2 CTA | V 维切成两个 `VD=64` CTA | 低并行度场景 CTA 数翻倍，资源占用降低 |

其中最关键的两个设计点是：

1. K1 与 K2 的 workspace 不再是“逻辑 row-major tensor”，而是 K1 shared-memory 物理字节图像的私有 ABI。
2. K2 的递推 state 不再以 shared memory 为每轮真值来源，寄存器中的 `resident_state` 才是 chunk 循环内的真值。

## 2. 整体数据流变化

原始版本可抽象为：

```text
K1 shared tensor
    -> tensor TMA store（按 TensorMap 做布局/坐标解释）
    -> global workspace 的逻辑 tensor
    -> tensor TMA load（再次按 TensorMap 解释）
    -> K2 shared tensor

K2 每个 chunk：shared state -> MMA registers -> 更新 -> shared state
```

当前版本变为：

```text
K1 shared 的物理 byte image
    -> raw bulk S2G
    -> global workspace 中相同 byte image
    -> raw bulk G2S
    -> K2 相同 shared layout 的物理 byte image

K2：initial state -> shared -> resident_state registers
    -> 所有 chunk 都在 registers 中递推
    -> final state（需要时）-> shared -> global
```

raw bulk 没有让 workspace 数据量消失；它省掉的是对这六块中间数据使用 tensor TMA 时的描述符、坐标和布局处理。真正减少大量片上数据搬运的是 K2 state 驻留寄存器。

## 3. K1：prepare kernel 的变化

### 3.1 256 threads 缩成 128 threads

代码锚点：

```cpp
// csrc/smxx/fwd_launch.cu
constexpr int kK1Threads = 128;

// csrc/smxx/fwd_kernel1.cuh
static_assert(NumThreads == 128);
```

原始 K1 是 256 threads，即 8 个物理 warp。当前 K1 是 128 threads，即 4 个物理 warp。

原来 `decay_apply` 的工作划分隐含了 8 个 warp，当前实现没有修改它的逻辑覆盖，而是引入 virtual warp：

```cpp
constexpr int PHYSICAL_WARPS = NumThreads / 32;  // 4
constexpr int WARP_PASSES = 8 / PHYSICAL_WARPS; // 2

for (int pass = 0; pass < WARP_PASSES; ++pass) {
    int virtual_warp_id = warp_id + pass * PHYSICAL_WARPS;
    // 原 8-warp 工作中的第 virtual_warp_id 份
}
```

所以准确说法是：**4 个物理 warp 串行覆盖原来 8 个逻辑 warp 的任务**，不是删掉一半计算。

这里 shared storage 使用了 union：输入 `q/k/g` 和输出 `k_decayed/q_decayed/k_inv/...` 会复用同一段 shared memory。因此当前代码必须先把两个 pass 的所有输入读进寄存器，再 `__syncthreads()`，最后才允许覆盖 union 中的输入：

```cpp
float reg_g[WARP_PASSES][N_TILES][2];
BF16  reg_q[WARP_PASSES][N_TILES][2];
BF16  reg_k[WARP_PASSES][N_TILES][2];
float reg_gt[WARP_PASSES][N_TILES][2];

// pass 0/1: shared -> registers
...
__syncthreads();
// pass 0/1: registers -> union'd shared outputs
```

这是缩线程后最容易被误改坏的依赖：如果某个 pass 尚未读完 raw Q/K，就提前写 `k_decayed/q_decayed`，会覆盖另一个 virtual warp 尚未消费的数据。

### 3.2 Q/K normalization 不再整块写回 shared

原始路径：

```text
shared raw Q/K
  -> 每行求 norm
  -> normalized Q/K 整块写回 shared
  -> sync
  -> decay_apply 再读 normalized Q/K
```

当前路径：

```text
shared raw Q/K
  -> 16-byte BF16x8 load 求每行 norm
  -> 每行只保存 q_norm_inv/k_norm_inv 两个 float
  -> decay_apply 读取 raw Q/K，在寄存器里乘 inverse norm
  -> 直接生成 BF16 decay 结果
```

对应代码：

```cpp
using BF16x8 = cutlass::AlignedArray<BF16, 8, 16>;

shared_storage.q_norm_inv.begin()[my_row] = rsqrtf(q_sq + 1e-6f);
shared_storage.k_norm_inv.begin()[my_row] = rsqrtf(k_sq + 1e-6f);
```

以及 decay 阶段：

```cpp
float q_inv = shared_storage.q_norm_inv.begin()[row];
float k_inv_scale = shared_storage.k_norm_inv.begin()[row];

BF16 q = BF16(bf16_to_f32(raw_q) * q_inv);
BF16 k = row < actual_len
    ? BF16(bf16_to_f32(raw_k) * k_inv_scale)
    : BF16(0);
```

在 `CHUNK=16, D=128, BF16=2 bytes` 下，省掉的 normalized Q/K shared 写回是：

$$
2 \times 16 \times 128 \times 2 = 8192\ \text{bytes/CTA}
$$

新增的 inverse norm shared 写入只有：

$$
2 \times 16 \times 4 = 128\ \text{bytes/CTA}
$$

此外还去掉了 normalize 写回之后专用的一次同步。这里的收益是 shared traffic 和同步减少，不是少做归一化数学。

### 3.3 beta sigmoid 与 `expf(A_log)` 去重

当前 shared storage 新增：

```cpp
float beta_act[16];
float q_norm_inv[16];
float k_norm_inv[16];
float a_log_exp;
```

`A_log` 对同一个 `(tile, head)` 是标量，当前只让 thread 0 计算一次：

```cpp
if (threadIdx.x == 0) {
    shared_storage.a_log_exp = expf(A_log_ptr[head_idx]);
}
```

beta 对每个 token row 也只有一个激活值，当前前 16 个线程各计算一次并缓存：

```cpp
if (compute_tid < CHUNK) {
    beta_act[compute_tid] = sigmoid_tanh_approx_f32(beta[... + compute_tid]);
}
```

后面的下三角矩阵构造直接复用 `beta_act[i]`。原始实现会在每个下三角元素处重新算该行 beta 的 sigmoid；当前从“按矩阵元素重复”降为“每行一次”。

### 3.4 tail K 的清零融合进 decay

原始 256-thread 版本可以让额外线程显式清零 tail K。缩成 128 threads 后，当前代码不再单独做这一步，而是在生成 decayed/restored K 时直接选择：

```cpp
BF16 k = row < actual_len ? normalized_k : BF16(0);
```

这避免了一个独立的 shared 清零阶段，也保证最后一个不足 16 token 的 tile 不会把越界 K 参与计算。

### 3.5 16x16 小矩阵改为 grid-stride 覆盖

`L/INV/Mqk` 的 16x16 元素共 256 个。原始 256 threads 可以一线程一个元素；当前 128 threads 使用：

```cpp
for (int matrix_idx = compute_tid;
     matrix_idx < CHUNK * CHUNK;
     matrix_idx += NumThreads) {
    ...
}
```

即每线程最多处理两个元素。数学和矩阵覆盖不变。

### 3.6 K1 workspace store：tensor TMA 改 raw bulk

原始 K1 kernel 模板参数里携带六个 workspace TMA store 描述符：

```text
KD, QD, KR, GT, INV, Mqk
```

每一块都要构造 global tensor、计算 `ws_off`、partition source/destination，再 `cute::copy(tma_store_...)`。

当前 kernel 只接收六个裸 workspace 指针，然后逐块复制连续物理字节：

```cpp
cute::SM90_BULK_COPY_S2G::copy(
    shared_storage.k_decayed.begin(),
    ws_kd + int64_t(ws_idx) * (CHUNK * D),
    int32_t(CHUNK * D * sizeof(BF16)));
tma_store_arrive();
```

其余五块完全同理。

当前版本还去掉了 K1 尾部的：

```cpp
tma_store_wait<0>();
__syncthreads();
```

这依赖 bulk async copy 的完成语义足以保证目标 global workspace 后续可见；K1 不再为了复用 source shared memory 而等待，因为 kernel 随即结束。

## 4. raw bulk workspace 的真实语义

### 4.1 数据量并没有减少

对于 `CHUNK=16, D=128`，每个 `(head, tile)` 的 workspace 是：

| 数据 | 大小 |
|---|---:|
| `KD/QD/KR` | $3 \times 16 \times 128 \times 2 = 12288$ B |
| `GT` | $128 \times 4 = 512$ B |
| `INV/Mqk` | $2 \times 16 \times 16 \times 2 = 1024$ B |
| 合计 | **13824 B** |

K1 仍写 13824 B，K2 仍读 13824 B。因此 raw bulk 的主要价值不是省 DRAM 字节数，而是：

- K1 少 6 个 workspace tensor-store 描述符。
- K2 少 6 个 workspace tensor-load 描述符。
- 少做 TensorMap 坐标计算、切片与 layout 映射。
- raw bulk 指令直接处理已知连续字节数，减少每条不同 tensor TMA 的准备/发射前端成本。

不要把它描述为“合成了一条 TMA”：源码中仍然是 6 条 S2G 和 6 条 G2S，只是每条从 tensor copy 换成 bulk copy。

### 4.2 workspace 现在是 shared byte image，不是逻辑矩阵

K1 的输出 tensor 使用：

```cpp
Tensor k_decayed = make_tensor(
    make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
```

线程通过逻辑坐标 `(row, col)` 写它时，CUTE 的 `MMALayout` 会把逻辑坐标映射成 shared memory 的物理地址。raw bulk 随后不理解 `(row, col)`，只是把这段物理字节原样搬到 global workspace。

K2 再 raw bulk 到使用同一 layout 的 shared 数组，因此物理图像被恢复。launch 处用编译期断言保护这份私有 ABI：

```cpp
static_assert(std::is_same_v<K1L::MMALayout, K2L::MMALayout>);
static_assert(std::is_same_v<K1L::GTotalLayout, K2L::GTotalLayout>);
static_assert(std::is_same_v<K1L::LMLayout, K2L::LMLayout>);
```

以后只改 K1 或 K2 一侧 layout，会直接编译失败；但如果两边一起改成了同一个、却错误的 layout，`static_assert` 无法证明 MMA 读取语义正确，仍需 correctness test。

### 4.3 `MMALayout`、`ldmatrix` 和线程数据所有权

这里最容易混淆的三件事必须分开：

1. `MMALayout` 是 shared-memory 地址布局，不是 MMA 寄存器 fragment 本身。
2. K1 写 `MMALayout(row, col)` 时，线程负责哪些逻辑元素由 K1 的循环划分决定；这不等于 K2 的 MMA lane ownership。
3. K2 的 `ldmatrix`/CUTE tiled copy 从 shared layout 取数，并重排到 MMA 指令要求的寄存器 fragment。

因此不存在“既然 K1 按 MMALayout 写了，K2 就不该再用 ldmatrix”的矛盾：

```text
K1 普通线程寄存器
  -> 按 MMALayout 地址写 shared
  -> raw bulk 保留物理字节图像
  -> K2 shared 中恢复相同图像
  -> ldmatrix 按 warp/lane 规则装入 MMA fragment
```

同理，`stmatrix` 是反方向：MMA fragment 经 warp/lane 规则写入兼容的 shared layout。它不是“每线程随便给一个地址，然后硬件自动避开所有 bank conflict”。地址布局和每 lane 的地址集合仍必须满足指令契约。

当前 BF16 `MMALayout` 来自：

```cpp
GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>
```

在当前 CUTLASS 实现中对应 `Swizzle<0,4,3>`；`B=0` 表示这里没有 XOR swizzle。它仍然是特定的 K-interleaved shared layout，但不能把它口头简称成“做了 bank-conflict swizzle”。源码注释里的 “swizzled byte image” 更准确应理解为“layout-specific physical byte image”。

另外，K1 的原始 Q/K 是 tensor TMA 按 `QKLayout` 载入 shared；它们并没有在 global→shared 时变成 `MMALayout`。只有 K1 算出的 `k_decayed/q_decayed/k_restored/...` 写入相应输出 shared tensor 时才使用 MMA 兼容 layout。

## 5. K2：recurrence kernel 的变化

### 5.1 workspace load 同样改成 raw bulk G2S

K2 的 V 和 beta 仍使用 tensor TMA，因为它们来自用户 tensor，需要正常的 global tensor 坐标/边界语义：

```cpp
cute::copy(tma_load_v.with(*tma_barrier), ...);
cute::copy(tma_load_beta.with(*tma_barrier), ...);
```

六块 K1 workspace 则直接 raw bulk 到 pipeline stage：

```cpp
cute::SM90_BULK_COPY_G2S::copy(
    ws_kd + int64_t(ws_idx) * (CHUNK * D),
    reinterpret_cast<uint64_t*>(tma_barrier),
    shared_storage.input[stage].k_decayed.begin(),
    int32_t(CHUNK * D * sizeof(BF16)));
```

六条 bulk load 都挂在当前 input-stage 的同一个 transaction barrier 上；`kTmaTransactionBytes` 仍统计 V、beta 和六块 workspace 的总字节数。

### 5.2 state 常驻寄存器

这是 K2 最核心的变化。

原始版本在每个 chunk 中：

1. Phase 1 从 shared state 用 `ldmatrix` 装成 MMA B operand。
2. Phase 6 再从 shared state 装成 C fragment。
3. 更新 state 后用 `stmatrix` 写回 shared。
4. 多个 MMA warp 之间用 named barrier 确保 state 写回完成后才能释放 input stage。

当前版本在进入 chunk loop 前只加载一次：

```cpp
ResidentStateFragment
    resident_state[kValueBlocksPerWarp][kResidentStateRowBlocks];

copy(resident_load_c,
     resident_thr_load_c.partition_S(state_block),
     resident_thr_load_c.retile_D(resident_state[bi][m]));
```

之后：

- Phase 1 直接从 `resident_state` 得到 MMA B operand。
- Phase 6 直接原地更新 `resident_state`。
- 只有最后一个 chunk 且 `HasStateOut` 时，才通过 `SM90_U16x8_STSM_T` 把最终 state spill 到 shared，供 TMA store。

对于完整 `D=VD=128` state，每个 chunk 原本会逻辑上经过大约三次完整 BF16 state shared traversal：Phase 1 read、Phase 6 read、Phase 6 write。其字节规模为：

$$
3 \times 128 \times 128 \times 2 = 98304\ \text{bytes/chunk}
$$

这只是逻辑 shared traffic 估算，不等于 profiler 中精确的 bank wavefront/transaction 数；实际还受 `ldmatrix/stmatrix` 地址合并和 bank 行为影响。

代价是寄存器占用。完整路径下，4 个 MMA warp 各负责 32 个 V 列，每个 warp 持有：

$$
128 \times 32 \times 2 = 8192\ \text{bytes/warp}
$$

平均到 32 个 lane，单是 resident state 就约为每 lane 256 B，即 64 个 32-bit register 的逻辑容量。最终实际 register count 应以 `ptxas -v`/Nsight Compute 为准，因为 fragment、临时累加器和编译器 live range 也会占寄存器。

注意：`state_acc` shared buffer仍然存在，因为初始 state 的 TMA load、FP32↔BF16 转换以及最终 state store 仍需要它。优化减少的是**每 chunk 的 state shared 往返**，不等于完全删除 state shared allocation。

### 5.3 `MOVM.T` 的新增用途

resident state 保存成 C fragment，但 Phase 1 GEMM 需要 B fragment。当前用：

```cpp
movm_transpose_c_to_b_16x16(resident_state[bi][k], tCrB);
```

helper 内部是 4 次 `SM75_U32x1_MOVM_T`，把 16x16 C fragment 的寄存器排布转成 B operand 所需排布。

这不是说原版完全没用过 MOVM；原始 FlashKDA 已经在 U 等寄存器转置路径使用过它。当前新增的关键用途是 **state C fragment → state B operand**，由此才能让 state 跨 chunk 常驻寄存器而无需先落 shared 再 `ldmatrix`。

### 5.4 output stage 延迟 acquire

原始代码在每个 chunk 一开始就：

```cpp
store_pipeline.producer_acquire(out_write);
```

但真正第一次写 output shared stage 是 Phase 5。当前把 acquire 移到 Phase 5 前：

```cpp
// Phase 1--4
...
store_pipeline.producer_acquire(out_write);
// Phase 5: write output stage
```

这样如果 store warp 尚未释放对应 stage，等待可以与本 chunk 的 Phase 1--4 计算重叠。它不改变 pipeline 深度，仍然是 3 个 input stages、2 个 output stages。

### 5.5 warp/pipeline 收尾调整

当前还包含几项较小但相关的修改：

- K2 改成 `__launch_bounds__(NumThreads, 2)`，向编译器声明至少 2 CTA/SM 的目标。
- warp id 使用 `cutlass::canonical_warp_idx_sync()`。
- load/store warp 用一次 `cute::elect_one_sync()` 得到 lane predicate。
- pipeline 初始化等待被合并，避免每构造一个 pipeline 就单独等待。
- state 驻留后删除了原来每个 chunk 的 128-thread named barrier。
- 变长序列的 K2 launch 会反向安排 sequence，使 vLLM 常见的 trailing long-prefill CTA 更早发射。
- K2 grid 从原始的 sequence/head 次序改成 x 映射 head、y 映射 sequence，使同一 sequence 的 heads 连续发射。

这些项目中，`__launch_bounds__(..., 2)` 是编译约束/提示，不保证运行时一定常驻 2 CTA；仍受实际 registers 和 shared memory 限制。

## 6. K2 V-split

### 6.1 切分对象

当前完整 K2 CTA 处理一个 `(sequence, head)` 的整个 `V=128` 维。V-split 把它拆成两个独立 CTA：

```text
CTA 0: V columns [0, 64)
CTA 1: V columns [64, 128)
```

代码把 K2 layout 参数化为：

```cpp
template <int D, int CHUNK = 16, int VD = D>
struct K2Layouts;
```

只允许：

```cpp
static_assert(VD == D || VD == 64);
static_assert(D % VD == 0);
```

重要的是，切分只发生在 V/output/state 的行或列片段上：

- `VOLayout`：`[CHUNK, VD]`。
- `StateSmemLayout`：`[VD, D]`。
- `TransposedStateSmemLayout`：`[D, VD]`。
- `FP32StateSmemLayout`：`[VD, D]`。

K1 产生的 `KD/QD/KR/GT/INV/Mqk` 仍是完整 `D=128` 的公共数据，没有跟着 V 切分。

### 6.2 CTA 映射和地址偏移

V-split 路径 launch：

```cpp
dim3 grid_k2(H * (D / VD), N);  // VD=64 时为 (2H, N)
```

kernel 内恢复：

```cpp
constexpr int kVSlices = D / VD;
int v_idx = int(blockIdx.x) % kVSlices;
int head_idx = int(blockIdx.x) / kVSlices;
```

V、state、output 的 global 地址都增加：

```cpp
v_idx * VD
```

因此两个 CTA 写的是互不重叠的 output/state 切片，不需要跨 CTA reduction。

### 6.3 每个 CTA 的资源变化

关键编译期量：

```cpp
constexpr int kValueBlocksPerWarp = VD / (4 * 16);
```

- 完整路径 `VD=128`：每个 MMA warp 两个 value blocks。
- split 路径 `VD=64`：每个 MMA warp 一个 value block。

所以以下数组的 V 方向都减半：

- `resident_state`。
- V fragments。
- U fragments/accumulators。
- output accumulators/fragments。

shared storage 里的 state、V input stage、output stage 和 FP32 state scratch 也随 `VD` 减半。按数组载荷粗略计算（不含 barrier/alignment padding）：

- 完整路径约 96 KiB 级 shared storage。
- `VD=64` 路径约 67 KiB 级 shared storage。

准确值应直接看对应实例的 `sizeof(SharedStorageK2<...>)` 或 launch 时的 `smem_size_k2`，因为 union、对齐和 barrier 会改变最终大小。

### 6.4 为什么只在低并行度自动启用

V-split 会把 K2 CTA 数从：

$$
H \times N
$$

增加到：

$$
2 \times H \times N
$$

当前 host 自动选择条件在 `csrc/flash_kda.cpp`：

```cpp
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
bool use_vsplit = 2 * H * N_val <= num_sms;
```

含义是：只有拆成两倍 CTA 后仍不超过 SM 数时才启用。目标是填补原来 `H*N` 太小时的设备空洞，而不是让所有 workload 都拆分。

这个 heuristic 很保守也很简单，它目前没有考虑：

- sequence length / `t_tiles`。
- BF16/FP32 state 输出路径差异。
- 实际寄存器占用和 occupancy。
- 不同 SM90 GPU 的具体调度差异。

### 6.5 V-split 的额外成本

两个 split CTA 的 V/state/output 是互不重叠的，但公共输入会被两个 CTA 各加载、各计算一次：

- `KD/QD/KR`。
- `GT`。
- `INV/Mqk`。
- beta。
- 一部分与 V 无关的控制和 MMA 前置工作。

当前分支是普通 6-warp V-split，**没有 cluster、没有 multicast，也没有 fused 5-warp 变体**。因此收益来自多一个 CTA 和每 CTA 更低的资源压力；代价是公共 workspace traffic/工作重复。只在原始 CTA 并行度不足时启用，正是为了让前者覆盖后者。

split TensorMap 也只在 `use_vsplit` 为真时构造，避免给默认路径增加 host launch 准备成本：

```cpp
if (use_vsplit) {
    auto tma_load_v_vsplit = make_tma_copy(...);
    auto tma_store_out_vsplit = make_tma_copy(...);
    auto [split_state_load, split_state_store] = make_state_tma(...);
    ...
}
```

最初 `1fb1d9c` 的显式 opt-in API 已在 `3fcf41b` 中改为内部自动 dispatch；当前 Python/C++ 用户接口没有 V-split flag。

## 7. 哪些东西没有变

为了避免以后把所有差异都误归为优化，下面这些边界也要记住：

- K1→K2 workspace 的总 global 读写字节数没有因为 raw bulk 减少。
- K1/K2 的主要数学公式没有因缩线程改变，只是工作分配、融合位置和数据驻留位置变化。
- V-split 的总 V/state/output 数学规模没有减半；它把一个完整 head 的 V 维分摊给两个 CTA。
- V-split 会重复加载完整公共 K1 workspace，不是免费并行。
- K2 state shared allocation没有完全消失，只是不再每 chunk 读写它。
- `MMALayout` 不代表数据已经处于 MMA 寄存器排布；`ldmatrix` 仍负责 shared→fragment。
- raw bulk 不会自动解决 bank conflict；它只复制字节。bank 行为由 restored shared layout 和后续访问指令共同决定。
- 当前 `Layout_K_INTER_Atom<BF16>` 的 `Swizzle<0,4,3>` 中 `B=0`，不能宣称它做了 XOR swizzle bank-fix。

## 8. 维护时的检查清单

以后修改这套代码，至少检查以下项目：

1. 改 K1 workspace layout 时，K2 对应 layout 是否同步修改，三个 `static_assert` 是否仍成立。
2. raw bulk 的源/目标是否仍是连续 `cosize(layout) * sizeof(T)` 字节，而不是逻辑 `size(layout)` 的错误假设。
3. K1 union 覆盖前，两个 virtual-warp pass 是否都已完成输入寄存器化。
4. tail tile 的 K 是否仍在 decay 生成点清零。
5. `resident_state` 是否跨整个 chunk loop 保持 live，是否出现意外 spill/local memory。
6. final state 只在最后 chunk spill 到 shared 的条件是否仍正确。
7. output stage acquire 是否仍发生在第一次 shared 写之前，而不是挪得过晚。
8. V-split 的 V/state/output 地址是否都带 `v_idx * VD`；公共 workspace 地址则不应带它。
9. V-split 两个 CTA 是否仍写互不重叠的 output/final-state 区间。
10. 用 `ptxas -v` 和 Nsight Compute 重新核对 register、shared、occupancy、shared wavefront 和 TMA 指标，不能只凭源码推断性能。
11. 至少运行 `tests/test_vsplit.py`，覆盖 split/default、varlen/fixed、state in/out、BF16/FP32 state 等路径，并检查要求的 bitwise/numerical 一致性。

## 9. 与其他实验文档的关系

- [FlashKDA v1: A Deep Dive](20260420-flashkda-v1-deep-dive.md)：原始 FlashKDA v1 的整体算法和 kernel 分解。
- [K2 V-split、Cluster 与寄存器实验复盘](k2-vsplit-experiment-retrospective.md)：统一收录 V-split、cluster、5/6-warp、bankfix 和 state transpose 实验。

实验复盘包含没有进入当前 `feature/k2-vsplit@3fcf41b` 的分支。判断“当前代码到底有什么”时，以本文的版本边界和实际源码为准；特别是不要把 cluster multicast、5-warp、per-warp cluster release 或 bankfix 写成当前分支既有能力。
