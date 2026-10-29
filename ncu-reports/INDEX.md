# NCU 报告索引

本目录集中保存 FlashKDA 开发过程中留下的 NCU 报告。文件名依据报告中实际捕获的 kernel template、block size、cluster size 和 profiler command line 命名，而不是沿用采集时可能写错的输出名。

2026-10-29 汇总了两个本地 FlashKDA 仓库的 **23 份报告**，共 **1,118,532,472 字节**。二进制由 Git 忽略，仍随工作区 ZIP 携带；本索引、[REPORTS.json](REPORTS.json) 和 [MANIFEST.sha256](MANIFEST.sha256) 可进入 Git。下方原有映射保留原始分析结论，新增文件另列，不能只凭文件名认定 kernel 版本。

## 命名规则

```text
<GPU>__<kernel 版本或对比组>__<workload shape>__<采集范围>.ncu-rep
```

- `bN-tT-hH`：旧 `tests/bench_vsplit.py --cases B,T,H`；
- `nN-tT-hH`：vLLM benchmark 的 `pure_nN_tT_hH`；
- `full`：`--set full`；
- `pre-perwarp`：cluster kernel 仍使用四个 MMA warp 的 CTA-wide `NamedBarrier`，早于 `fa8d0bb`；
- `cache-all` / `cache-none`：采集命令显式使用的 NCU cache-control 模式。

## Kernel 版本定义

| 名称 | `VD` | CTA threads | NCU `Cluster Size` | 含义 |
|---|---:|---:|---:|---|
| `original` | 128 | 192 | 0 | 未切 V 的 K2 |
| `vsplit` | 64 | 192 | 0 | 第一代 FlashKDA V-split |
| `warp6split` | 64 | 192 | 0 | vLLM 基线，4 MMA + LOAD + STORE |
| `warp6cluster` | 64 | 192 | 2 | warp6 cluster multicast |
| `warp5split` | 64 | 160 | 0 | 4 MMA + fused LOAD/STORE，无 cluster |
| `warp5cluster` | 64 | 160 | 2 | 4 MMA + fused LOAD/STORE，cluster multicast |

下方早期报告表中的 vLLM warp5 NCU 早于 per-warp release 和 packed state bankfix：

- per-warp release：`fa8d0bb`，2026-08-07 15:21 +0800；
- warp5 packed state bankfix：`0c8a2d0`，2026-08-07 15:57 +0800。

原索引编写时尚未包含最终版本的报告。此后新增了文件名带 `final-0c8a2d0` 的报告和其他实验报告，见下节；迁移只校验文件完整性，不重新解释报告中的 kernel 元数据。

## 补入原索引未列出的报告

以下文件均在 `vllm-flashkda/`；前六份来自旧 `FlashKDA/ncu-reports`，最后一份原已位于本仓库。除后者外，以下命名按来源原样保留。

- `h100__baseline-current-toolchain__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__packed-tma__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__warp5cluster-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__warp5cluster-final-0c8a2d0__n16-t8192-h96__launchstats.ncu-rep`
- `h100__warp5split-final-0c8a2d0__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100__warp6-cluster__n16-t8192-h96__cache-none-full.ncu-rep`
- `h100-delayed-release-n16-t8192-h96-full.ncu-rep`

相关解释见 [vLLM NCU 热点分析](../docs/ncu-vllm-flashkda-hotspot-analysis.md)。从仓库根目录校验全部报告：

```bash
sha256sum -c ncu-reports/MANIFEST.sha256
```

## Microbench

设备均为 NVIDIA GeForce RTX 5060 Ti，36 SM，CC 12.0。

| 新文件 | 原文件 | Kernel | Launch |
|---|---|---|---|
| `microbench/rtx5060ti__scalar-bf16__grid1-block32__full.ncu-rep` | `bc.ncu-rep` | `scalar_bf16` | grid 1，block 32 |
| `microbench/rtx5060ti__shared-no-conflict__grid1024-block32__full.ncu-rep` | `bc_no_conflict.ncu-rep` | `no_bank_conflict` | grid 1024，block 32 |
| `microbench/rtx5060ti__shared-32way-conflict__grid1024-block32__full.ncu-rep` | `bc_conflict.ncu-rep` | `bank_conflict_32way` | grid 1024，block 32 |

## 第一代 FlashKDA

| 新文件 | 原文件 | 设备 | Shape | 实际捕获内容 |
|---|---|---|---|---|
| `flashkda/rtx5060ti__original-vsplit__b1-t2048-h32__full.ncu-rep` | `kda_k2_1.ncu-rep` | RTX 5060 Ti | `B=1,T=2048,H=32` | original + V-split，各一份 full profile |
| `flashkda/rtx5060ti__vsplit__b1-t2048-h32__launchstats.ncu-rep` | `kda_probe.ncu-rep` | RTX 5060 Ti | `B=1,T=2048,H=32` | V-split，LaunchStats |
| `flashkda/rtx5060ti__vsplit__b1-t2048-h32__launch-occupancy-sol.ncu-rep` | `kda_overview.ncu-rep` | RTX 5060 Ti | `B=1,T=2048,H=32` | V-split，LaunchStats + Occupancy + SpeedOfLight |
| `flashkda/rtx5060ti__vsplit__b1-t2048-h32__full.ncu-rep` | `kda_full.ncu-rep` | RTX 5060 Ti | `B=1,T=2048,H=32` | V-split，full profile |
| `flashkda/h100__original-vsplit__b8-t2048-h96__full.ncu-rep` | `FlashKDA_h100_k2_full_compare.ncu-rep` | H100 80GB HBM3 | `B=8,T=2048,H=96` | original + V-split，各一份 full profile |

## vLLM FlashKDA

设备均为 NVIDIA H100 80GB HBM3，132 SM，CC 9.0。

| 新文件 | 原文件 | Shape | 实际捕获内容 | 同步/缓存说明 |
|---|---|---|---|---|
| `vllm-flashkda/h100__warp6-original-split-cluster__n1-t8192-h96__full.ncu-rep` | `Vllm-FlashKDA_h100_k2_full_compare.ncu-rep` | `pure_n1_t8192_h96` | original + warp6 split + warp6 cluster | warp5 之前；cluster 使用旧 CTA-wide barrier；cache none |
| `vllm-flashkda/h100__warp6-original-split-cluster__n4-t2048-h12__full.ncu-rep` | `Vllm-FlashKDA_h100_k2_full_compare 2x1024x12.ncu-rep` | `pure_n4_t2048_h12` | original + warp6 split + warp6 cluster | 旧文件名中的 shape 错误；cluster 使用旧 CTA-wide barrier；cache none |
| `vllm-flashkda/h100__original-warp6split-warp5cluster__n1-t8192-h96__pre-perwarp-full.ncu-rep` | `Vllm-FlashKDA_h100_cluster_5warp 1x8192x96.ncu-rep` | `pure_n1_t8192_h96` | original + warp6 split + warp5 cluster | warp5 只用于 cluster；旧 CTA-wide barrier；cache none |
| `vllm-flashkda/h100__original-warp6split-warp5cluster__n4-t2048-h12__pre-perwarp-full.ncu-rep` | `Vllm-FlashKDA_h100_cluster_5warp 2x2048x12.ncu-rep` | `pure_n4_t2048_h12` | original + warp6 split + warp5 cluster | 旧文件名中的 `2x` 错误；旧 CTA-wide barrier；cache none |
| `vllm-flashkda/h100__original__n16-t8192-h96__cache-all-full.ncu-rep` | `h100_b16_t8192_h96_cluster5_original.ncu-rep` | `pure_n16_t8192_h96` | original | cache all |
| `vllm-flashkda/h100__warp5cluster__n16-t8192-h96__pre-perwarp-cache-all-full.ncu-rep` | `h100_b16_t8192_h96_cluster5_cluster.ncu-rep` | `pure_n16_t8192_h96` | warp5 cluster | 旧 CTA-wide barrier；cache all |
| `vllm-flashkda/h100__warp6split__n16-t8192-h96__cache-none-full.ncu-rep` | `h100_b16_t8192_h96_split5.ncu-rep` | `pure_n16_t8192_h96` | **实际为 warp6 split：block 192** | 旧文件名错误；cache none |
| `vllm-flashkda/h100__warp5split__n16-t8192-h96__cache-none-full.ncu-rep` | `cutlass/h100_b16_t8192_h96_split5_noncluster.ncu-rep` | `pure_n16_t8192_h96` | 实际 warp5 split：block 160 | 原报告误放在 CUTLASS 子模块目录；cache none |

## 关键时间线

| 时间 | 事件 |
|---|---|
| 2026-08-04 | 采集 warp6 original/split/cluster 两组 full report |
| 2026-08-05 12:33 +0800 | `4bdcb45`：cluster 改为 warp5 fused I/O |
| 2026-08-05 | 采集 original + warp6 split + warp5 cluster 两组报告 |
| 2026-08-07 12:45～13:59 +0800 | 采集大 shape original、warp5 cluster、warp6 split、warp5 split |
| 2026-08-07 15:21 +0800 | `fa8d0bb`：取消每 chunk 的 CTA-wide consumer barrier，改为 per-warp release |
| 2026-08-07 15:57 +0800 | `0c8a2d0`：warp5 packed state bankfix |

读取报告示例：

```bash
ncu --import \
  ncu-reports/vllm-flashkda/h100__warp5cluster__n16-t8192-h96__pre-perwarp-cache-all-full.ncu-rep \
  --page details
```
