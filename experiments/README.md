# FlashKDA 实验资产

根目录继续维护 vLLM FlashKDA。`flashkda-v1/` 保存原 `../FlashKDA` 在 2026-10-29 的工作树，基线为 `master@f302cd6`，包含原有未提交修改与未追踪文件。两个版本的 Python 扩展接口不同，因此旧版 kernel 没有直接覆盖根目录实现。

| 位置 | 内容 |
|---|---|
| [旧版源码](flashkda-v1/csrc/) | `fwd_kernel2_1.cuh`、TMA multicast 兼容层及对应 API/launch |
| [旧版测试与 benchmark](flashkda-v1/tests/) | V-split/cluster 正确性测试、旧版 `bench_vsplit.py` |
| [Modal 入口](flashkda-v1/modal_flashkda.py) | 早期 H100 构建和测试脚本 |
| [bank conflict 微基准](flashkda-v1/bc.cu) | 与 `ncu-reports/microbench/` 对应的源码 |
| [ldmatrix 微基准](flashkda-v1/ldmatrix_cycle_bench.cu) | 同目录还有两个 `ldmatrix_*trans_case.cu` |
| [CuTe DSL GEMM](flashkda-v1/01_dense_gemm.py) | 旧工作目录中的辅助实验 |
| [历史备份](../artifacts/git/) | 原仓库 Git bundle、未提交修改 patch、refs 和工作树清单，Git clone 与 ZIP 均包含 |

`flashkda-v1/cutlass`、`docs`、`ncu-reports` 使用指向本仓库内部的相对软链接，共用依赖和文档。两份源仓库的 CUTLASS 都是 `5c149f52a436782210263fb2f19b354443a61c6a`。旧版 `pr.md` 指向统一保存的 benchmark 输出。

根目录 `pytest.ini` 将默认测试收集限定为 `tests/`，避免两个版本的同名测试和扩展混用。重跑旧版时先进入该目录，并在适合旧版 API 的环境中重新构建；携带的 `.so` 和独立 CUDA 可执行文件只是当时机器的构建产物。

```bash
conda activate base
cd experiments/flashkda-v1
FLASH_KDA_VERSION_SUFFIX=+f302cd6.local pip install -e . --no-build-isolation
python -m pytest tests/test_vsplit_cluster.py
```

旧版 `setup.py` 仅做迁移适配：检查共享 CUTLASS 是否存在，不再从子目录执行外层仓库的 submodule 命令；默认版本后缀固定为 `+f302cd6.local`，也可用 `FLASH_KDA_VERSION_SUFFIX` 覆盖。CUDA 源码原样保留。上述命令仅为恢复入口；迁移操作没有执行安装或 GPU 测试。

完整恢复旧仓库的已提交历史时，可在本工作区之外执行：

```bash
git clone artifacts/git/FlashKDA.bundle ../FlashKDA-history
```

bundle 保存已提交对象；未提交和未追踪内容以 `flashkda-v1/` 快照为准，patch 仅覆盖原仓库已追踪文件的改动。当前 vLLM 仓库的全部分支和 reflog 随根目录 `.git/` 打包，保留 `feature/k2-vsplit`、5-warp 和 6-warp 实验提交。
