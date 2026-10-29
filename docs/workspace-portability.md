# FlashKDA 工作区迁移与打包

2026-10-29 将本机分散的 FlashKDA 调优资料集中到 `Vllm-FlashKDA`。根目录工作树的两个 CUDA 修改原样保留；原来源目录也保留，迁移采用复制、校验和去重。

## 收集范围

| 来源 | 仓库内位置 | 处理 |
|---|---|---|
| `../FlashKDA` 源码、测试、微基准、Modal 脚本 | `experiments/flashkda-v1/` | 保留基线 `f302cd6` 及本地修改，避免混用旧/新 API |
| `../FlashKDA/docs` | `docs/` | 共用同名文档和图片；早期源码讲解标明版本 |
| `../FlashKDA/pr.md` | `docs/archive/benchmark-session-notes.md` | 保留 benchmark 原始输出 |
| 两个 FlashKDA 仓库的 NCU | `ncu-reports/` | 23 份报告，合计 1,118,532,472 字节；Git 忽略二进制 |
| `../cutlass-cute-sample` 的 FlashKDA 笔记与流水图 | `docs/FlashKDA Kernel 讲解.md`、`docs/assets/` | 合并新增图示和说明 |
| `../cuLA/docs` 中重复的 FlashKDA/KDA 笔记 | `docs/`、`artifacts/import-originals/` | 合并新增公式，保存较早原稿供追溯 |
| `../papers`、`../flash-linear-attention` 中重复的 KDA 笔记 | 来源清单记录 | 按 SHA256 去重 |
| 旧 FlashKDA 的 Git 历史 | `artifacts/git/FlashKDA.bundle` | 保存全部 refs；另存工作树 patch 和状态 |

逐文件来源、原始 SHA256、目标 SHA256 和去重记录见 [consolidation-manifest.json](consolidation-manifest.json)。源仓库中的构建缓存、Python/clangd/pytest 缓存和 egg-info 没有复制。旧 `.so`、独立微基准二进制、PTX/SASS、图像与编辑器配置随本地快照保留，但不进入 Git。

旁边的 `vllm` 仓库工作树没有本地修改；本次没有复制整个 vLLM、SGLang、cuLA、FLA 或 Python 环境。`/tmp` 中记录的三个旧 FlashKDA worktree 已不存在，不能恢复其中没有提交、没有另存的内容；仍可用当前 `.git` 中的历史提交重建已提交版本。

Modal 的脚本、Volume 名和 run ID 已随源码/笔记保存。查询当前 Modal `main` 环境时，`vllm-flashkda-reports` 不存在，`cutlass-dump` 根目录列出的是 GEMM 实验；本次未下载远端构建缓存，也未启动云端编译或 GPU 任务。这里的“全部报告”指已发现的本地 FlashKDA 报告。

## Git 与 ZIP 的边界

Git 保留源码、测试、脚本、文档、报告索引和校验清单，以及 `artifacts/import-originals/` 中的原稿、`artifacts/git/` 中约 2.3 MiB 的旧版历史备份与 patch。原始 NCU/NSYS、构建产物和编辑器状态由 `.gitignore` 排除，但 ZIP 仍会收录。

在另一台机器通过 Git 获取工作区：

```bash
git clone --recurse-submodules --branch experiment/vsplit-final \
  https://github.com/tangpanyu/Vllm-FlashKDA.git
cd Vllm-FlashKDA
conda activate base
git submodule status
```

如果已有这个仓库：

```bash
git switch experiment/vsplit-final
git pull --ff-only origin experiment/vsplit-final
git submodule update --init --recursive
```

普通 clone 不包含原始 NCU；报告索引和文件校验清单仍然可读。需要报告时，从工作区 ZIP 解压 `ncu-reports/`，再运行 `sha256sum -c ncu-reports/MANIFEST.sha256`。Git 中的旧版快照使用相对软链接，可在另一台 Linux 机器上继续访问共用的 CUTLASS、文档和报告目录。

```bash
# 在仓库根目录执行；默认输出到上一级目录。
bash scripts/package_workspace.sh

# 也可指定仓库之外的输出文件；已有文件不会被覆盖。
bash scripts/package_workspace.sh ../Vllm-FlashKDA-transfer.zip
```

打包包含当前磁盘内容（包括未提交/未追踪文件）、NCU、旧版快照、共享 CUTLASS、根目录 `.git` 和 `artifacts`。只跳过可重新生成的缓存、`build`、egg-info、`dist`，以及已经失效或属于其他路径的 `.git/worktrees` 管理记录。脚本校验 ZIP 的 CRC，并生成独立 SHA256 文件。内部相对软链接由 `zip -y` 保留，使用 Linux `unzip` 可恢复。

`git archive` 或只复制 Git 已追踪文件会漏掉报告和本地修改，不能用于这次搬迁。根目录 `.git` 一并携带，因此解压后的分支、未提交状态和 CUTLASS submodule 信息也能保留。

## 解压与核对

```bash
sha256sum -c Vllm-FlashKDA-transfer.zip.sha256
unzip Vllm-FlashKDA-transfer.zip
cd Vllm-FlashKDA
sha256sum -c ncu-reports/MANIFEST.sha256
git status --short
git submodule status
```

进入新机器后使用 `conda activate base`，按目标 GPU/CUDA/PyTorch 环境重编译扩展。ZIP 没有携带 conda 环境、CUDA 工具链或 Modal 登录凭据；旧 `.so` 不代表在新机器上可用。`compile_commands.json`、`.clangd`、`.vscode` 可能仍包含旧机器路径，需按新的编译环境重新生成或调整。
