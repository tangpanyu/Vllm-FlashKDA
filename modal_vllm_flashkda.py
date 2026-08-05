"""Build and validate the vLLM FlashKDA V-split work on Modal GPUs.

The CUDA extension is compiled explicitly by a CPU-only function and stored in
an isolated persistent Volume.  Test and benchmark functions only consume that
Volume, so invoking them can never trigger NVCC implicitly.

Commands:

    # Upload changed build inputs and compile all supported CUDA architectures
    # (sm_90a, sm_100a, sm_103a, and sm_120a) in a CPU container.
    modal run modal_vllm_flashkda.py --action install

    # Run the focused V-split/cluster correctness tests on the selected GPU.
    modal run modal_vllm_flashkda.py --action test

    # Also run the repository's complete forward correctness matrix.
    modal run modal_vllm_flashkda.py --action test --full

    # Run the PR #4 workload suite on the selected GPU.
    modal run modal_vllm_flashkda.py --action bench

    # Reuse the same all-architecture build on B300 without reinstalling.
    VLLM_FLASHKDA_GPU=B300 \
      modal run modal_vllm_flashkda.py --action bench

    # Eagerly bisect and localize the long-sequence cluster exactness issue.
    modal run modal_vllm_flashkda.py --action debug \
        --debug-max-t 8192 --debug-heads 12 --repeats 10

    # Skip bisection and repeatedly probe selected sequence lengths.
    modal run modal_vllm_flashkda.py --action debug \
        --debug-lengths '2048,4096,8192' --repeats 20

    # Override settings or select named PR #4 cases (semicolon-separated).
    modal run modal_vllm_flashkda.py --action bench \
        --cases 'pure_n1_t8192_h12;pure_n2_t4096_h12' \
        --warmup 30 --iterations 200 --repeats 6 --l2-scrub-mib 256

    # Measure steady-state cache behavior instead of cold-L2 launches.
    modal run modal_vllm_flashkda.py --action bench --l2-scrub-mib 0

    # Build dev@b5d1101 in an isolated Volume, then time only its original K2.
    VLLM_FLASHKDA_SOURCE_ROOT=/path/to/clean/dev/checkout \
    VLLM_FLASHKDA_BUILD_VOLUME=vllm-flashkda-baseline \
    VLLM_FLASHKDA_REQUIRE_VSPLIT=0 \
      modal run modal_vllm_flashkda.py --action install --force-rebuild

    VLLM_FLASHKDA_SOURCE_ROOT=/path/to/clean/dev/checkout \
    VLLM_FLASHKDA_BUILD_VOLUME=vllm-flashkda-baseline \
    VLLM_FLASHKDA_REQUIRE_VSPLIT=0 \
      modal run modal_vllm_flashkda.py --action bench \
        --mode upstream-baseline --l2-scrub-mib 256

The benchmark action accepts either ``benchmarks/bench_vsplit.py`` or
``tests/bench_vsplit.py`` while the feature is being developed.  The final PR
should keep the benchmark under ``benchmarks/``.
"""

from __future__ import annotations

import os
import pathlib

import modal


TOOL_ROOT = pathlib.Path(__file__).resolve().parent
# Development-only overrides let the same B300 harness build an independent
# upstream/dev checkout in a separate Volume for absolute baseline timings.
SOURCE_ROOT = pathlib.Path(
    os.environ.get("VLLM_FLASHKDA_SOURCE_ROOT", str(TOOL_ROOT))
).resolve()
REQUIRE_VSPLIT_SCHEMA = os.environ.get(
    "VLLM_FLASHKDA_REQUIRE_VSPLIT", "1"
) != "0"

REMOTE_REPO = "/workspace/Vllm-FlashKDA"
REMOTE_SOURCE = "/workspace/Vllm-FlashKDA-source"
REMOTE_TESTS = "/workspace/Vllm-FlashKDA-tests"
REMOTE_BENCHMARKS = "/workspace/benchmarks"
REMOTE_REPORTS = "/workspace/reports"

APP_NAME = "vllm-flashkda"
GPU_MODEL = os.environ.get("VLLM_FLASHKDA_GPU", "H100!")
BUILD_VOLUME = os.environ.get(
    "VLLM_FLASHKDA_BUILD_VOLUME", "vllm-flashkda-build"
)
REPORT_VOLUME = "vllm-flashkda-reports"

CUDA_HOME = "/usr/local/cuda-13.2"
CUDA_ARCHS = "all"
PYTHON_VERSION = "3.12"
PYTORCH_VERSION = "2.13.0+cu132"
BUILD_FINGERPRINT = (
    f"python={PYTHON_VERSION};torch={PYTORCH_VERSION};"
    f"cuda_home={CUDA_HOME};archs={CUDA_ARCHS}"
)

app = modal.App(name=APP_NAME)
build_volume = modal.Volume.from_name(BUILD_VOLUME, create_if_missing=True)
report_volume = modal.Volume.from_name(REPORT_VOLUME, create_if_missing=True)


# B300 requires a CUDA 13-compatible userspace. CUDA 13.2 and the matching
# PyTorch wheel also let the CPU builder emit an sm_103a cubin without a GPU.
base_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("wget", "gnupg", "git")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-13-2")
    .workdir("/workspace")
    .pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu132",
    )
)

runtime_image = (
    base_image
    .apt_install("build-essential", "ninja-build", "cmake", "rsync")
    .pip_install("pytest", "numpy")
    .env({"PYTHONPATH": f"{REMOTE_REPO}:/workspace"})
)


# Only the explicit CPU install action receives CUDA/C++ build inputs.  The
# CUTLASS public headers are sufficient for this extension; docs, examples,
# tests, benchmark reports, and the rest of the submodule are not uploaded.
install_image = (
    runtime_image
    .add_local_file(
        SOURCE_ROOT / "setup.py",
        remote_path=f"{REMOTE_SOURCE}/setup.py",
        copy=False,
    )
    .add_local_dir(
        SOURCE_ROOT / "flash_kda",
        remote_path=f"{REMOTE_SOURCE}/flash_kda",
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**", "*.so", "*.pyc"],
    )
    .add_local_dir(
        SOURCE_ROOT / "csrc",
        remote_path=f"{REMOTE_SOURCE}/csrc",
        copy=False,
    )
    .add_local_dir(
        SOURCE_ROOT / "cutlass" / "include",
        remote_path=f"{REMOTE_SOURCE}/cutlass/include",
        copy=False,
    )
)

test_image = (
    runtime_image
    .add_local_dir(
        TOOL_ROOT / "tests",
        remote_path=REMOTE_TESTS,
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**", "*.pyc"],
    )
    .add_local_dir(
        TOOL_ROOT / "benchmarks",
        remote_path=REMOTE_BENCHMARKS,
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**", "*.pyc"],
    )
)

# Benchmarks and tests are tiny runtime mounts and are deliberately excluded
# from the compiled-source hash. Editing them never triggers NVCC.
bench_image = (
    runtime_image
    .add_local_dir(
        TOOL_ROOT / "benchmarks",
        remote_path=REMOTE_BENCHMARKS,
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**", "*.pyc"],
    )
    .add_local_dir(
        TOOL_ROOT / "tests",
        remote_path=REMOTE_TESTS,
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**", "*.pyc"],
    )
)

B300_VOLUMES = {
    REMOTE_REPO: build_volume,
    REMOTE_REPORTS: report_volume,
}


def _extension_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return stable-ABI extension files produced by this repository."""
    return sorted((root / "flash_kda").glob("_C*.so"))


def _source_hash(root: pathlib.Path) -> str:
    """Hash source plus the toolchain/target that determines the cubin."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(BUILD_FINGERPRINT.encode())
    for relative in ("setup.py", "flash_kda", "csrc", "cutlass/include"):
        path = root / relative
        files = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
        for file in files:
            relative_file = file.relative_to(root)
            if (
                "__pycache__" in relative_file.parts
                or file.suffix in {".pyc", ".so"}
            ):
                continue
            digest.update(str(relative_file).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    import subprocess

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_REPO, check=True)


def _print_environment() -> None:
    import torch
    from flash_kda import _C

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Compute capability: {properties.major}.{properties.minor}, "
        f"SMs: {properties.multi_processor_count}"
    )
    print(f"PyTorch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(f"Extension: {_C.__file__}")


@app.function(
    image=install_image,
    cpu=16.0,
    memory=32768,
    timeout=7200,
    volumes={REMOTE_REPO: build_volume},
)
def install(
    force_rebuild: bool = False,
    require_vsplit_schema: bool = True,
) -> None:
    """Synchronize build inputs and compile every supported arch without a GPU."""
    import os
    import pathlib
    import shutil
    import subprocess
    import sys

    import torch

    source_root = pathlib.Path(REMOTE_SOURCE)
    repo_root = pathlib.Path(REMOTE_REPO)
    marker = repo_root / ".vllm-flashkda-source.sha256"
    source_hash = _source_hash(source_root)

    previous_hash = marker.read_text().strip() if marker.exists() else ""
    did_build = False
    if (
        not force_rebuild
        and previous_hash == source_hash
        and _extension_files(repo_root)
    ):
        print("Build inputs unchanged; reusing FlashKDA from the build Volume")
    else:
        did_build = True
        repo_root.mkdir(parents=True, exist_ok=True)
        for relative in ("csrc", "flash_kda", "cutlass/include"):
            (repo_root / relative).mkdir(parents=True, exist_ok=True)

        for relative in ("csrc", "flash_kda", "cutlass/include"):
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    f"{source_root / relative}/",
                    f"{repo_root / relative}/",
                ],
                check=True,
            )
        shutil.copy2(source_root / "setup.py", repo_root / "setup.py")

        # rsync preserves source mtimes, while objects persisted in the
        # Volume can have newer timestamps.  A changed fingerprint must never
        # let Ninja incorrectly reuse those stale objects.
        shutil.rmtree(repo_root / "build", ignore_errors=True)

        env = os.environ.copy()
        env.update(
            {
                "CUDA_HOME": CUDA_HOME,
                "FLASH_KDA_CUDA_ARCHS": CUDA_ARCHS,
                "NVCC_THREADS": "16",
                "MAX_JOBS": "16",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-v",
                "-e",
                ".",
                "--no-build-isolation",
            ],
            cwd=repo_root,
            env=env,
            check=True,
        )

    sys.path.insert(0, str(repo_root))
    from flash_kda import _C

    extensions = _extension_files(repo_root)
    if not extensions:
        raise RuntimeError(
            "editable install did not leave flash_kda/_C*.so in the build Volume"
        )
    schema = str(torch.ops.flash_kda.fwd.default._schema)
    if require_vsplit_schema and (
        "use_vsplit" not in schema or "use_vsplit_cluster" not in schema
    ):
        raise RuntimeError(f"installed extension has a stale schema: {schema}")

    if did_build:
        # Persist the valid marker only after the extension can be loaded and
        # its stable operator schema has been checked in this CPU container.
        marker.write_text(source_hash + "\n")
        build_volume.commit()
        print("FlashKDA all-architecture build committed to the persistent Volume")

    print(f"PyTorch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(f"CUDA device visible during install: {torch.cuda.is_available()}")
    print(f"Extension: {_C.__file__}")
    print(f"Schema: {schema}")


@app.function(
    image=runtime_image,
    timeout=120,
    volumes={REMOTE_REPO: build_volume},
)
def check_install(expected_hash: str) -> None:
    """Reject missing or stale builds before allocating a B300."""
    import pathlib
    import sys

    import torch

    repo_root = pathlib.Path(REMOTE_REPO)
    marker = repo_root / ".vllm-flashkda-source.sha256"
    if not marker.exists() or not _extension_files(repo_root):
        raise RuntimeError(
            "FlashKDA is not installed in vllm-flashkda-build; run "
            "`modal run modal_vllm_flashkda.py --action install` first"
        )

    installed_hash = marker.read_text().strip()
    if installed_hash != expected_hash:
        raise RuntimeError(
            "Local build inputs changed after the Volume build; run "
            "`modal run modal_vllm_flashkda.py --action install` first"
        )

    sys.path.insert(0, str(repo_root))
    from flash_kda import _C

    print(f"Preflight extension: {_C.__file__}")
    print(f"Preflight CUDA device visible: {torch.cuda.is_available()}")


@app.function(
    gpu=GPU_MODEL,
    image=test_image,
    timeout=3600,
    volumes=B300_VOLUMES,
)
def test(full: bool = False) -> None:
    """Run focused V-split tests, optionally followed by the full suite."""
    import pathlib
    import sys

    _print_environment()

    tests_root = pathlib.Path(REMOTE_TESTS)
    focused = [
        path
        for path in (
            tests_root / "test_vsplit_cluster.py",
            tests_root / "test_vsplit.py",
        )
        if path.exists()
    ]
    if not focused:
        raise RuntimeError(
            "No focused V-split correctness test found; expected "
            "tests/test_vsplit_cluster.py or tests/test_vsplit.py"
        )

    targets = [str(path) for path in focused]
    if full:
        targets.append(str(tests_root / "test_fwd_full.py"))

    _run([sys.executable, "-m", "pytest", *targets, "-x", "-q", "-s"])


def _benchmark_script(suite: str) -> pathlib.Path:
    if suite not in ("default", "large-batch"):
        raise ValueError("benchmark suite must be 'default' or 'large-batch'")
    filename = (
        "bench_vsplit_large_batch.py"
        if suite == "large-batch"
        else "bench_vsplit.py"
    )
    candidates = (
        pathlib.Path(REMOTE_BENCHMARKS) / filename,
        pathlib.Path(REMOTE_TESTS) / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"No V-split benchmark found for suite {suite!r}; expected {filename} "
        "under benchmarks/ or tests/"
    )


def _cluster_debug_script() -> pathlib.Path:
    script = pathlib.Path(REMOTE_BENCHMARKS) / "debug_cluster.py"
    if script.exists():
        return script
    raise RuntimeError(
        "No cluster diagnostic found; expected benchmarks/debug_cluster.py"
    )


@app.function(
    gpu=GPU_MODEL,
    image=bench_image,
    timeout=3600,
    volumes=B300_VOLUMES,
)
def bench(
    suite: str,
    mode: str,
    cases: str,
    warmup: int,
    iterations: int,
    repeats: int,
    l2_scrub_mib: int,
    profile_variant: str,
    profile_launches: int,
) -> None:
    """Run a paired original/split/cluster full-forward comparison."""
    import sys

    _print_environment()
    script = _benchmark_script(suite)
    command = [
        sys.executable,
        str(script),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--l2-scrub-mib",
        str(l2_scrub_mib),
    ]
    if suite == "default":
        command.extend(["--mode", mode])
    elif mode != "cases" or cases:
        raise ValueError(
            "the large-batch suite runs its three fixed cases with "
            "mode='cases' and no --cases filter"
        )
    if suite == "default" and mode == "profile":
        command.extend(
            [
                "--profile-variant",
                profile_variant,
                "--profile-launches",
                str(profile_launches),
            ]
        )
    if suite == "default":
        for case in (item.strip() for item in cases.split(";")):
            if case:
                command.extend(["--case", case])

    _run(command)


@app.function(
    gpu=GPU_MODEL,
    image=bench_image,
    timeout=3600,
    volumes=B300_VOLUMES,
)
def debug_cluster(
    min_t: int,
    max_t: int,
    heads: int,
    batch: int,
    repeats: int,
    seed: int,
    lengths: str,
    top: int,
    details_every_failure: bool,
) -> None:
    """Bisect cluster exactness failures using repeated eager launches."""
    import sys

    _print_environment()
    command = [
        sys.executable,
        str(_cluster_debug_script()),
        "--min-t",
        str(min_t),
        "--max-t",
        str(max_t),
        "--heads",
        str(heads),
        "--batch",
        str(batch),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--top",
        str(top),
    ]
    if lengths:
        command.extend(["--lengths", lengths])
    if details_every_failure:
        command.append("--details-every-failure")
    _run(command)


@app.local_entrypoint()
def main(
    action: str = "bench",
    suite: str = "default",
    mode: str = "cases",
    cases: str = "",
    warmup: int = 30,
    iterations: int = 200,
    repeats: int = 6,
    l2_scrub_mib: int = 256,
    profile_variant: str = "all",
    profile_launches: int = 1,
    full: bool = False,
    force_rebuild: bool = False,
    debug_min_t: int = 16,
    debug_max_t: int = 8192,
    debug_heads: int = 12,
    debug_batch: int = 1,
    debug_seed: int = 42,
    debug_lengths: str = "",
    debug_top: int = 8,
    debug_details_every_failure: bool = False,
) -> None:
    if action == "install":
        install.remote(force_rebuild, REQUIRE_VSPLIT_SCHEMA)
    elif action == "test":
        check_install.remote(_source_hash(SOURCE_ROOT))
        test.remote(full)
    elif action == "bench":
        check_install.remote(_source_hash(SOURCE_ROOT))
        bench.remote(
            suite,
            mode,
            cases,
            warmup,
            iterations,
            repeats,
            l2_scrub_mib,
            profile_variant,
            profile_launches,
        )
    elif action == "debug":
        check_install.remote(_source_hash(SOURCE_ROOT))
        debug_cluster.remote(
            debug_min_t,
            debug_max_t,
            debug_heads,
            debug_batch,
            repeats,
            debug_seed,
            debug_lengths,
            debug_top,
            debug_details_every_failure,
        )
    else:
        raise ValueError("action must be 'install', 'test', 'bench', or 'debug'")
