"""FlashKDA build, correctness, and benchmark entry points for Modal H100.

Commands:

    # Sync changed sources and compile into the persistent Volume (CPU only).
    modal run modal_flashkda.py --action install

    # Run forward and cluster correctness tests on an H100.
    modal run modal_flashkda.py --action test

    # Reproduce the FP32-state, CUDA-graph, cold-L2 workload used by pr.md.
    modal run modal_flashkda.py --action bench

    # Run selected named cases with the same timing parameters.
    modal run modal_flashkda.py --action bench \
        --cases 'pure_n1_t8192_h12;mixed_d31_p8161_h96' \
        --warmup 30 --iterations 200 --repeats 6 --l2-scrub-mib 256

    # Reuse the same build on B300.
    FLASHKDA_MODAL_GPU=B300 \
        modal run modal_flashkda.py --action bench
"""

from __future__ import annotations

import os
import pathlib

import modal


ROOT_DIR = pathlib.Path(__file__).parent
REMOTE_REPO = "/workspace/FlashKDA"
REMOTE_SOURCE = "/workspace/FlashKDA-source"
REMOTE_TESTS = "/workspace/FlashKDA-tests"
GPU_MODEL = os.environ.get("FLASHKDA_MODAL_GPU", "H100!")
APP_NAME = "cutlass"
REPORT_VOLUME = "cutlass-dump"
BUILD_VOLUME = "flashkda-build"

app = modal.App(name=APP_NAME)
report_volume = modal.Volume.from_name(REPORT_VOLUME, create_if_missing=True)
build_volume = modal.Volume.from_name(BUILD_VOLUME, create_if_missing=True)

# This is intentionally identical to the heavy image stack in
# 01_dense_gemm.py. Modal's content-addressed cache reuses its CUDA 13.2,
# PyTorch cu132, and CuTe DSL layers.
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg", "git")
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
    .pip_install("nvidia-cutlass-dsl==4.5.2")
)

runtime_image = (
    base_image
    .apt_install("build-essential", "ninja-build", "cmake", "rsync")
    .pip_install("pytest")
    .env({"PYTHONPATH": REMOTE_REPO})
)

# Source mounts belong only to the explicit CPU install action. They are not
# dependencies of test/bench, so those actions cannot trigger NVCC.
install_image = (
    runtime_image
    .add_local_dir(
        ROOT_DIR / "cutlass" / "include",
        remote_path=f"{REMOTE_SOURCE}/cutlass/include",
        copy=False,
    )
    .add_local_file(
        ROOT_DIR / "setup.py",
        remote_path=f"{REMOTE_SOURCE}/setup.py",
        copy=False,
    )
    .add_local_dir(
        ROOT_DIR / "flash_kda",
        remote_path=f"{REMOTE_SOURCE}/flash_kda",
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**"],
    )
    .add_local_dir(
        ROOT_DIR / "csrc",
        remote_path=f"{REMOTE_SOURCE}/csrc",
        copy=False,
    )
)

test_image = (
    runtime_image
    .add_local_dir(
        ROOT_DIR / "tests",
        remote_path=REMOTE_TESTS,
        copy=False,
        ignore=["__pycache__", "**/__pycache__/**"],
    )
)

H100_FUNCTION = {
    "gpu": GPU_MODEL,
    "image": test_image,
    "timeout": 1800,
    "volumes": {
        REMOTE_REPO: build_volume,
        "/workspace/dump": report_volume,
    },
}


def _run(command: list[str]) -> None:
    import subprocess

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_REPO, check=True)


def _print_environment() -> None:
    import torch
    import flash_kda_C

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(f"Extension: {flash_kda_C.__file__}")


def _source_hash(root: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    # Bump when the persistent build procedure changes so a stale extension
    # cannot be accepted only because its source marker already exists.
    digest.update(b"flashkda-volume-build-v2")
    for relative in ("setup.py", "flash_kda", "csrc", "cutlass/include"):
        path = root / relative
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            relative_file = file.relative_to(root)
            if "__pycache__" in relative_file.parts or file.suffix == ".pyc":
                continue
            digest.update(str(relative_file).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


@app.function(
    image=install_image,
    timeout=1800,
    volumes={REMOTE_REPO: build_volume},
)
def install() -> None:
    """Sync changed sources and compile the editable extension into the Volume."""
    import os
    import pathlib
    import shutil
    import subprocess
    import sys

    import torch

    source_root = pathlib.Path(REMOTE_SOURCE)
    repo_root = pathlib.Path(REMOTE_REPO)
    marker = repo_root / ".flashkda-source.sha256"

    source_hash = _source_hash(source_root)

    extension_exists = any(repo_root.glob("flash_kda_C*.so"))
    previous_hash = marker.read_text().strip() if marker.exists() else ""
    if previous_hash == source_hash and extension_exists:
        print("FlashKDA sources unchanged; reusing the compiled extension from Volume")
    else:
        repo_root.mkdir(parents=True, exist_ok=True)
        for relative in ("csrc", "flash_kda", "cutlass/include"):
            (repo_root / relative).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--delete", f"{source_root}/csrc/", f"{repo_root}/csrc/"],
            check=True,
        )
        subprocess.run(
            ["rsync", "-a", "--delete", f"{source_root}/flash_kda/", f"{repo_root}/flash_kda/"],
            check=True,
        )
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                f"{source_root}/cutlass/include/",
                f"{repo_root}/cutlass/include/",
            ],
            check=True,
        )
        shutil.copy2(source_root / "setup.py", repo_root / "setup.py")

        # torch's legacy editable build does not reliably notice changed CUDA
        # headers in this persistent Volume.  Refresh the owning translation
        # unit so Ninja recompiles the kernel instead of re-linking stale code.
        os.utime(repo_root / "csrc/smxx/fwd_launch.cu")

        env = os.environ.copy()
        env.update(
            {
                "CUDA_HOME": "/usr/local/cuda-13.2",
                "FLASH_KDA_CUDA_ARCHS": "90a",
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
        if not any(repo_root.glob("flash_kda_C*.so")):
            raise RuntimeError("editable install did not leave flash_kda_C in the build Volume")
        marker.write_text(source_hash + "\n")
        build_volume.commit()
        print("FlashKDA compiled and committed to persistent Volume")

    sys.path.insert(0, str(repo_root))
    import flash_kda_C

    print(f"PyTorch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(f"CUDA device visible: {torch.cuda.is_available()}")
    print(f"Extension: {flash_kda_C.__file__}")


@app.function(
    image=runtime_image,
    timeout=120,
    volumes={REMOTE_REPO: build_volume},
)
def check_install(expected_hash: str) -> None:
    """Fail on CPU before test/bench allocates an H100 if the Volume is stale."""
    import pathlib
    import sys

    import torch

    repo_root = pathlib.Path(REMOTE_REPO)
    marker = repo_root / ".flashkda-source.sha256"
    extensions = list(repo_root.glob("flash_kda_C*.so"))
    if not extensions or not marker.exists():
        raise RuntimeError(
            "FlashKDA is not installed in flashkda-build; run "
            "`modal run modal_flashkda.py --action install` first"
        )

    installed_hash = marker.read_text().strip()
    if installed_hash != expected_hash:
        raise RuntimeError(
            "FlashKDA sources changed since the Volume build; run "
            "`modal run modal_flashkda.py --action install` before test/bench"
        )

    sys.path.insert(0, str(repo_root))
    import flash_kda_C

    print(f"Preflight extension: {flash_kda_C.__file__}")
    print(f"Preflight CUDA device visible: {torch.cuda.is_available()}")


@app.function(**H100_FUNCTION)
def test() -> None:
    """Run the basic forward and V-split cluster correctness suites."""
    _print_environment()
    _run(
        [
            "python",
            "-m",
            "pytest",
            f"{REMOTE_TESTS}/test_fwd_simple.py",
            f"{REMOTE_TESTS}/test_vsplit_cluster.py",
            "-q",
            "-s",
        ]
    )


@app.function(**H100_FUNCTION)
def bench(
    mode: str,
    cases: str,
    warmup: int,
    iterations: int,
    repeats: int,
    l2_scrub_mib: int,
) -> None:
    """Run the same FP32-state, cold-cache suite used to produce pr.md."""
    _print_environment()
    command = [
        "python",
        f"{REMOTE_TESTS}/bench_vsplit.py",
        "--mode",
        mode,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--l2-scrub-mib",
        str(l2_scrub_mib),
    ]
    for case in (item.strip() for item in cases.split(";")):
        if case:
            command.extend(["--case", case])
    _run(command)


@app.local_entrypoint()
def main(
    action: str = "bench",
    mode: str = "cases",
    cases: str = "",
    warmup: int = 30,
    iterations: int = 200,
    repeats: int = 6,
    l2_scrub_mib: int = 256,
) -> None:
    if action == "install":
        install.remote()
    elif action == "test":
        check_install.remote(_source_hash(ROOT_DIR))
        test.remote()
    elif action == "bench":
        check_install.remote(_source_hash(ROOT_DIR))
        bench.remote(
            mode,
            cases,
            warmup,
            iterations,
            repeats,
            l2_scrub_mib,
        )
    else:
        raise ValueError("action must be 'install', 'test', or 'bench'")
