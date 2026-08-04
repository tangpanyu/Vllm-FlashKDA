"""Correctness and latency comparison for the K2 V-split variants.

The full case set follows the vLLM FlashKDA benchmark shape: D=128, FP32
initial/final state, and 8192 total tokens.  ``mixed`` cases model vLLM's
ordering of one-token decodes followed by one prefill sequence.

Examples:
    python benchmarks/bench_vsplit.py --mode smoke
    python benchmarks/bench_vsplit.py --mode cases
    python benchmarks/bench_vsplit.py --mode cases --case pure_n1_t8192_h96
    python benchmarks/bench_vsplit.py --mode profile \
        --case mixed_d31_p8161_h96 --profile-variant cluster

For NCU, use ``--mode profile`` so the driver emits only the requested forward
launches, then filter for ``_flash_kda_fwd_recurrence`` in NCU.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
import statistics
from collections.abc import Iterable

import torch
import torch.nn.functional as F

import flash_kda


D = 128
TOTAL_TOKENS = 8192
LOWER_BOUND = -5.0
VARIANTS = ("original", "split", "cluster")
DEFAULT_L2_SCRUB_MIB = 256


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    heads: int
    batch: int = 1
    seq_len: int = 0
    decode_count: int = 0
    prefill_len: int = 0

    def __post_init__(self) -> None:
        fixed = self.decode_count == 0 and self.prefill_len == 0
        if fixed:
            if self.batch <= 0 or self.seq_len <= 0:
                raise ValueError(f"invalid fixed case: {self}")
        elif self.batch != 1 or self.seq_len != 0:
            raise ValueError(f"mixed cases use batch=1 and seq_len=0: {self}")
        elif self.decode_count <= 0 or self.prefill_len <= 0:
            raise ValueError(f"invalid mixed case: {self}")
        if self.heads <= 0:
            raise ValueError(f"heads must be positive: {self}")

    @property
    def is_mixed(self) -> bool:
        return self.decode_count > 0

    @property
    def total_tokens(self) -> int:
        if self.is_mixed:
            return self.decode_count + self.prefill_len
        return self.batch * self.seq_len

    @property
    def num_sequences(self) -> int:
        return self.decode_count + 1 if self.is_mixed else self.batch

    @property
    def shape(self) -> tuple[int, int, int, int]:
        if self.is_mixed:
            return (1, self.total_tokens, self.heads, D)
        return (self.batch, self.seq_len, self.heads, D)

    @property
    def sequence_lengths(self) -> list[int]:
        if self.is_mixed:
            return [1] * self.decode_count + [self.prefill_len]
        return [self.seq_len] * self.batch


def _pure_cases() -> list[Case]:
    return [
        Case(
            name=f"pure_n{batch}_t{seq_len}_h{heads}",
            batch=batch,
            seq_len=seq_len,
            heads=heads,
        )
        for heads in (12, 96)
        for batch, seq_len in ((1, 8192), (2, 4096), (4, 2048))
    ]


def _mixed_cases() -> list[Case]:
    return [
        Case(
            name=f"mixed_d{decodes}_p{TOTAL_TOKENS - decodes}_h{heads}",
            heads=heads,
            decode_count=decodes,
            prefill_len=TOTAL_TOKENS - decodes,
        )
        for heads in (12, 96)
        for decodes in (7, 15, 31)
    ]


DEFAULT_CASES = _pure_cases() + _mixed_cases()

SMOKE_CASES = [
    Case(name="smoke_fixed_tail", batch=2, seq_len=37, heads=1),
    Case(name="smoke_mixed_tail", heads=1, decode_count=3, prefill_len=61),
]


@dataclasses.dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    out: torch.Tensor
    initial_state: torch.Tensor
    final_state: torch.Tensor
    workspace: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    cu_seqlens: torch.Tensor | None
    scale: float


def make_inputs(case: Case, seed: int = 42) -> Inputs:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_vsplit.py requires a CUDA device")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    shape = case.shape

    # Match the existing FlashKDA benchmark's input distribution.  K1 performs
    # normalization again; pre-normalizing here keeps this driver comparable.
    q = F.normalize(
        torch.randn(shape, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(shape, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, dtype=torch.bfloat16, device=device)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device=device)
    A_log = torch.rand(case.heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(case.heads, D, dtype=torch.float32, device=device)

    state_shape = (case.num_sequences, case.heads, D, D)
    initial_state = torch.randn(state_shape, dtype=torch.float32, device=device)
    final_state = torch.empty_like(initial_state)
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)

    cu_seqlens = None
    if case.is_mixed:
        prefix = [0]
        for length in case.sequence_lengths:
            prefix.append(prefix[-1] + length)
        # vLLM supplies int32 cu_seqlens.  Decodes precede the trailing prefill.
        cu_seqlens = torch.tensor(prefix, dtype=torch.int32, device=device)

    workspace_bytes = flash_kda.get_workspace_size(
        case.total_tokens, case.heads, case.num_sequences
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    return Inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        workspace=workspace,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        scale=1.0 / math.sqrt(D),
    )


def run_variant(inputs: Inputs, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    common_kwargs = dict(
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=inputs.initial_state,
        final_state=inputs.final_state,
        cu_seqlens=inputs.cu_seqlens,
        workspace=inputs.workspace,
    )
    if variant == "original":
        # Deliberately omit the experimental kwargs. This lets the same driver
        # measure an independently built upstream/dev baseline whose operator
        # schema predates V-split.
        flash_kda.fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            inputs.scale,
            inputs.out,
            **common_kwargs,
        )
        return

    flash_kda.fwd(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.scale,
        inputs.out,
        **common_kwargs,
        use_vsplit=True,
        use_vsplit_cluster=variant == "cluster",
    )


def _snapshot(inputs: Inputs, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs.out.zero_()
    inputs.final_state.zero_()
    run_variant(inputs, variant)
    torch.cuda.synchronize()
    return inputs.out.clone(), inputs.final_state.clone()


def assert_case_exact(case: Case, seed: int = 42) -> None:
    inputs = make_inputs(case, seed=seed)
    assert_case_exact_with_inputs(case, inputs)


def _capture_variant_graph(
    inputs: Inputs, variant: str, capture_warmup: int = 3
) -> torch.cuda.CUDAGraph:
    """Capture exactly one forward launch for one variant.

    Each variant owns an independent graph-private memory pool.  In particular,
    the beta transpose performed by the stable API may allocate during capture;
    sharing graph pools would constrain replay order, which conflicts with the
    deliberately rotated benchmark order below.
    """

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmup):
            run_variant(inputs, variant)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=capture_stream):
            run_variant(inputs, variant)
    except Exception as error:
        raise RuntimeError(
            f"CUDA graph capture failed for variant={variant}. "
            "V-split benchmarks require graph replay and will not silently "
            "fall back to eager timing."
        ) from error

    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph


def _variant_order(
    repeat: int, variants: tuple[str, ...] = VARIANTS
) -> tuple[str, ...]:
    """Balance first/middle/last position and alternate traversal direction."""

    shift = (repeat // 2) % len(variants)
    rotated = variants[shift:] + variants[:shift]
    return rotated if repeat % 2 == 0 else tuple(reversed(rotated))


def _time_graph_launches(
    graph: torch.cuda.CUDAGraph,
    scrub: torch.Tensor | None,
    scrub_sink: torch.Tensor | None,
    iterations: int,
) -> list[float]:
    """Time graph replays, optionally reading a buffer to evict L2 first."""

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for start, end in zip(starts, ends):
        if scrub is not None:
            assert scrub_sink is not None
            # Read the eviction buffer without modifying it.  The scalar sink
            # prevents the reduction from being discarded while avoiding the
            # cache-dirtying full-buffer write traffic of ``scrub.add_``.  The
            # reduction is ordered before ``start`` on the same stream, so its
            # cost is deliberately excluded from the reported launch time.
            torch.sum(
                scrub,
                dim=(0,),
                dtype=torch.int64,
                out=scrub_sink,
            )
        start.record()
        graph.replay()
        end.record()

    ends[-1].synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def benchmark_variants(
    inputs: Inputs,
    warmup: int,
    iterations: int,
    repeats: int,
    l2_scrub_mib: int,
    variants: tuple[str, ...] = VARIANTS,
) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, torch.cuda.CUDAGraph],
    tuple[torch.Tensor | None, torch.Tensor | None],
]:
    """Capture, replay, and summarize all variants under one timing policy."""

    graphs = {
        variant: _capture_variant_graph(inputs, variant) for variant in variants
    }

    # Exercise graph replay itself before timing.  Rotate warmups as well so no
    # variant is always the first graph launched after capture.
    for step in range(warmup):
        for variant in _variant_order(step, variants):
            graphs[variant].replay()
    torch.cuda.synchronize()

    if l2_scrub_mib:
        scrub_bytes = l2_scrub_mib * 1024 * 1024
        scrub = torch.zeros(scrub_bytes, dtype=torch.uint8, device="cuda")
        scrub_sink = torch.empty((), dtype=torch.int64, device="cuda")
    else:
        # Zero selects steady-state timing: launch graph replays back-to-back
        # without deliberately disturbing the cache between them.
        scrub = None
        scrub_sink = None
    samples: dict[str, list[float]] = {variant: [] for variant in variants}

    for repeat in range(repeats):
        for variant in _variant_order(repeat, variants):
            samples[variant].extend(
                _time_graph_launches(
                    graphs[variant], scrub, scrub_sink, iterations
                )
            )

    summary = {
        variant: (
            statistics.median(values),
            min(values),
            max(values),
        )
        for variant, values in samples.items()
    }
    # Return graph/scrub owners so their allocations stay alive until all
    # reported timings have been consumed by the caller.
    return summary, graphs, (scrub, scrub_sink)


def _select_cases(pool: Iterable[Case], names: list[str]) -> list[Case]:
    cases = list(pool)
    if not names:
        return cases
    by_name = {case.name: case for case in cases}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            f"unknown case(s): {', '.join(missing)}; "
            f"available: {', '.join(by_name)}"
        )
    return [by_name[name] for name in names]


def _print_case_legend() -> None:
    print("Case names:")
    print("  pure_nN_tT_hH  = N equal-length sequences, T tokens each, H local heads")
    print("  mixed_dD_pP_hH = D one-token decodes plus one P-token prefill, H local heads")


def run_cases(
    cases: list[Case],
    warmup: int,
    iterations: int,
    repeats: int,
    l2_scrub_mib: int,
) -> None:
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("FP32 state in/out: True")
    cache_mode = "steady" if l2_scrub_mib == 0 else f"cold ({l2_scrub_mib} MiB read)"
    print(f"Cache mode: {cache_mode}")
    _print_case_legend()
    print("Latency columns are medians in microseconds (us).")
    print("Timing uses CUDA events; torch milliseconds are converted to us.")
    print("Speedup = original latency / variant latency; values above 1 are faster.")
    print("Exact = split and cluster output/state are bitwise equal to original.")

    case_width = max(20, *(len(case.name) for case in cases))
    header = (
        f"{'case':<{case_width}}  {'original us':>11}  {'split us':>11}  "
        f"{'cluster us':>11}  {'split speedup':>13}  "
        f"{'cluster speedup':>15}  {'exact':>5}"
    )
    print(header)
    print("-" * len(header))

    for index, case in enumerate(cases):
        inputs = make_inputs(case, seed=42 + index)
        assert_case_exact_with_inputs(case, inputs)
        timings, graphs, scrub_owners = benchmark_variants(
            inputs,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            l2_scrub_mib=l2_scrub_mib,
        )

        original = timings["original"][0]
        split = timings["split"][0]
        cluster = timings["cluster"][0]
        print(
            f"{case.name:<{case_width}}  {original * 1000:>11.2f}  "
            f"{split * 1000:>11.2f}  {cluster * 1000:>11.2f}  "
            f"{original / split:>12.3f}x  "
            f"{original / cluster:>14.3f}x  {'yes':>5}"
        )

        del graphs, scrub_owners, inputs
        gc.collect()
        torch.cuda.empty_cache()


def run_upstream_baseline_cases(
    cases: list[Case],
    warmup: int,
    iterations: int,
    repeats: int,
    l2_scrub_mib: int,
) -> None:
    """Time only the default path from an independent upstream/dev build."""

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("FP32 state in/out: True")
    cache_mode = "steady" if l2_scrub_mib == 0 else f"cold ({l2_scrub_mib} MiB read)"
    print(f"Cache mode: {cache_mode}")
    _print_case_legend()
    print("Latency is the median in microseconds (us).")
    print("Timing uses CUDA events; torch milliseconds are converted to us.")
    case_width = max(20, *(len(case.name) for case in cases))
    header = f"{'case':<{case_width}}  {'upstream original us':>20}"
    print(header)
    print("-" * len(header))

    for index, case in enumerate(cases):
        inputs = make_inputs(case, seed=42 + index)
        # Synchronization plus NaN sentinels catches a missing store without
        # requiring the patched split/cluster schema in this baseline process.
        inputs.out.fill_(float("nan"))
        inputs.final_state.fill_(float("nan"))
        run_variant(inputs, "original")
        torch.cuda.synchronize()
        if torch.isnan(inputs.out).any() or torch.isnan(inputs.final_state).any():
            raise AssertionError(f"{case.name} upstream baseline left NaN outputs")

        timings, graphs, scrub_owners = benchmark_variants(
            inputs,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            l2_scrub_mib=l2_scrub_mib,
            variants=("original",),
        )
        median = timings["original"][0]
        print(f"{case.name:<{case_width}}  {median * 1000:>20.2f}")

        del graphs, scrub_owners, inputs
        gc.collect()
        torch.cuda.empty_cache()


def assert_case_exact_with_inputs(case: Case, inputs: Inputs) -> None:
    reference_out, reference_state = _snapshot(inputs, "original")
    for variant in ("split", "cluster"):
        actual_out, actual_state = _snapshot(inputs, variant)
        if not torch.equal(actual_out, reference_out):
            mismatch = (actual_out != reference_out).sum().item()
            max_diff = (actual_out.float() - reference_out.float()).abs().max().item()
            raise AssertionError(
                f"{case.name} {variant} output mismatch: "
                f"elements={mismatch}, max_abs={max_diff}"
            )
        if not torch.equal(actual_state, reference_state):
            mismatch = (actual_state != reference_state).sum().item()
            max_diff = (actual_state - reference_state).abs().max().item()
            raise AssertionError(
                f"{case.name} {variant} state mismatch: "
                f"elements={mismatch}, max_abs={max_diff}"
            )


def run_profile(case: Case, variant: str, launches: int, seed: int) -> None:
    inputs = make_inputs(case, seed=seed)
    variants = ("original", "split", "cluster") if variant == "all" else (variant,)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"profile case={case.name} variants={','.join(variants)} launches={launches}")
    for current in variants:
        for _ in range(launches):
            run_variant(inputs, current)
    torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "cases", "profile", "upstream-baseline"),
        default="smoke",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case name to run; repeat to select more than one",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--l2-scrub-mib",
        type=int,
        default=DEFAULT_L2_SCRUB_MIB,
        help=(
            "read-only eviction buffer before each timed graph replay, in MiB; "
            "use 0 for steady-state timing without cache scrubbing"
        ),
    )
    parser.add_argument(
        "--profile-variant",
        choices=("original", "split", "cluster", "all"),
        default="all",
    )
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cases:
        for case in DEFAULT_CASES:
            print(case.name)
        return

    if (
        min(args.warmup, args.iterations, args.repeats) < 0
        or args.iterations == 0
        or args.repeats == 0
        or args.l2_scrub_mib < 0
    ):
        raise ValueError(
            "warmup and l2-scrub-mib must be non-negative; "
            "iterations and repeats must be positive"
        )

    if args.mode == "smoke":
        cases = _select_cases(SMOKE_CASES, args.case)
        run_cases(
            cases,
            warmup=min(args.warmup, 2),
            iterations=min(args.iterations, 5),
            repeats=min(args.repeats, 2),
            l2_scrub_mib=args.l2_scrub_mib,
        )
        return

    cases = _select_cases(DEFAULT_CASES, args.case)
    if args.mode == "upstream-baseline":
        run_upstream_baseline_cases(
            cases,
            args.warmup,
            args.iterations,
            args.repeats,
            args.l2_scrub_mib,
        )
        return

    if args.mode == "profile":
        if len(cases) != 1:
            raise ValueError("profile mode requires exactly one --case")
        if args.profile_launches <= 0:
            raise ValueError("--profile-launches must be positive")
        run_profile(cases[0], args.profile_variant, args.profile_launches, args.seed)
        return

    run_cases(
        cases,
        args.warmup,
        args.iterations,
        args.repeats,
        args.l2_scrub_mib,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
