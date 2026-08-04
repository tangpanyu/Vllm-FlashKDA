"""Bisect and localize long-sequence V-split cluster exactness failures.

This is an eager-only diagnostic, not a latency benchmark.  For each sequence
length it computes one original reference, checks the non-cluster V-split as a
control, and then reruns the cluster variant several times.  A failure report
is grouped by logical token/chunk, head, and 64-column V slice.

Examples:
    python benchmarks/debug_cluster.py
    python benchmarks/debug_cluster.py --max-t 8192 --heads 12 --repeats 10
    python benchmarks/debug_cluster.py --lengths 2048,4096,8192 --repeats 20
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math

import torch

from bench_vsplit import D, Case, Inputs, make_inputs, run_variant


CHUNK = 16
V_SLICE = 64
V_SLICES = D // V_SLICE


@dataclasses.dataclass(frozen=True)
class DiffCounts:
    output: int
    state: int
    output_max_abs: float
    state_max_abs: float

    @property
    def exact(self) -> bool:
        return self.output == 0 and self.state == 0


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    t: int
    failed_runs: int
    repeats: int

    @property
    def failed(self) -> bool:
        return self.failed_runs != 0


def _snapshot(inputs: Inputs, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    # NaNs also expose a missing output/state store instead of letting stale
    # data accidentally compare equal across repeated eager launches.
    inputs.out.fill_(float("nan"))
    inputs.final_state.fill_(float("nan"))
    run_variant(inputs, variant)
    torch.cuda.synchronize()
    return inputs.out.clone(), inputs.final_state.clone()


def _diff_counts(
    actual_out: torch.Tensor,
    expected_out: torch.Tensor,
    actual_state: torch.Tensor,
    expected_state: torch.Tensor,
) -> DiffCounts:
    output_mask = actual_out != expected_out
    state_mask = actual_state != expected_state
    output_count = int(output_mask.sum().item())
    state_count = int(state_mask.sum().item())
    output_max_abs = (
        float((actual_out.float() - expected_out.float()).abs().max().item())
        if output_count
        else 0.0
    )
    state_max_abs = (
        float((actual_state - expected_state).abs().max().item())
        if state_count
        else 0.0
    )
    return DiffCounts(
        output=output_count,
        state=state_count,
        output_max_abs=output_max_abs,
        state_max_abs=state_max_abs,
    )


def _format_top(values: torch.Tensor, labels: list[str], limit: int) -> str:
    values = values.detach().to(device="cpu", dtype=torch.int64)
    nonzero = torch.nonzero(values, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return "none"
    ranked = sorted(
        ((int(values[index]), labels[int(index)]) for index in nonzero),
        reverse=True,
    )
    return ", ".join(f"{label}:{count}" for count, label in ranked[:limit])


def _report_output_mask(mask: torch.Tensor, top: int) -> None:
    # Logical output is [batch, T, head, V].
    batch, seq_len, heads, width = mask.shape
    if width != D:
        raise RuntimeError(f"unexpected output width: {width}")

    per_token = mask.sum(dim=(2, 3), dtype=torch.int64)
    token_locations = torch.nonzero(per_token, as_tuple=False)
    first = token_locations[0].tolist()
    last = token_locations[-1].tolist()
    print(
        "      output token span: "
        f"b{first[0]}:t{first[1]}:chunk{first[1] // CHUNK} .. "
        f"b{last[0]}:t{last[1]}:chunk{last[1] // CHUNK}; "
        f"affected_tokens={token_locations.shape[0]}"
    )
    print(
        "      output by token: "
        + _format_top(
            per_token.flatten(),
            [
                f"b{batch_index}:t{token}"
                for batch_index in range(batch)
                for token in range(seq_len)
            ],
            top,
        )
    )

    chunk_values: list[torch.Tensor] = []
    chunk_labels: list[str] = []
    for batch_index in range(batch):
        for chunk in range(math.ceil(seq_len / CHUNK)):
            start = chunk * CHUNK
            chunk_values.append(per_token[batch_index, start : start + CHUNK].sum())
            chunk_labels.append(f"b{batch_index}:chunk{chunk}")
    print(
        "      output by chunk: "
        + _format_top(torch.stack(chunk_values), chunk_labels, top)
    )

    per_head = mask.sum(dim=(0, 1, 3), dtype=torch.int64)
    print(
        "      output by head: "
        + _format_top(per_head, [f"h{head}" for head in range(heads)], top)
    )

    sliced = mask.reshape(batch, seq_len, heads, V_SLICES, V_SLICE)
    per_slice = sliced.sum(dim=(0, 1, 2, 4), dtype=torch.int64)
    print(
        "      output by V-slice: "
        + _format_top(
            per_slice,
            [
                f"v{index}[{index * V_SLICE}:{(index + 1) * V_SLICE})"
                for index in range(V_SLICES)
            ],
            V_SLICES,
        )
    )

    # Count mismatched elements for each (batch, token, head, V-slice) group.
    groups = sliced.sum(dim=-1, dtype=torch.int64)
    flat = groups.flatten()
    nonzero_groups = int((flat != 0).sum().item())
    count = min(top, nonzero_groups)
    values, indices = torch.topk(flat, k=count)
    descriptions: list[str] = []
    for value, flat_index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        v_slice = flat_index % V_SLICES
        quotient = flat_index // V_SLICES
        head = quotient % heads
        quotient //= heads
        token = quotient % seq_len
        batch_index = quotient // seq_len
        descriptions.append(
            f"b{batch_index}:t{token}:chunk{token // CHUNK}:"
            f"h{head}:v{v_slice}:{value}"
        )
    print("      top output groups: " + ", ".join(descriptions))


def _report_state_mask(mask: torch.Tensor, top: int) -> None:
    # The split state mapping slices the first matrix dimension: [V, K].
    sequences, heads, value_dim, key_dim = mask.shape
    if value_dim != D or key_dim != D:
        raise RuntimeError(f"unexpected state shape: {tuple(mask.shape)}")

    sliced = mask.reshape(sequences, heads, V_SLICES, V_SLICE, key_dim)
    per_slice = sliced.sum(dim=(0, 1, 3, 4), dtype=torch.int64)
    print(
        "      state by V-slice: "
        + _format_top(
            per_slice,
            [
                f"v{index}[{index * V_SLICE}:{(index + 1) * V_SLICE})"
                for index in range(V_SLICES)
            ],
            V_SLICES,
        )
    )

    groups = sliced.sum(dim=(3, 4), dtype=torch.int64)
    flat = groups.flatten()
    nonzero_groups = int((flat != 0).sum().item())
    count = min(top, nonzero_groups)
    values, indices = torch.topk(flat, k=count)
    descriptions: list[str] = []
    for value, flat_index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        v_slice = flat_index % V_SLICES
        quotient = flat_index // V_SLICES
        head = quotient % heads
        sequence = quotient // heads
        descriptions.append(f"seq{sequence}:h{head}:v{v_slice}:{value}")
    print("      top state groups: " + ", ".join(descriptions))


def _report_diff(
    actual_out: torch.Tensor,
    expected_out: torch.Tensor,
    actual_state: torch.Tensor,
    expected_state: torch.Tensor,
    counts: DiffCounts,
    top: int,
) -> None:
    if counts.output:
        print(
            f"      output elements={counts.output}, "
            f"max_abs={counts.output_max_abs:.9g}"
        )
        _report_output_mask(actual_out != expected_out, top)
    if counts.state:
        print(
            f"      state elements={counts.state}, "
            f"max_abs={counts.state_max_abs:.9g}"
        )
        _report_state_mask(actual_state != expected_state, top)


def _probe(
    *,
    t: int,
    batch: int,
    heads: int,
    repeats: int,
    seed: int,
    top: int,
    details_every_failure: bool,
) -> ProbeResult:
    chunks = math.ceil(t / CHUNK)
    tail = t % CHUNK or CHUNK
    print(
        f"\nT={t}, chunks={chunks}, tail={tail}, "
        f"B={batch}, H={heads}, cluster_repeats={repeats}"
    )
    case = Case(name=f"debug_b{batch}_t{t}_h{heads}", batch=batch, seq_len=t, heads=heads)
    inputs = make_inputs(case, seed=seed)

    reference_out, reference_state = _snapshot(inputs, "original")
    split_out, split_state = _snapshot(inputs, "split")
    split_diff = _diff_counts(
        split_out, reference_out, split_state, reference_state
    )
    if not split_diff.exact:
        print("  CONTROL FAIL: non-cluster split differs from original")
        _report_diff(
            split_out,
            reference_out,
            split_state,
            reference_state,
            split_diff,
            top,
        )
        raise RuntimeError(
            "plain split control failed; the failure is not cluster-specific"
        )
    print("  control original == split: exact")

    failed_runs = 0
    first_cluster: tuple[torch.Tensor, torch.Tensor] | None = None
    first_failure_reported = False
    for repeat in range(repeats):
        cluster_out, cluster_state = _snapshot(inputs, "cluster")
        diff = _diff_counts(
            cluster_out, reference_out, cluster_state, reference_state
        )
        if first_cluster is None:
            first_cluster = (cluster_out, cluster_state)
            repeat_diff = DiffCounts(0, 0, 0.0, 0.0)
        else:
            repeat_diff = _diff_counts(
                cluster_out,
                first_cluster[0],
                cluster_state,
                first_cluster[1],
            )

        status = "exact" if diff.exact else "FAIL"
        nondeterminism = (
            "first-run exact"
            if repeat == 0
            else (
                "same-as-run1"
                if repeat_diff.exact
                else (
                    f"diff-vs-run1(out={repeat_diff.output},"
                    f"state={repeat_diff.state})"
                )
            )
        )
        print(
            f"  cluster run {repeat + 1}/{repeats}: {status}; "
            f"out={diff.output}, state={diff.state}, {nondeterminism}"
        )

        if not diff.exact:
            failed_runs += 1
            if details_every_failure or not first_failure_reported:
                _report_diff(
                    cluster_out,
                    reference_out,
                    cluster_state,
                    reference_state,
                    diff,
                    top,
                )
                first_failure_reported = True

    print(f"  result: cluster failures={failed_runs}/{repeats}")
    del inputs, reference_out, reference_state, split_out, split_state, first_cluster
    gc.collect()
    torch.cuda.empty_cache()
    return ProbeResult(t=t, failed_runs=failed_runs, repeats=repeats)


def _parse_lengths(value: str) -> list[int]:
    if not value.strip():
        return []
    normalized = value.replace(";", ",")
    lengths = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if any(length <= 0 for length in lengths):
        raise ValueError("all explicit lengths must be positive")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-t", type=int, default=16)
    parser.add_argument("--max-t", type=int, default=8192)
    parser.add_argument(
        "--lengths",
        default="",
        help="comma/semicolon-separated explicit T values; disables bisection",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--details-every-failure",
        action="store_true",
        help="print full location histograms for every failing eager run",
    )
    parser.add_argument(
        "--no-tail-neighbors",
        action="store_true",
        help="do not probe first/full/last token around the observed chunk boundary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.min_t, args.max_t, args.batch, args.heads, args.repeats, args.top) <= 0:
        raise ValueError("T, batch, heads, repeats, and top must all be positive")
    if args.min_t > args.max_t:
        raise ValueError("min-t cannot exceed max-t")
    if D % V_SLICE:
        raise RuntimeError(f"D={D} is not divisible by V_SLICE={V_SLICE}")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"mode: eager exactness; CHUNK={CHUNK}, V-slices={V_SLICES}x{V_SLICE}; "
        "reference=original, control=plain-split"
    )

    cache: dict[int, ProbeResult] = {}

    def probe(t: int) -> ProbeResult:
        if t not in cache:
            cache[t] = _probe(
                t=t,
                batch=args.batch,
                heads=args.heads,
                repeats=args.repeats,
                seed=args.seed,
                top=args.top,
                details_every_failure=args.details_every_failure,
            )
        return cache[t]

    explicit = _parse_lengths(args.lengths)
    if explicit:
        for t in explicit:
            probe(t)
        failed = [result.t for result in cache.values() if result.failed]
        print(f"\nexplicit summary: failing T={failed if failed else 'none'}")
        return

    # Exponentially scan chunk counts, then binary-search the first observed
    # pass/fail transition.  A race is probabilistic, so this is explicitly an
    # observed boundary under --repeats, not a proof that failure is monotonic.
    min_chunks = max(1, math.ceil(args.min_t / CHUNK))
    max_chunks = max(min_chunks, math.ceil(args.max_t / CHUNK))
    previous_pass: int | None = None
    first_fail: int | None = None
    chunks = min_chunks
    while True:
        t = min(args.max_t, chunks * CHUNK)
        result = probe(t)
        if result.failed:
            first_fail = chunks
            break
        previous_pass = chunks
        if chunks >= max_chunks:
            break
        chunks = min(max_chunks, chunks * 2)

    if first_fail is None:
        print(
            f"\nno cluster failure observed through T={args.max_t} "
            f"with {args.repeats} eager repeats per probe"
        )
        return

    if previous_pass is not None:
        low = previous_pass
        high = first_fail
        while high - low > 1:
            middle = (low + high) // 2
            result = probe(min(args.max_t, middle * CHUNK))
            if result.failed:
                high = middle
            else:
                low = middle
        first_fail = high

    boundary_t = min(args.max_t, first_fail * CHUNK)
    print(
        f"\nobserved transition: first failing chunk-count={first_fail}, "
        f"T={boundary_t}, repeats={args.repeats}"
    )

    if not args.no_tail_neighbors:
        neighbors = (
            (first_fail - 1) * CHUNK + 1,
            first_fail * CHUNK - 1,
            first_fail * CHUNK,
        )
        print("probing tail neighbors around the observed chunk boundary")
        for t in neighbors:
            if args.min_t <= t <= args.max_t:
                probe(t)

    passing = sorted(result.t for result in cache.values() if not result.failed)
    failing = sorted(result.t for result in cache.values() if result.failed)
    print(f"\nfinal observed passing T: {passing}")
    print(f"final observed failing T: {failing}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
