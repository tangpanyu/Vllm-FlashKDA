"""Compare V-split variants at larger batches on 96 local heads.

Sequence length stays fixed at 2048 so every K2 CTA performs the same amount
of recurrent work while increasing batch size expands only the CTA grid.

Example:
    python benchmarks/bench_vsplit_large_batch.py \
        --warmup 30 --iterations 200 --repeats 6 --l2-scrub-mib 256
"""

from __future__ import annotations

import argparse

import torch

from bench_vsplit import Case, run_cases


CASES = [
    Case(name="pure_n8_t2048_h96", batch=8, seq_len=2048, heads=96),
    Case(name="pure_n16_t2048_h96", batch=16, seq_len=2048, heads=96),
    Case(name="pure_n32_t2048_h96", batch=32, seq_len=2048, heads=96),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--l2-scrub-mib", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.repeats <= 0
        or args.l2_scrub_mib < 0
    ):
        raise ValueError(
            "warmup and l2-scrub-mib must be non-negative; "
            "iterations and repeats must be positive"
        )
    run_cases(
        CASES,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        l2_scrub_mib=args.l2_scrub_mib,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
