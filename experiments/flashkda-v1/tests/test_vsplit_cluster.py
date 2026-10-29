"""Correctness checks for the two-CTA V-split cluster path."""

import math

import pytest
import torch

import flash_kda_C


D = 128


def cluster_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.skipif(not cluster_available(), reason="TMA multicast requires SM90+")
@pytest.mark.parametrize(
    "batch,seqlen,cu_seqlens_values,state_dtype",
    [
        (1, 32, None, None),
        (2, 17, None, torch.bfloat16),
        (1, 35, [0, 17, 35], torch.float32),
    ],
)
def test_vsplit_cluster_matches_vsplit(
    batch, seqlen, cu_seqlens_values, state_dtype
):
    torch.manual_seed(2026)
    heads = 2
    shape = (batch, seqlen, heads, D)

    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(
        (batch, seqlen, heads), device="cuda", dtype=torch.bfloat16
    )
    a_log = torch.rand(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(heads, D, device="cuda", dtype=torch.float32)

    cu_seqlens = None
    sequences = batch
    if cu_seqlens_values is not None:
        cu_seqlens = torch.tensor(
            cu_seqlens_values, device="cuda", dtype=torch.int64
        )
        sequences = len(cu_seqlens_values) - 1

    initial_state = None
    if state_dtype is not None:
        initial_state = torch.randn(
            (sequences, heads, D, D), device="cuda", dtype=state_dtype
        )

    workspace = torch.empty(
        flash_kda_C.get_workspace_size(batch * seqlen, heads, sequences),
        device="cuda",
        dtype=torch.uint8,
    )

    def run(use_cluster):
        out = torch.empty_like(q)
        final_state = torch.empty_like(initial_state) if initial_state is not None else None
        flash_kda_C.fwd(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(D),
            out,
            workspace,
            a_log,
            dt_bias,
            -5.0,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            use_vsplit=True,
            use_vsplit_cluster=use_cluster,
        )
        torch.cuda.synchronize()
        return out, final_state

    expected_out, expected_state = run(False)
    cluster_out, cluster_state = run(True)

    assert torch.equal(cluster_out, expected_out)
    if expected_state is not None:
        assert torch.equal(cluster_state, expected_state)
