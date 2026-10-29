"""Single-shape smoke/correctness test for FlashKDA forward.

Run with:
    pytest tests/test_fwd_simple.py -q -s
"""

import math

import torch

import flash_kda
from torch_ref import torch_ref


def test_fwd_simple():
    # One batch, one chunk, one head.
    B, T, H, D = 2, 16, 32, 128
    lower_bound = -5.0
    device = "cuda"

    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.zeros(H, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    out_kernel = torch.zeros_like(q)
    out_ref = torch.zeros_like(q)

    flash_kda.fwd(
        q, k, v, g, beta, scale, out_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    torch_ref(
        q, k, v, g, beta, scale, out_ref,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    torch.cuda.synchronize()

    max_abs_error = (out_kernel.float() - out_ref.float()).abs().max().item()
    print(f"shape={[B, T, H, D]}, max_abs_error={max_abs_error}")
    torch.testing.assert_close(out_kernel, out_ref, rtol=0, atol=0)

if __name__ == "__main__":
    test_fwd_simple()