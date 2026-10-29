# H100 / B300 benchmark 原始输出

来自旧 FlashKDA 工作目录的 `pr.md`，下方保留原始输出；版本与实验解读见 [实验复盘](../k2-vsplit-experiment-retrospective.md)。

```bash
GPU: NVIDIA B300 SXM6 AC
FP32 state in/out: True
Cache mode: cold (256 MiB read)
Case names:
  pure_nN_tT_hH  = N equal-length sequences, T tokens each, H local heads
  mixed_dD_pP_hH = D one-token decodes plus one P-token prefill, H local heads
Latency columns are medians in microseconds (us).
Timing uses CUDA events; torch milliseconds are converted to us.
Speedup = original latency / variant latency; values above 1 are faster.
Exact = split and cluster output/state are bitwise equal to original.
case                  original us     split us   cluster us  split speedup  cluster speedup  exact
--------------------------------------------------------------------------------------------------
pure_n1_t8192_h12          561.02       444.29       444.29         1.263x           1.263x    yes
pure_n2_t4096_h12          309.12       249.73       249.66         1.238x           1.238x    yes
pure_n4_t2048_h12          182.14       151.26       151.42         1.204x           1.203x    yes
pure_n1_t8192_h96          785.28       827.30       861.66         0.949x           0.911x    yes
pure_n2_t4096_h96          688.00       824.19       869.25         0.835x           0.791x    yes
pure_n4_t2048_h96          614.37       696.19       733.02         0.882x           0.838x    yes
mixed_d7_p8185_h12         571.26       451.97       454.53         1.264x           1.257x    yes
mixed_d15_p8177_h12        575.23       455.46       458.02         1.263x           1.256x    yes
mixed_d31_p8161_h12        582.53       461.70       464.77         1.262x           1.253x    yes
mixed_d7_p8185_h96         823.20       851.95       893.66         0.966x           0.921x    yes
mixed_d15_p8177_h96        848.80       880.51       921.47         0.964x           0.921x    yes
mixed_d31_p8161_h96        895.90       930.43       978.85         0.963x           0.915x    yes
```

```bash
GPU: NVIDIA H100 80GB HBM3
FP32 state in/out: True
Cache mode: cold (256 MiB read)
Case names:
  pure_nN_tT_hH  = N equal-length sequences, T tokens each, H local heads
  mixed_dD_pP_hH = D one-token decodes plus one P-token prefill, H local heads
Latency columns are medians in microseconds (us).
Timing uses CUDA events; torch milliseconds are converted to us.
Speedup = original latency / variant latency; values above 1 are faster.
Exact = split and cluster output/state are bitwise equal to original.
case                  original us     split us   cluster us  split speedup  cluster speedup  exact
--------------------------------------------------------------------------------------------------
pure_n1_t8192_h12          645.02       474.53       490.75         1.359x           1.314x    yes
pure_n2_t4096_h12          365.22       277.09       284.22         1.318x           1.285x    yes
pure_n4_t2048_h12          225.28       177.86       182.30         1.267x           1.236x    yes
pure_n1_t8192_h96         1036.77      1012.96      1050.85         1.024x           0.987x    yes
pure_n2_t4096_h96          923.78      1027.98      1076.48         0.899x           0.858x    yes
pure_n4_t2048_h96          972.26       912.16       946.56         1.066x           1.027x    yes
mixed_d7_p8185_h12         655.10       486.27       500.93         1.347x           1.308x    yes
mixed_d15_p8177_h12        660.16       491.41       504.99         1.343x           1.307x    yes
mixed_d31_p8161_h12        669.28       499.42       513.28         1.340x           1.304x    yes
mixed_d7_p8185_h96        1382.05      1027.57      1074.08         1.345x           1.287x    yes
mixed_d15_p8177_h96       1384.58      1031.42      1080.51         1.342x           1.281x    yes
mixed_d31_p8161_h96       1417.09      1188.77      1247.52         1.192x           1.136x    yes
```


GPU: NVIDIA B300 SXM6 AC
FP32 state in/out: True
Cache mode: cold (256 MiB read)
Case names:
  pure_nN_tT_hH  = N equal-length sequences, T tokens each, H local heads
  mixed_dD_pP_hH = D one-token decodes plus one P-token prefill, H local heads
Latency columns are medians in microseconds (us).
Timing uses CUDA events; torch milliseconds are converted to us.
Speedup = original latency / variant latency; values above 1 are faster.
Exact = split output/state is bitwise equal to original.
case                  original us     split us  split speedup  exact
--------------------------------------------------------------------
pure_n1_t8192_h12          561.12       444.35         1.263x    yes
pure_n2_t4096_h12          309.22       249.79         1.238x    yes
pure_n4_t2048_h12          184.14       151.49         1.216x    yes
pure_n1_t8192_h96          785.41       829.18         0.947x    yes
pure_n2_t4096_h96          690.11       826.30         0.835x    yes
pure_n4_t2048_h96          614.46       696.26         0.883x    yes
mixed_d7_p8185_h12         572.26       452.54         1.265x    yes
mixed_d15_p8177_h12        576.29       455.62         1.265x    yes
mixed_d31_p8161_h12        583.65       462.78         1.261x    yes
mixed_d7_p8185_h96         824.10       852.96         0.966x    yes
mixed_d15_p8177_h96        849.89       882.50         0.963x    yes
mixed_d31_p8161_h96        898.72       931.81         0.964x    yes


GPU: NVIDIA H100 80GB HBM3
Compute capability: 9.0, SMs: 132
PyTorch: 2.13.0+cu132, torch CUDA: 13.2
Extension: /workspace/Vllm-FlashKDA/flash_kda/_C.abi3.so
+ /usr/local/bin/python /workspace/benchmarks/bench_vsplit.py --mode cases --warmup 30 --iterations 200 --repeats 6 --l2-scrub-mib 256
GPU: NVIDIA H100 80GB HBM3
FP32 state in/out: True
Cache mode: cold (256 MiB read)
Case names:
  pure_nN_tT_hH  = N equal-length sequences, T tokens each, H local heads
  mixed_dD_pP_hH = D one-token decodes plus one P-token prefill, H local heads
Latency columns are medians in microseconds (us).
Timing uses CUDA events; torch milliseconds are converted to us.
Speedup = original latency / variant latency; values above 1 are faster.
Exact = split output/state is bitwise equal to original.
case                  original us     split us  split speedup  exact
--------------------------------------------------------------------
pure_n1_t8192_h12          644.80       474.69         1.358x    yes
pure_n2_t4096_h12          365.09       277.02         1.318x    yes
pure_n4_t2048_h12          225.28       177.89         1.266x    yes
pure_n1_t8192_h96         1041.70      1016.45         1.025x    yes
pure_n2_t4096_h96          920.74      1023.97         0.899x    yes
pure_n4_t2048_h96          971.20       910.66         1.066x    yes
mixed_d7_p8185_h12         654.78       485.92         1.348x    yes
mixed_d15_p8177_h12        659.36       491.04         1.343x    yes
mixed_d31_p8161_h12        669.18       499.04         1.341x    yes
mixed_d7_p8185_h96        1376.19      1024.46         1.343x    yes
mixed_d15_p8177_h96       1377.12      1028.74         1.339x    yes
mixed_d31_p8161_h96       1401.06      1181.31         1.186x    yes

