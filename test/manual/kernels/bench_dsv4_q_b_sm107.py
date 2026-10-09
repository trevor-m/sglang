"""SM107 fused q_b kernel, every launch config, vs the unfused chain SGLang runs
with --dsv4-attn-backend trtllm: FlashInfer MXFP8 GEMM (bf16 out), then
fused_q_norm_rope into an FP8 Q (bf16 temp + cast). Both start from the same
MXFP8 q_lora. Timed under CUDA graphs with rotated input copies (cold L2).

The 1cta_1x1 column includes its sub-tile TMA box at <= 64 tokens.

Use the "best" column to set _DEFAULT_LAUNCH in
sglang/kernels/ops/gemm/dsv4_q_b_sm107/__init__.py, and the speedup of the best
config to set SGLANG_OPT_DSV4_FUSED_Q_B_SM107_MIN_TOKENS.

    python test/manual/kernels/bench_dsv4_q_b_sm107.py [--shapes v41_tp4 ...]
"""

import argparse

import torch
from flashinfer import mxfp8_quantize

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import (
    LAUNCH_CONFIGS,
    _default_launch_config,
    dsv4_q_b_gemm_fused,
    is_dsv4_q_b_fused_available,
)
from sglang.srt.layers.quantization.fp8_utils import (
    flashinfer_mxfp8_blockscaled_linear,
)

# name -> (q_lora_rank, local heads, eps)
SHAPES = {
    "v41_tp4": (1280, 16, None),
    "v41_tp2": (1280, 32, None),
    "v41_dep2": (1280, 64, None),
    "v4_pro_dp": (1536, 128, 1e-6),
}
TOKENS = [1, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384]
MAX_POSITION = 65536


def _median_us(fn, args) -> float:
    result = do_bench(fn, input_args=args, disable_log_bandwidth=True)
    return result.times[0] * 1e6


def _bench_shape(name, k, heads, eps):
    n = heads * 512
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    w_q, w_sf = mxfp8_quantize(w, True, alignment=32)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).cuda()
    configs = list(LAUNCH_CONFIGS)
    print(f"\n{name}: K={k} N={n} norm={eps is not None}  (us, median)")
    header = f"{'M':>6} {'unfused':>8} " + " ".join(f"{c:>9}" for c in configs)
    print(f"{header} {'best':>9} {'speedup':>8} {'default':>9}")
    for m in TOKENS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        a_q, a_sf = mxfp8_quantize(x, True, alignment=32)
        positions = torch.randint(0, MAX_POSITION, (m,), device="cuda")
        q_out = torch.empty(m, heads, 512, device="cuda", dtype=torch.float8_e4m3fn)
        args = (a_q, a_sf, w_q, w_sf, positions)

        def unfused(a_q, a_sf, w_q, w_sf, positions):
            q = flashinfer_mxfp8_blockscaled_linear(
                input=a_q,
                weight=w_q,
                weight_scale=w_sf,
                input_scale=a_sf,
                output_dtype=torch.bfloat16,
                backend="cute-dsl",
            )
            fused_q_norm_rope(q.view(m, heads, 512), q_out, eps, freqs_cis, positions)

        def fused(launch_config):
            def run(a_q, a_sf, w_q, w_sf, positions):
                dsv4_q_b_gemm_fused(
                    a_q,
                    a_sf,
                    w_q,
                    w_sf,
                    freqs_cis,
                    positions,
                    eps=eps,
                    out=q_out.view(m, n),
                    launch_config=launch_config,
                )

            return run

        t_unfused = _median_us(unfused, args)
        times = {c: _median_us(fused(c), args) for c in configs}
        best = min(times, key=times.get)
        row = f"{m:>6} {t_unfused:>8.2f} " + " ".join(
            f"{times[c]:>9.2f}" for c in configs
        )
        print(
            f"{row} {best:>9} {t_unfused / times[best]:>7.2f}x "
            f"{_default_launch_config(m, n):>9}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=None)
    args = parser.parse_args()
    assert is_dsv4_q_b_fused_available(), "needs SM107 and a Rubin-capable CuTe DSL"
    for name in args.shapes or SHAPES:
        _bench_shape(name, *SHAPES[name])
