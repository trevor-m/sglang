"""SM107 fused q_b kernel vs the unfused chain SGLang runs with
--dsv4-attn-backend trtllm: FlashInfer MXFP8 GEMM (bf16 out), then
fused_q_norm_rope into an FP8 Q (bf16 temp + cast). Both start from the same
MXFP8 q_lora. Timed under CUDA graphs with rotated input copies (cold L2).

    python test/manual/kernels/bench_dsv4_q_b_sm107.py
"""

import torch
from flashinfer import mxfp8_quantize

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import (
    dsv4_q_b_gemm_fused,
    is_dsv4_q_b_fused_available,
)
from sglang.srt.layers.quantization.fp8_utils import (
    flashinfer_mxfp8_blockscaled_linear,
)

# (name, q_lora_rank, local heads, eps): V4-Pro with DP attention, V4.1-Flash at TP4.
SHAPES = [
    ("v4_pro_dp", 1536, 128, 1e-6),
    ("v41_flash_tp4", 1280, 16, None),
]
TOKENS = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
MAX_POSITION = 65536


def _median_us(fn, args) -> float:
    result = do_bench(fn, input_args=args, disable_log_bandwidth=True)
    return result.times[0] * 1e6


def _bench_shape(name, k, heads, eps):
    n = heads * 512
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    w_q, w_sf = mxfp8_quantize(w, True, alignment=32)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).cuda()
    print(f"\n{name}: K={k} N={n} norm={eps is not None}")
    print(f"{'M':>6} {'unfused us':>11} {'fused us':>9} {'speedup':>8}")
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

        def fused(a_q, a_sf, w_q, w_sf, positions):
            dsv4_q_b_gemm_fused(
                a_q,
                a_sf,
                w_q,
                w_sf,
                freqs_cis,
                positions,
                eps=eps,
                out=q_out.view(m, n),
            )

        t_unfused = _median_us(unfused, args)
        t_fused = _median_us(fused, args)
        print(f"{m:>6} {t_unfused:>11.2f} {t_fused:>9.2f} {t_unfused / t_fused:>7.2f}x")


if __name__ == "__main__":
    assert is_dsv4_q_b_fused_available(), "needs SM107 and a Rubin-capable CuTe DSL"
    for shape in SHAPES:
        _bench_shape(*shape)
