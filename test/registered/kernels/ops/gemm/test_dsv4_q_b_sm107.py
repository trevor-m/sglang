"""SM107 fused q_b kernel against a dequantized torch reference of the unfused chain:
wq_b GEMM, per-head RMSNorm (models with q_head_norm), RoPE and the FP8 Q cast."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import (
    LAUNCH_CONFIGS,
    dsv4_q_b_gemm_fused,
    is_dsv4_q_b_fused_available,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
    disabled="needs an SM107 (Rubin) runner",
)

pytestmark = pytest.mark.skipif(
    not is_dsv4_q_b_fused_available(),
    reason="needs SM107 and a CuTe DSL build with the Rubin helpers",
)

DEVICE = "cuda"
NUM_HEADS = 16
HEAD_DIM = 512
NOPE_DIM = 448
MAX_POSITION = 4096


def _quant_blocks(x: torch.Tensor, block_m: int, block_k: int):
    """E4M3 values and power-of-two fp32 scales per [block_m, block_k] block."""
    rows, cols = x.shape
    blocks = x.float().view(rows // block_m, block_m, cols // block_k, block_k)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=2.0**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(rows, cols)
    return q, scale


def _dequant(q: torch.Tensor, scale: torch.Tensor, block_m: int, block_k: int):
    expanded = scale.repeat_interleave(block_m, 0).repeat_interleave(block_k, 1)
    return q.float() * expanded


def _swizzle(scale: torch.Tensor) -> torch.Tensor:
    from flashinfer import block_scale_interleave

    # UE8M0 bytes; one entry per 32 K-elements of each row.
    e8m0 = (scale.view(torch.int32) >> 23).to(torch.uint8)
    return block_scale_interleave(e8m0.contiguous()).reshape(-1)


def _weight(k: int, block: tuple[int, int]):
    """wq_b as the checkpoint stores it, plus the kernel's swizzled MXFP8 scale."""
    from flashinfer import block_scale_interleave

    from sglang.srt.models.deepseek_common.dsv4_fused_q_b import (
        _block_scale_to_mxfp8_e8m0,
    )

    n = NUM_HEADS * HEAD_DIM
    w = torch.randn(n, k, device=DEVICE) / k**0.5
    w_q, w_scale = _quant_blocks(w, *block)
    e8m0 = _block_scale_to_mxfp8_e8m0(w_scale, n=n, k=k, block_size=list(block))
    assert e8m0 is not None
    w_sf = block_scale_interleave(e8m0).reshape(-1)
    return w_q, w_sf, _dequant(w_q, w_scale, *block)


def _activation(m: int, k: int, quantizer: str):
    """q_lora as MXFP8 (kernel inputs) plus its dequantized value (reference)."""
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
    if quantizer == "torch":
        x_q, x_scale = _quant_blocks(x, 1, 32)
        return x_q, _swizzle(x_scale), _dequant(x_q, x_scale, 1, 32)
    from flashinfer import mxfp8_quantize

    # The swizzled scales feed the kernel; the linear ones give the reference.
    x_q, x_sf = mxfp8_quantize(x, True, alignment=32)
    x_q_ref, x_sf_linear = mxfp8_quantize(x, False, alignment=32)
    assert torch.equal(x_q.view(torch.uint8), x_q_ref.view(torch.uint8))
    scale = (x_sf_linear.view(m, k // 32).to(torch.int32) << 23).view(torch.float32)
    return x_q, x_sf, _dequant(x_q, scale, 1, 32)


def _reference(a, w, freqs_cis, positions, eps):
    m = a.shape[0]
    x = (a @ w.t()).view(m, NUM_HEADS, HEAD_DIM)
    if eps is not None:
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    x = x.to(torch.bfloat16).float()
    pe = torch.view_as_complex(
        x[..., NOPE_DIM:].reshape(m, NUM_HEADS, 32, 2).contiguous()
    )
    pe = pe * freqs_cis[positions.long()].unsqueeze(1)
    x[..., NOPE_DIM:] = torch.view_as_real(pe).flatten(-2).to(torch.bfloat16).float()
    return x.to(torch.float8_e4m3fn)


def _assert_close(out, expected):
    out = out.view(-1, NUM_HEADS, HEAD_DIM).float()
    expected = expected.float()
    torch.testing.assert_close(
        out[..., :NOPE_DIM], expected[..., :NOPE_DIM], rtol=0.25, atol=2.0**-8
    )
    torch.testing.assert_close(
        out[..., NOPE_DIM:], expected[..., NOPE_DIM:], rtol=0.25, atol=2.0**-5
    )
    cos = torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), 0)
    assert cos > 0.999, cos


# (q_lora_rank, wq_b block, eps): DeepSeek-V4-Pro normalizes each Q head,
# DeepSeek-V4.1-Flash (q_head_norm=False) does not.
MODELS = {
    "v4_pro": (1536, (128, 128), 1e-6),
    "v41_flash": (1280, (32, 32), None),
}


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("m", [1, 5, 128, 333])
@pytest.mark.parametrize("quantizer", ["torch", "flashinfer"])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
def test_matches_reference(model, m, quantizer, pos_dtype):
    k, block, eps = MODELS[model]
    torch.manual_seed(m)
    a_q, a_sf, a = _activation(m, k, quantizer)
    w_q, w_sf, w = _weight(k, block)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).to(DEVICE)
    positions = torch.randint(0, MAX_POSITION, (m,), device=DEVICE, dtype=pos_dtype)

    out = dsv4_q_b_gemm_fused(a_q, a_sf, w_q, w_sf, freqs_cis, positions, eps=eps)
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, w, freqs_cis, positions, eps))


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("launch_config", list(LAUNCH_CONFIGS))
def test_every_launch_config_matches_reference(model, launch_config):
    k, block, eps = MODELS[model]
    w_q, w_sf, w = _weight(k, block)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).to(DEVICE)
    # One partial M tile, and several that leave the last cluster partly empty.
    for m in (5, 333):
        torch.manual_seed(m)
        a_q, a_sf, a = _activation(m, k, "torch")
        positions = torch.randint(0, MAX_POSITION, (m,), device=DEVICE)
        out = dsv4_q_b_gemm_fused(
            a_q,
            a_sf,
            w_q,
            w_sf,
            freqs_cis,
            positions,
            eps=eps,
            launch_config=launch_config,
        )
        torch.cuda.synchronize()
        _assert_close(out, _reference(a, w, freqs_cis, positions, eps))


@pytest.mark.parametrize("model", list(MODELS))
def test_1cta_sub_tile_boundaries_match_reference(model):
    """1cta_1x1 under 65 tokens loads the activation through a TMA box of the next
    power of two rows (at least 8), and each box may only serve M up to its size;
    these M sit on both sides of every box edge."""
    k, block, eps = MODELS[model]
    w_q, w_sf, w = _weight(k, block)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).to(DEVICE)
    for m in (1, 8, 9, 16, 17, 32, 33, 64, 65):
        torch.manual_seed(m)
        a_q, a_sf, a = _activation(m, k, "torch")
        positions = torch.randint(0, MAX_POSITION, (m,), device=DEVICE)
        out = dsv4_q_b_gemm_fused(
            a_q,
            a_sf,
            w_q,
            w_sf,
            freqs_cis,
            positions,
            eps=eps,
            launch_config="1cta_1x1",
        )
        torch.cuda.synchronize()
        _assert_close(out, _reference(a, w, freqs_cis, positions, eps))


@pytest.mark.parametrize("model", list(MODELS))
def test_cuda_graph_replay_matches_eager(model):
    k, block, eps = MODELS[model]
    m = 8
    torch.manual_seed(0)
    a_q, a_sf, _ = _activation(m, k, "torch")
    w_q, w_sf, _ = _weight(k, block)
    freqs_cis = precompute_freqs_cis(64, MAX_POSITION, 0, 10000, 1.0, 32, 1).to(DEVICE)
    positions = torch.arange(m, device=DEVICE, dtype=torch.int64)

    eager = dsv4_q_b_gemm_fused(a_q, a_sf, w_q, w_sf, freqs_cis, positions, eps=eps)
    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_q_b_gemm_fused(
            a_q, a_sf, w_q, w_sf, freqs_cis, positions, eps=eps, out=out
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), eager.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
