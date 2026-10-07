"""DeepSeek-V4 wq_b fused with the per-head Q RMSNorm, RoPE and FP8 cast on SM107.

The fused kernel (``sglang.kernels.ops.gemm.dsv4_q_b_sm107``) replaces the
wq_b GEMM, ``fused_q_norm_rope`` and the FP8 cast of the attention query. It
reads wq_b as MXFP8, so the weight scale is re-expressed once after loading
as UE8M0 per 32 K-elements in FlashInfer's 128x4 swizzled layout.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.runtime_context import get_exec, get_platform
from sglang.srt.utils import ceil_div, is_cuda

logger = logging.getLogger(__name__)


def fused_q_b_supported(*, head_dim: int, rope_head_dim: int, q_lora_rank: int) -> bool:
    """Static eligibility: opt-in flag, the trtllm attention backend (the only one
    that consumes FP8 Q), SM107, a Rubin-capable CuTe DSL, and the kernel's
    geometry (512-wide heads, 64 RoPE dims, K a multiple of 128)."""
    if not envs.SGLANG_OPT_DSV4_FUSED_Q_B_SM107.get():
        return False
    if get_exec().kernel.dsv4_attn_backend != "trtllm":
        return False
    if not (is_cuda() and get_platform().device_capability == (10, 7)):
        return False
    if (head_dim, rope_head_dim) != (512, 64) or q_lora_rank % 128 != 0:
        return False
    from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import is_dsv4_q_b_fused_available

    return is_dsv4_q_b_fused_available()


def _block_scale_to_mxfp8_e8m0(
    scale: torch.Tensor, *, n: int, k: int, block_size: Optional[list]
) -> Optional[torch.Tensor]:
    """Block-FP8 weight scales as per-row UE8M0 bytes [n, k // 32], or None when
    they are not powers of two (no lossless MXFP8 view)."""
    if scale.dtype == torch.int32:
        # DeepGEMM's packed UE8M0 (requant_block_scale_ue8m0_for_deepgemm) is
        # always 128x128 blocks.
        from sglang.srt.layers.quantization.fp8_utils import (
            inverse_transform_scale_ue8m0,
        )

        block_size = [128, 128]
        scale = inverse_transform_scale_ue8m0(scale, mn=n)[:, : ceil_div(k, 128)]
    if block_size is None or len(block_size) != 2:
        return None
    block_n, block_k = block_size
    if block_k % 32 != 0:
        return None
    if tuple(scale.shape) != (ceil_div(n, block_n), ceil_div(k, block_k)):
        return None
    scale = scale.float().contiguous()
    bits = scale.view(torch.int32)
    # A positive normal power of two has a zero mantissa; its exponent field is the e8m0 code.
    if not bool(torch.all((bits & 0x7FFFFF) == 0)) or not bool(torch.all(scale > 0)):
        return None
    e8m0 = (bits >> 23).to(torch.uint8)
    e8m0 = e8m0.repeat_interleave(block_n, dim=0)[:n]
    return e8m0.repeat_interleave(block_k // 32, dim=1)[:, : k // 32].contiguous()


def build_fused_q_b_weight_scale(wq_b: nn.Module) -> Optional[torch.Tensor]:
    """wq_b's weight scale in the fused kernel's layout, or None when the loaded
    weight has no MXFP8 view the kernel can read."""
    method = wq_b.quant_method
    weight = wq_b.weight
    if not isinstance(method, Fp8LinearMethod) or weight.dtype != torch.float8_e4m3fn:
        return None
    n, k = weight.shape
    backend = method.mxfp8_dense_backend
    if backend is not None and backend.is_flashinfer_trtllm():
        # The TRT-LLM MXFP8 GEMM shuffles the weight rows at load.
        return None
    mxfp8_view = method.use_mxfp8 or (
        method.block_fp8_as_mxfp8 and wq_b.block_fp8_mxfp8_ready
    )
    if mxfp8_view:
        if backend is None or not (
            backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl()
        ):
            return None
        scale = wq_b.weight_scale_inv_swizzled.view(torch.uint8).reshape(-1)
    else:
        e8m0 = _block_scale_to_mxfp8_e8m0(
            wq_b.weight_scale_inv.data,
            n=n,
            k=k,
            block_size=method.weight_block_size,
        )
        if e8m0 is None:
            return None
        from flashinfer import block_scale_interleave

        scale = block_scale_interleave(e8m0).reshape(-1)
    # n % 128 == 0 and (k // 32) % 4 == 0 leave the swizzled layout unpadded.
    if scale.numel() != n * k // 32:
        return None
    return scale


def prepare_fused_q_b(
    wq_b: nn.Module, *, apply_norm: bool, layer_id: int
) -> Optional[torch.Tensor]:
    """Post-load setup: the swizzled weight scale (None disables the fused path
    for this layer) and the compiled kernel, both ready before graph capture."""
    from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import precompile_dsv4_q_b_fused

    scale = build_fused_q_b_weight_scale(wq_b)
    if scale is None:
        logger.warning(
            "Layer %d: wq_b has no MXFP8 view the SM107 fused q_b kernel can read; "
            "keeping the unfused path.",
            layer_id,
        )
        return None
    precompile_dsv4_q_b_fused(device=wq_b.weight.device, apply_norm=apply_norm)
    return scale


def fused_q_b_forward(
    q: torch.Tensor | Mxfp8SwizzledInput,
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: Optional[float],
    q_out: torch.Tensor,
) -> torch.Tensor:
    """wq_b(q) with the per-head RMSNorm (eps not None), RoPE and FP8 cast,
    written into q_out [M, H, 512] float8_e4m3fn."""
    from sglang.kernels.ops.gemm.dsv4_q_b_sm107 import dsv4_q_b_gemm_fused

    if isinstance(q, Mxfp8SwizzledInput):
        a, a_scale = q.data, q.scales
    else:
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        a, a_scale = flashinfer_mxfp8_quantize(
            q, is_sf_swizzled_layout=True, alignment=32
        )
    dsv4_q_b_gemm_fused(
        a,
        a_scale,
        weight,
        weight_scale,
        freqs_cis,
        positions,
        eps=eps,
        out=q_out.view(q_out.shape[0], -1),
    )
    return q_out
