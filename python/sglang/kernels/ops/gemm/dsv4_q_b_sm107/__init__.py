"""DeepSeek-V4 q_b projection fused with per-head RMSNorm, RoPE and FP8 quant (SM107).

One persistent CuTe DSL kernel computes ``wq_b(q_lora)`` as an MXFP8 GEMM and
writes the attention query directly as E4M3: per 512-wide head an optional
weightless RMSNorm, GPT-J RoPE on the last 64 columns, then a satfinite cast.
The kernel is vendored from TensorRT-LLM (see ``kernel.py``); importing this
package does not import CuTe DSL.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

# name -> (mma_inst_tile, cluster_shape_mn, fallback_cluster_shape_mn). The first
# three are TRT-LLM's q_b autotuner candidates (user/lizhiz/rubin-dsv4-latest);
# 2cta_4x2 is the fixed launch of its main-branch custom op.
LAUNCH_CONFIGS = {
    "1cta_1x1": ((128, 256), (1, 1), None),
    "2cta_2x1": ((256, 256), (2, 1), None),
    "2cta_2x2": ((256, 256), (2, 2), (2, 1)),
    "2cta_4x2": ((256, 256), (4, 2), (2, 1)),
}
# (min N, max tokens, launch config); the first entry whose bounds hold wins, and
# the last entry takes everything else. Measured on SM107: 2cta_2x1 is best or
# within run-to-run noise for V4.1-Flash at TP4 and TP2 at every token count. At
# N = 65536 (V4-Pro, 128 local heads) 1cta_1x1 is 14-36% faster up to 128 tokens:
# 2-CTA launches stall while the activation is under 128 KiB, single-CTA does not.
# Re-tune with test/manual/kernels/bench_dsv4_q_b_sm107.py.
_DEFAULT_LAUNCH = (
    (65536, 128, "1cta_1x1"),
    (0, None, "2cta_2x1"),
)
# 1cta_1x1 at <= _SUB_TILE_MAX_TOKENS tokens shrinks the activation's TMA box to
# the next power of two rows (kernel tma_tile; a handle serves only M <= its box).
# The kernel author measured 6-17% faster at <= 32 tokens and 1.6% at 64 on R100
# at N = 65536, and no gain at 128.
_SUB_TILE_MAX_TOKENS = 64
# Arbitrary floor that bounds the number of compiled handles.
_SUB_TILE_MIN_ROWS = 8

HEAD_DIM = 512
# Floats per position row of torch.view_as_real(freqs_cis): 32 (cos, sin) pairs.
COS_SIN_ROW_FLOATS = 64


@functools.cache
def is_dsv4_q_b_fused_available() -> bool:
    """SM107 device and a CuTe DSL build that ships the Rubin helpers."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    try:
        import cutlass.utils.rubin_helpers  # noqa: F401
    except ImportError:
        return False
    return True


def _default_launch_config(num_tokens: int, n: int) -> str:
    for min_n, max_tokens, launch_config in _DEFAULT_LAUNCH[:-1]:
        if n >= min_n and num_tokens <= max_tokens:
            return launch_config
    return _DEFAULT_LAUNCH[-1][2]


def _sub_tile_rows(launch_config: str, num_tokens: int) -> Optional[int]:
    if launch_config != "1cta_1x1" or num_tokens > _SUB_TILE_MAX_TOKENS:
        return None
    return max(_SUB_TILE_MIN_ROWS, 1 << (num_tokens - 1).bit_length())


@functools.cache
def _compiled_kernel(
    *,
    launch_config: str,
    apply_norm: bool,
    positions_int64: bool,
    sub_tile_rows: Optional[int] = None,
):
    # Cached here: kernel.compile builds the kernel object before its own cache lookup.
    from .kernel import compile as compile_kernel

    mma_inst_tile, cluster_shape_mn, fallback_cluster_shape_mn = LAUNCH_CONFIGS[
        launch_config
    ]
    tma_tile = None if sub_tile_rows is None else (sub_tile_rows, 512, 128)
    return compile_kernel(
        mma_inst_tile=mma_inst_tile,
        cluster_shape_mn=cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        store_mode="stg256",
        swizzle_size=1,
        raster_along_m=True,
        with_quant_scale=True,
        apply_norm=apply_norm,
        cos_sin_row_floats=COS_SIN_ROW_FLOATS,
        positions_int64=positions_int64,
        tma_tile=tma_tile,
    )


@functools.cache
def _unit_quant_scale(device_index: int) -> torch.Tensor:
    # Allocated eagerly (precompile or warmup), never inside a graph capture.
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


def precompile_dsv4_q_b_fused(
    *,
    device: torch.device,
    n: int,
    apply_norm: bool,
    positions_dtype: torch.dtype = torch.int64,
) -> None:
    """Compile the default launch configs for an N-wide weight and allocate the
    kernel's constants before CUDA graph capture."""
    for min_n, _, launch_config in _DEFAULT_LAUNCH:
        if n < min_n:
            continue
        sub_tiles = {None}
        if launch_config == "1cta_1x1":
            sub_tiles |= {
                _sub_tile_rows(launch_config, m)
                for m in range(1, _SUB_TILE_MAX_TOKENS + 1)
            }
        for sub_tile_rows in sub_tiles:
            _compiled_kernel(
                launch_config=launch_config,
                apply_norm=apply_norm,
                positions_int64=positions_dtype == torch.int64,
                sub_tile_rows=sub_tile_rows,
            )
    device = torch.device(device)
    _unit_quant_scale(
        torch.cuda.current_device() if device.index is None else device.index
    )


def dsv4_q_b_gemm_fused(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: Optional[float],
    out: Optional[torch.Tensor] = None,
    launch_config: Optional[str] = None,
) -> torch.Tensor:
    """``out[m, h*512:(h+1)*512] = e4m3(rope(rmsnorm(a[m] @ weight[h*512:(h+1)*512].T)))``.

    a: [M, K] float8_e4m3fn, K % 128 == 0.
    a_scale: UE8M0 (uint8 or float8_e8m0fnu) per 32 K-elements of ``a``, in
        FlashInfer's 128x4 swizzled layout with rows padded to 128.
    weight: [N, K] float8_e4m3fn, N % 512 == 0.
    weight_scale: UE8M0 per 32 K-elements of ``weight``, 128x4 swizzled.
    cos_sin_cache: [P, 32] complex64 ``freqs_cis`` or its [P, 64] float32
        ``view_as_real``; rows are indexed by ``positions``.
    positions: [>= M] int32 or int64; the first M are used.
    eps: RMSNorm epsilon over each 512-wide head (no weight), or None to skip it.
    out: optional [M, N] float8_e4m3fn contiguous output.
    launch_config: a ``LAUNCH_CONFIGS`` key; None picks one by token count.
    """
    import cuda.bindings.driver as cuda_driver
    from cutlass.cute.runtime import from_dlpack

    m, k = a.shape
    n = weight.shape[0]
    if a.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("a and weight must be float8_e4m3fn")
    if weight.shape[1] != k or k % 128 != 0 or n % HEAD_DIM != 0:
        raise ValueError(
            f"needs K % 128 == 0 and N % {HEAD_DIM} == 0, got a {tuple(a.shape)} "
            f"weight {tuple(weight.shape)}"
        )
    if not (a.is_contiguous() and weight.is_contiguous()):
        raise ValueError("a and weight must be contiguous")
    padded_m = (m + 127) // 128 * 128
    if a_scale.numel() < padded_m * k // 32 or weight_scale.numel() != n * k // 32:
        raise ValueError(
            f"scale sizes {a_scale.numel()}, {weight_scale.numel()} do not match "
            f"the 128x4 swizzled layouts of a {tuple(a.shape)} and weight "
            f"{tuple(weight.shape)}"
        )
    if cos_sin_cache.is_complex():
        cos_sin_cache = torch.view_as_real(cos_sin_cache)
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous float32 (or complex64)")
    if cos_sin_cache.numel() % COS_SIN_ROW_FLOATS != 0:
        raise ValueError(f"cos_sin_cache rows must hold {COS_SIN_ROW_FLOATS} floats")
    if positions.dtype not in (torch.int32, torch.int64) or positions.numel() < m:
        raise ValueError("positions must be int32/int64 with at least M entries")
    if out is None:
        out = a.new_empty((m, n), dtype=torch.float8_e4m3fn)
    elif out.shape != (m, n) or out.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out must be a [{m}, {n}] float8_e4m3fn tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if launch_config is None:
        launch_config = _default_launch_config(m, n)
    elif launch_config not in LAUNCH_CONFIGS:
        raise ValueError(
            f"unknown launch_config {launch_config!r}; one of {list(LAUNCH_CONFIGS)}"
        )
    if m == 0:
        return out

    compiled = _compiled_kernel(
        launch_config=launch_config,
        apply_norm=eps is not None,
        positions_int64=positions.dtype == torch.int64,
        sub_tile_rows=_sub_tile_rows(launch_config, m),
    )

    def tvm_tensor(tensor: torch.Tensor, alignment: int = 16):
        return from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True)

    pos = positions.reshape(-1)[:m]
    compiled(
        tvm_tensor(a),
        tvm_tensor(a_scale.view(torch.float8_e8m0fnu).reshape(-1)),
        tvm_tensor(weight),
        tvm_tensor(weight_scale.view(torch.float8_e8m0fnu).reshape(-1)),
        tvm_tensor(out, 32),
        tvm_tensor(cos_sin_cache.reshape(1, -1)),
        tvm_tensor(pos, pos.element_size()),
        tvm_tensor(_unit_quant_scale(a.device.index), 4),
        0.0 if eps is None else float(eps),
        cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream),
    )
    return out


__all__ = [
    "COS_SIN_ROW_FLOATS",
    "LAUNCH_CONFIGS",
    "dsv4_q_b_gemm_fused",
    "is_dsv4_q_b_fused_available",
    "precompile_dsv4_q_b_fused",
]
