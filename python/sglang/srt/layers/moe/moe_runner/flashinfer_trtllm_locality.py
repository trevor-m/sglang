"""Locality-domain partitioning for the FlashInfer TRT-LLM MoE runner.

GPUs built from two dies (Rubin) expose two memory *locality domains*.
FlashInfer's Prims-TS partitioned MoE ops split every expert's FC1/FC2 into two
shards and run each shard on a green-context stream pinned to the SMs that are
local to one domain, reading weights placed in that domain's memory. This module
owns the SGLang side of that contract behind ``--enable-moe-locality-partition``:

* ``get_moe_locality_partition`` -- the process-wide ``LocalizedDomains``
  (pools + streams) and ``MoePartitionResources`` (reusable fork/join events)
  handle. It lives on the runtime-context resources tier so it is created once,
  at weight-loading time, before any CUDA graph capture.
* ``partition_*_moe_weights_for_locality`` -- shard the already TRT-LLM-prepared
  (shuffled / interleaved) expert weights straight into the domain pools and
  release the full-size prepared tensors, so steady-state memory holds exactly
  one copy of every weight.
* ``run_locality_partitioned_*_moe`` -- the partitioned forward calls used by
  the fused-experts functions in ``flashinfer_trtllm``.

Only the ``tp_n_tp_n`` strategy is used: FC1 output rows and FC2 output rows are
sharded, so each shard writes a disjoint half of the hidden dimension and no
cross-shard reduction is needed. The shards consume the same shuffled MajorK /
BlockMajorK tensors and R128c4-interleaved block scales as the unpartitioned
TRT-LLM kernels, so no extra weight transform is required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.runtime_context import get_context, get_exec, get_resources
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmBf16MoeQuantInfo,
        FlashInferTrtllmFp4MoeQuantInfo,
        FlashInferTrtllmFp8MoeQuantInfo,
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
    )

logger = logging.getLogger(__name__)

# The only partition geometry SGLang uses (see the module docstring).
MOE_LOCALITY_PARTITION_STRATEGY = "tp_n_tp_n"

# flashinfer.tllm_enums.WeightLayout values, mirrored so the prepared layout can
# be named without importing FlashInfer at module import time.
_WEIGHT_LAYOUT_MAJOR_K = 0
_WEIGHT_LAYOUT_BLOCK_MAJOR_K = 2

_PARTITION_COUNT = 2

_REQUIRED_FLASHINFER_HINT = (
    "--enable-moe-locality-partition requires a FlashInfer build with Prims-TS "
    "MoE localization (flashinfer.locality_domain and "
    "flashinfer.prims_ts.moe.partition)."
)


# ---------------------------------------------------------------------------
# Process-wide resources
# ---------------------------------------------------------------------------


class MoeLocalityPartition(msgspec.Struct, frozen=True):
    """Process-lifetime locality-domain resources shared by every MoE layer.

    ``domains`` is FlashInfer's ``LocalizedDomains`` (two ``MemPool``s backed by
    the locality-domain allocator plus two green-context ``ExternalStream``s);
    ``resources`` is FlashInfer's ``MoePartitionResources`` wrapping those
    streams with reusable fork/join events. Both must outlive every captured
    CUDA graph that uses them, which the resources tier guarantees.
    """

    sm_split: str
    domains: Any
    resources: Any

    @property
    def strategy(self) -> str:
        return MOE_LOCALITY_PARTITION_STRATEGY

    @property
    def device(self) -> torch.device:
        return self.domains.device

    def alloc(
        self, partition_id: int, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        return self.domains.empty(partition_id, shape, dtype)

    def pointer_domain(self, tensor: torch.Tensor) -> int:
        return int(self.domains.pointer_domain(tensor))


class MoeLocalityShards(msgspec.Struct, frozen=True):
    """Per-layer weight shards living in their matching locality domains.

    ``fc1[i]`` / ``fc2[i]`` (and the optional block-scale shards) are passed as
    ``gemm{1,2}_weights[_scale]`` to the partitioned ops; index ``i`` is both the
    partition id and the locality domain holding it.
    """

    partition: MoeLocalityPartition
    fc1: tuple[torch.Tensor, torch.Tensor]
    fc2: tuple[torch.Tensor, torch.Tensor]
    fc1_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    fc2_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def strategy(self) -> str:
        return self.partition.strategy

    @property
    def resources(self) -> Any:
        return self.partition.resources


def is_moe_locality_partition_enabled() -> bool:
    """Read the flag; an unpublished config (bare unit tests) means disabled."""
    if not get_context().is_config_namespace_published("exec"):
        return False
    return bool(get_exec().moe.enable_moe_locality_partition)


def _create_moe_locality_partition(device: torch.device) -> MoeLocalityPartition:
    sm_split = str(get_exec().moe.moe_locality_sm_split)
    try:
        from flashinfer.locality_domain import (
            LocalityDomainError,
            get_localized_domains,
        )
        from flashinfer.prims_ts.moe.partition import MoePartitionResources
    except ImportError as exc:
        raise RuntimeError(_REQUIRED_FLASHINFER_HINT) from exc

    try:
        domains = get_localized_domains(device, sm_split=sm_split)
    except (LocalityDomainError, ValueError) as exc:
        raise RuntimeError(
            "--enable-moe-locality-partition: CUDA locality domains are "
            f"unavailable on {device}: {exc}"
        ) from exc

    # Validates that both streams are green-context streams and allocates the
    # fork/join events; must exist before graph capture.
    resources = MoePartitionResources(streams=domains.streams)
    logger.info(
        "MoE locality partition (%s) on %s: %d domains, sm_split=%s, "
        "sm_counts=%s, remainder_sms=%d",
        MOE_LOCALITY_PARTITION_STRATEGY,
        device,
        domains.domain_count,
        sm_split,
        tuple(domains.sm_counts),
        domains.remainder_sm_count,
    )
    return MoeLocalityPartition(sm_split=sm_split, domains=domains, resources=resources)


def get_moe_locality_partition(
    device: torch.device | str | int,
) -> MoeLocalityPartition:
    """Get or create the process-wide locality partition for ``device``.

    Creation is a driver call (green contexts, VMM pools), so the first call
    must happen outside CUDA graph capture; weight post-processing is where
    that happens in practice.
    """
    resolved = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device)
    )
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())

    resources = get_resources()
    partition = resources.moe_locality_partition
    if partition is None:
        partition = _create_moe_locality_partition(resolved)
        resources.moe_locality_partition = partition
    elif partition.device != resolved:
        raise RuntimeError(
            "MoE locality partition was created for "
            f"{partition.device} but is requested for {resolved}"
        )
    return partition


# ---------------------------------------------------------------------------
# Weight sharding
# ---------------------------------------------------------------------------


def shard_prepared_moe_weights_for_locality(
    *,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    weight_layout: int,
    fc1_scale: Optional[torch.Tensor] = None,
    fc2_scale: Optional[torch.Tensor] = None,
    scale_k_group_size: Optional[int] = None,
    scale_dtype: Optional[torch.dtype] = None,
) -> MoeLocalityShards:
    """Shard TRT-LLM-prepared expert weights directly into the domain pools.

    ``fc1_weight`` is the prepared storage for logical ``[E, 2I, H]`` and
    ``fc2_weight`` for ``[E, H, I]``, exactly as the unpartitioned TRT-LLM kernels
    consume them; block scales are the R128c4-interleaved tensors. Each shard is
    copied straight into locality storage and its placement is verified.
    """
    try:
        from flashinfer.prims_ts.moe.partition import (
            prepared_block_major_k_bytes,
            prepared_weight_layouts,
            shard_prepared_moe_weights,
        )
    except ImportError as exc:
        raise RuntimeError(_REQUIRED_FLASHINFER_HINT) from exc

    partition = get_moe_locality_partition(fc1_weight.device)

    fc1_layout, fc2_layout = prepared_weight_layouts(
        weight_dtype=fc1_weight.dtype,
        weight_layout=weight_layout,
        use_shuffled_weight=True,
        scale_k_group_size=scale_k_group_size,
        scale_dtype=scale_dtype if scale_k_group_size is not None else None,
        fc1_block_major_k_bytes=prepared_block_major_k_bytes(fc1_weight, weight_layout),
        fc2_block_major_k_bytes=prepared_block_major_k_bytes(fc2_weight, weight_layout),
    )
    sharded = shard_prepared_moe_weights(
        fc1_weight,
        fc2_weight,
        strategy=partition.strategy,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fc1_layout=fc1_layout,
        fc2_layout=fc2_layout,
        fc1_scale=fc1_scale,
        fc2_scale=fc2_scale,
        alloc=partition.alloc,
    )

    shards = MoeLocalityShards(
        partition=partition,
        fc1=(sharded.fc1[0], sharded.fc1[1]),
        fc2=(sharded.fc2[0], sharded.fc2[1]),
        fc1_scale=(
            (sharded.fc1_scale[0], sharded.fc1_scale[1])
            if sharded.fc1_scale is not None
            else None
        ),
        fc2_scale=(
            (sharded.fc2_scale[0], sharded.fc2_scale[1])
            if sharded.fc2_scale is not None
            else None
        ),
    )
    _verify_shard_placement(shards)
    return shards


def _verify_shard_placement(shards: MoeLocalityShards) -> None:
    for partition_id in range(_PARTITION_COUNT):
        tensors = [shards.fc1[partition_id], shards.fc2[partition_id]]
        if shards.fc1_scale is not None:
            tensors.append(shards.fc1_scale[partition_id])
        if shards.fc2_scale is not None:
            tensors.append(shards.fc2_scale[partition_id])
        for tensor in tensors:
            actual = shards.partition.pointer_domain(tensor)
            if actual != partition_id:
                raise RuntimeError(
                    f"MoE weight shard for partition {partition_id} was placed in "
                    f"locality domain {actual}"
                )


def release_full_moe_weights(layer: Module, names: Sequence[str]) -> None:
    """Drop the full-size prepared tensors once their shards exist.

    Each attribute is rebound to a one-element, stride-0 placeholder of the same
    shape/dtype/device: size readers (``w2_weight.shape[2]``, ...) keep working,
    the storage goes back to the allocator, and a write into the placeholder (a
    weight reload) fails loudly.
    """
    for name in names:
        value = getattr(layer, name)
        tensor = value.data if isinstance(value, Parameter) else value
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 1:
            continue
        placeholder = torch.empty(
            (1,) * tensor.dim(), dtype=tensor.dtype, device=tensor.device
        ).expand(*tensor.shape)
        if isinstance(value, Parameter):
            value.data = placeholder
        else:
            setattr(layer, name, placeholder)


def _should_partition(layer: Module) -> bool:
    return is_moe_locality_partition_enabled() and layer.moe_locality_shards is None


def _require_gated(layer: Module, what: str) -> None:
    config = layer.moe_runner_config
    if not config.is_gated or config.activation == "situ":
        raise NotImplementedError(
            "--enable-moe-locality-partition: FlashInfer partitioned "
            f"{what} MoE requires a gated SwiGLU/GeGLU activation, got "
            f"activation={config.activation!r} (is_gated={config.is_gated})"
        )


def _release_and_record(
    layer: Module, shards: MoeLocalityShards, names: Sequence[str]
) -> None:
    layer.moe_locality_shards = shards
    release_full_moe_weights(layer, names)
    logger.debug(
        "Partitioned MoE layer across locality domains: fc1 %s, fc2 %s",
        tuple(tuple(t.shape) for t in shards.fc1),
        tuple(tuple(t.shape) for t in shards.fc2),
    )


def partition_fp4_moe_weights_for_locality(
    layer: Module, *, scale_k_group_size: int
) -> None:
    """Shard TRT-LLM-prepared packed FP4 weights (NVFP4: group 16, MXFP4: 32)."""
    if not _should_partition(layer):
        return
    _require_gated(layer, "FP4")

    w13_weight = layer.w13_weight.data
    w2_weight = layer.w2_weight.data
    if w13_weight.dtype != torch.uint8 or w2_weight.dtype != torch.uint8:
        raise ValueError(
            "FP4 locality partitioning expects packed uint8 weights, got "
            f"{w13_weight.dtype} / {w2_weight.dtype}"
        )
    shards = shard_prepared_moe_weights_for_locality(
        fc1_weight=w13_weight,
        fc2_weight=w2_weight,
        hidden_size=int(w2_weight.shape[1]),
        intermediate_size=int(w2_weight.shape[2]) * 2,
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
        fc1_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        fc2_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        scale_k_group_size=scale_k_group_size,
        scale_dtype=torch.float8_e4m3fn,
    )
    _release_and_record(
        layer,
        shards,
        ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    )


def partition_mxfp4_trtllm_gen_moe_weights_for_locality(layer: Module) -> None:
    """Shard the SM100 trtllm-gen MXFP4 layout (``flashinfer_mxfp4`` backend).

    FlashInfer's partitioned Prims-TS MoE has no row-sharded bias path yet, so
    the per-channel GEMM biases that GPT-OSS checkpoints carry cannot be
    partitioned: all-zero biases are dropped, non-zero ones are rejected.
    """
    if not _should_partition(layer):
        return
    for name in ("w13_weight_bias", "w2_weight_bias"):
        bias = getattr(layer, name).data
        if bias.numel() and bool(torch.any(bias != 0).item()):
            raise NotImplementedError(
                "--enable-moe-locality-partition: FlashInfer's partitioned MoE does "
                f"not support per-channel GEMM biases yet, but {name} is non-zero"
            )
    partition_fp4_moe_weights_for_locality(layer, scale_k_group_size=32)


def partition_fp8_per_tensor_moe_weights_for_locality(layer: Module) -> None:
    """Shard TRT-LLM-prepared per-tensor FP8 weights (shuffled MajorK)."""
    if not _should_partition(layer):
        return
    _require_gated(layer, "FP8 per-tensor")

    w2_weight = layer.w2_weight.data
    shards = shard_prepared_moe_weights_for_locality(
        fc1_weight=layer.w13_weight.data,
        fc2_weight=w2_weight,
        hidden_size=int(w2_weight.shape[1]),
        intermediate_size=int(w2_weight.shape[2]),
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
    )
    _release_and_record(layer, shards, ("w13_weight", "w2_weight"))


def partition_mxfp8_moe_weights_for_locality(layer: Module) -> None:
    """Shard TRT-LLM-prepared MXFP8 weights and their UE8M0 block scales."""
    if not _should_partition(layer):
        return
    _require_gated(layer, "MXFP8")

    w2_weight = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale_inv.data
    w2_scale = layer.w2_weight_scale_inv.data
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise ValueError(
            "MXFP8 locality partitioning expects uint8 UE8M0 block scales, got "
            f"{w13_scale.dtype} / {w2_scale.dtype}"
        )
    shards = shard_prepared_moe_weights_for_locality(
        fc1_weight=layer.w13_weight.data,
        fc2_weight=w2_weight,
        hidden_size=int(w2_weight.shape[1]),
        intermediate_size=int(w2_weight.shape[2]),
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
        fc1_scale=w13_scale,
        fc2_scale=w2_scale,
        scale_k_group_size=32,
        scale_dtype=torch.uint8,
    )
    _release_and_record(
        layer,
        shards,
        ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"),
    )


def reject_deepseek_fp8_block_locality_partition() -> None:
    """DeepSeek-style 128x128 block FP8 has no partitioned Prims-TS path."""
    if not is_moe_locality_partition_enabled():
        return
    raise NotImplementedError(
        "--enable-moe-locality-partition: FlashInfer's partitioned MoE supports "
        "MXFP8 block scales but not DeepSeek 128x128 block FP8; use an MXFP8, "
        "NVFP4, per-tensor FP8 or BF16 checkpoint"
    )


def partition_bf16_moe_weights_for_locality(layer: Module) -> None:
    """Shard TRT-LLM-prepared BF16 weights (shuffled BlockMajorK, 128B blocks).

    After ``UnquantizedFusedMoEMethod`` converts to block layout the tensors are
    ``[E, H*2/128, 2I, 64]`` and ``[E, I*2/128, H, 64]``.
    """
    if not _should_partition(layer):
        return
    _require_gated(layer, "BF16")

    w13_weight = layer.w13_weight.data
    w2_weight = layer.w2_weight.data
    if w13_weight.dim() != 4 or w2_weight.dim() != 4:
        raise ValueError(
            "BF16 locality partitioning expects BlockMajorK weights "
            "[E, K_blocks, M, block]; got shapes "
            f"{tuple(w13_weight.shape)} / {tuple(w2_weight.shape)}"
        )
    shards = shard_prepared_moe_weights_for_locality(
        fc1_weight=w13_weight,
        fc2_weight=w2_weight,
        hidden_size=int(w2_weight.shape[2]),
        intermediate_size=int(w13_weight.shape[2]) // 2,
        weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
    )
    _release_and_record(layer, shards, ("w13_weight", "w2_weight"))


# ---------------------------------------------------------------------------
# Partitioned forwards
# ---------------------------------------------------------------------------


def _unwrap_output(result: Any, output: torch.Tensor) -> torch.Tensor:
    """The partitioned ops return ``[output]`` (FP4/FP8) or ``output`` (BF16)."""
    if isinstance(result, (list, tuple)):
        result = result[0] if result else output
    return result if isinstance(result, torch.Tensor) else output


def _enable_pdl(num_tokens: int) -> bool:
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        trtllm_moe_enable_pdl,
    )

    return trtllm_moe_enable_pdl(num_tokens)


def _routed_ids_and_weights(
    topk_output,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        _get_routing_for_flashinfer_routed,
    )

    routing = _get_routing_for_flashinfer_routed(topk_output)
    if isinstance(routing, tuple):
        return routing[0], routing[1]
    return routing, None


def _require_finalize(defer_finalize: bool, what: str) -> None:
    if defer_finalize:
        raise NotImplementedError(
            "--enable-moe-locality-partition: deferred finalize is not supported "
            f"for the partitioned {what} MoE path"
        )


def run_locality_partitioned_fp4_moe(
    *,
    shards: MoeLocalityShards,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    topk_output,
    use_routed_topk: bool,
    defer_finalize: bool,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    per_token_scale: Optional[torch.Tensor],
    activation_type: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 (ModelOpt / compressed-tensors) partitioned forward."""
    from flashinfer.fused_moe import prims_ts_fp4_block_scale_moe_partitioned

    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    _require_finalize(defer_finalize, "FP4")
    if per_token_scale is not None:
        raise NotImplementedError(
            "--enable-moe-locality-partition: per-token NVFP4 activation scaling "
            "is not supported by the partitioned FP4 MoE path"
        )
    num_tokens = hidden_states.shape[0]
    kwargs = dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=list(shards.fc1),
        gemm1_weights_scale=list(shards.fc1_scale),
        gemm2_weights=list(shards.fc2),
        gemm2_weights_scale=list(shards.fc2_scale),
        output1_scale_scalar=quant_info.g1_scale_c,
        output1_scale_gate_scalar=quant_info.g1_alphas,
        output2_scale_scalar=quant_info.g2_alphas,
        num_experts=quant_info.global_num_experts,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        partition_resources=shards.resources,
        partition_strategy=shards.strategy,
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
        do_finalize=True,
        enable_pdl=_enable_pdl(num_tokens),
        activation_type=activation_type,
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=next_power_of_2(num_tokens),
        gemm1_alpha=quant_info.gemm1_alpha,
        gemm1_beta=quant_info.gemm1_beta,
        gemm1_clamp_limit=quant_info.gemm1_clamp_limit,
    )
    if use_routed_topk:
        topk_ids, topk_weights = _routed_ids_and_weights(topk_output)
        kwargs.update(
            routing_logits=None,
            routing_bias=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            top_k=int(topk_ids.shape[1]),
            # Mirrors trtllm_fp4_block_scale_routed_moe: unused with precomputed
            # routing, but 0/0/None/Renormalize pass the kernel's validation.
            n_group=0,
            topk_group=0,
            routed_scaling_factor=None,
            routing_method_type=int(RoutingMethodType.Renormalize),
        )
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)
        topk_config = topk_output.topk_config
        routing_method_type = quant_info.routing_method_type
        kwargs.update(
            routing_logits=topk_output.router_logits,
            routing_bias=topk_config.correction_bias,
            topk_ids=None,
            topk_weights=None,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            routing_method_type=int(
                routing_method_type
                if routing_method_type is not None
                else RoutingMethodType.Default
            ),
        )
    result = prims_ts_fp4_block_scale_moe_partitioned(**kwargs)
    return _unwrap_output(result, output)


def run_locality_partitioned_mxfp4_moe(
    *,
    shards: MoeLocalityShards,
    quant_info: FlashInferTrtllmGenMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    topk_output,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    output: torch.Tensor,
) -> torch.Tensor:
    """SM100 trtllm-gen MXFP4 (``flashinfer_mxfp4``) partitioned forward.

    ``hidden_states`` is MXFP8 (with ``hidden_states_scale``) for the default
    precision or BF16 (scale ``None``) for ``--flashinfer-mxfp4-moe-precision
    bf16``; both activation modes have a partitioned Prims-TS runner.
    """
    from flashinfer.fused_moe import prims_ts_fp4_block_scale_moe_partitioned

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    if runner_config.activation == "situ":
        raise NotImplementedError(
            "--enable-moe-locality-partition: SiTU activation has no partitioned "
            "Prims-TS path"
        )
    assert TopKOutputChecker.format_is_bypassed(topk_output), (
        "partitioned flashinfer_mxfp4 requires bypassed (logits) routing"
    )
    topk_config = topk_output.topk_config
    num_tokens = hidden_states.shape[0]
    hidden_size = int(quant_info.hidden_size)

    # The kernel writes the (possibly padded) kernel hidden width; the caller's
    # buffer has the model's. Only write into it directly when they agree.
    kernel_output = output
    if int(output.shape[-1]) != hidden_size:
        kernel_output = torch.empty(
            num_tokens, hidden_size, dtype=output.dtype, device=output.device
        )

    result = prims_ts_fp4_block_scale_moe_partitioned(
        routing_logits=topk_output.router_logits.to(torch.bfloat16),
        routing_bias=None,
        topk_ids=None,
        topk_weights=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=list(shards.fc1),
        gemm1_weights_scale=list(shards.fc1_scale),
        gemm2_weights=list(shards.fc2),
        gemm2_weights_scale=list(shards.fc2_scale),
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=quant_info.global_num_experts,
        top_k=topk_config.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        partition_resources=shards.resources,
        partition_strategy=shards.strategy,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Renormalize),
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
        do_finalize=True,
        enable_pdl=_enable_pdl(num_tokens),
        activation_type=get_activation_type(runner_config.activation, is_gated=True),
        output=kernel_output,
        tune_max_num_tokens=next_power_of_2(num_tokens),
        gemm1_alpha=quant_info.gemm1_alpha,
        gemm1_beta=quant_info.gemm1_beta,
        gemm1_clamp_limit=quant_info.gemm1_clamp_limit,
    )
    kernel_output = _unwrap_output(result, kernel_output)
    if kernel_output is not output:
        output.copy_(kernel_output[:, : output.shape[-1]])
    return output


def run_locality_partitioned_fp8_moe(
    *,
    shards: MoeLocalityShards,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    topk_output,
    use_routed_topk: bool,
    defer_finalize: bool,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 partitioned forward: MXFP8 block scales or per-tensor scales."""
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        prims_ts_fp8_block_scale_moe_partitioned,
        prims_ts_fp8_per_tensor_scale_moe_partitioned,
    )

    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    _require_finalize(defer_finalize, "FP8")
    num_tokens = hidden_states.shape[0]
    routing_method_type = int(quant_info.routing_method_type)

    if use_routed_topk:
        topk_ids, topk_weights = _routed_ids_and_weights(topk_output)
        routing_kwargs = dict(
            routing_logits=None,
            routing_bias=None,
            topk_ids=topk_ids,
            expert_weights=topk_weights,
            top_k=int(topk_ids.shape[1]),
            n_group=None,
            topk_group=None,
            routing_method_type=int(
                RoutingMethodType.TopK
                if routing_method_type == RoutingMethodType.DeepSeekV3
                else routing_method_type
            ),
        )
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)
        topk_config = topk_output.topk_config
        routing_kwargs = dict(
            routing_logits=topk_output.router_logits,
            routing_bias=topk_config.correction_bias,
            topk_ids=None,
            expert_weights=None,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            routing_method_type=routing_method_type,
        )

    common = dict(
        hidden_states=hidden_states,
        num_experts=quant_info.global_num_experts,
        intermediate_size=quant_info.intermediate_size,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        partition_resources=shards.resources,
        partition_strategy=shards.strategy,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        do_finalize=True,
        enable_pdl=_enable_pdl(num_tokens),
        tune_max_num_tokens=next_power_of_2(num_tokens),
        output=output,
    )
    if quant_info.activation_type is not None:
        common["activation_type"] = int(quant_info.activation_type)

    if quant_info.block_quant:
        if not quant_info.use_mxfp8:
            raise NotImplementedError(
                "--enable-moe-locality-partition: DeepSeek block FP8 has no "
                "partitioned path"
            )
        assert hidden_states_scale is not None
        result = prims_ts_fp8_block_scale_moe_partitioned(
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=list(shards.fc1),
            gemm1_weights_scale=list(shards.fc1_scale),
            gemm2_weights=list(shards.fc2),
            gemm2_weights_scale=list(shards.fc2_scale),
            use_shuffled_weight=True,
            weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            gemm1_alpha=quant_info.gemm1_alpha,
            gemm1_beta=quant_info.gemm1_beta,
            gemm1_clamp_limit=quant_info.gemm1_clamp_limit,
            **routing_kwargs,
            **common,
        )
        return _unwrap_output(result, output)

    if use_routed_topk:
        raise NotImplementedError(
            "--enable-moe-locality-partition: per-tensor FP8 requires logits routing"
        )
    routing_kwargs["routing_logits"] = routing_kwargs["routing_logits"].to(
        torch.bfloat16
    )
    result = prims_ts_fp8_per_tensor_scale_moe_partitioned(
        gemm1_weights=list(shards.fc1),
        output1_scales_scalar=quant_info.output1_scales_scalar,
        output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,
        gemm2_weights=list(shards.fc2),
        output2_scales_scalar=quant_info.output2_scales_scalar,
        use_routing_scales_on_input=False,
        weight_layout=_WEIGHT_LAYOUT_MAJOR_K,
        **routing_kwargs,
        **common,
    )
    return _unwrap_output(result, output)


def run_locality_partitioned_bf16_moe(
    *,
    shards: MoeLocalityShards,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    topk_output,
    use_routed_topk: bool,
    hidden_states: torch.Tensor,
    activation_type: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """BF16 (unquantized, BlockMajorK) partitioned forward."""
    from flashinfer.fused_moe import prims_ts_bf16_moe_partitioned

    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    num_tokens = hidden_states.shape[0]
    if use_routed_topk:
        topk_ids, topk_weights = _routed_ids_and_weights(topk_output)
        routing_method_type = runner_config.routing_method_type
        if routing_method_type is None:
            routing_method_type = RoutingMethodType.Default
        elif routing_method_type == RoutingMethodType.DeepSeekV3:
            routing_method_type = RoutingMethodType.TopK
        routing_kwargs = dict(
            routing_logits=None,
            routing_bias=None,
            topk_ids=topk_ids,
            expert_weights=topk_weights,
            top_k=int(topk_ids.shape[1]),
            n_group=None,
            topk_group=None,
            routing_method_type=int(routing_method_type),
            routed_scaling_factor=(
                runner_config.routed_scaling_factor
                if runner_config.routed_scaling_factor is not None
                else 1.0
            ),
        )
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)
        topk_config = topk_output.topk_config
        routing_method_type = runner_config.routing_method_type
        routing_kwargs = dict(
            routing_logits=topk_output.router_logits,
            routing_bias=topk_config.correction_bias,
            topk_ids=None,
            expert_weights=None,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            routing_method_type=int(
                routing_method_type
                if routing_method_type is not None
                else RoutingMethodType.Default
            ),
            routed_scaling_factor=runner_config.routed_scaling_factor,
        )

    result = prims_ts_bf16_moe_partitioned(
        hidden_states=hidden_states,
        gemm1_weights=list(shards.fc1),
        gemm2_weights=list(shards.fc2),
        num_experts=quant_info.global_num_experts,
        intermediate_size=runner_config.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
        partition_resources=shards.resources,
        partition_strategy=shards.strategy,
        use_shuffled_weight=True,
        weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
        do_finalize=True,
        enable_pdl=_enable_pdl(num_tokens),
        tune_max_num_tokens=next_power_of_2(num_tokens),
        activation_type=activation_type,
        output=output,
        **routing_kwargs,
    )
    return _unwrap_output(result, output)
