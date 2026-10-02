"""CPU unit tests for srt/layers/moe/moe_runner/flashinfer_trtllm_locality.py.

FlashInfer's locality-domain pools, green-context streams and partitioned ops
are a driver boundary; they are replaced by a CPU model that allocates per
domain and slices prepared tensors the way ``tp_n_tp_n`` does. The assertions
are about SGLang's side of the contract: shards land in their own domain via
the pool allocator, the full-size tensors are released afterwards, the handle
is process-wide, and unsupported layouts are refused before any kernel runs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant before flashinfer_trtllm (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant  # noqa: F401
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp4MoeQuantInfo,
)

# isort: on
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm_locality as locality
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig
from sglang.srt.runtime_context import get_context, get_resources, reset_context
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 128
PARTITION_COUNT = 2


# ---------------------------------------------------------------------------
# CPU stand-ins for the FlashInfer locality boundary
# ---------------------------------------------------------------------------


class _FakeLocalizedDomains:
    """Allocates per domain and remembers which domain owns each storage."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.domain_count = PARTITION_COUNT
        self.sm_counts = (70, 70)
        self.remainder_sm_count = 0
        self.streams = ("green-stream-0", "green-stream-1")
        self._owner: dict[int, int] = {}
        self.alloc_calls: list[tuple[int, tuple[int, ...], torch.dtype]] = []

    def empty(self, domain, shape, dtype):
        tensor = torch.empty(tuple(shape), dtype=dtype)
        self._owner[tensor.untyped_storage().data_ptr()] = domain
        self.alloc_calls.append((domain, tuple(shape), dtype))
        return tensor

    def pointer_domain(self, tensor):
        return self._owner.get(tensor.untyped_storage().data_ptr(), -1)


class _FakePartitionResources:
    def __init__(self, streams):
        self.streams = tuple(streams)


def _fake_prepared_weight_layouts(**kwargs):
    layout = SimpleNamespace(
        kind="block_major_k" if kwargs["weight_layout"] == 2 else "major_k",
        **kwargs,
    )
    return layout, layout


def _fake_prepared_block_major_k_bytes(weight, weight_layout):
    if weight_layout != 2:
        return None
    return int(weight.shape[-1]) * weight.element_size()


def _fake_shard_prepared_moe_weights(
    fc1_weight,
    fc2_weight,
    *,
    strategy,
    hidden_size,
    intermediate_size,
    fc1_layout,
    fc2_layout,
    fc1_scale=None,
    fc2_scale=None,
    alloc=None,
):
    assert strategy == "tp_n_tp_n"
    rows_dim = 2 if fc1_layout.kind == "block_major_k" else 1
    num_experts = fc1_weight.shape[0]

    def place(partition_id, view):
        target = alloc(partition_id, tuple(view.shape), view.dtype)
        target.copy_(view)
        return target

    def row_shard(tensor, partition_id, rows):
        return tensor.narrow(rows_dim, partition_id * rows, rows)

    def scale_shard(scale, partition_id, full_rows, rows):
        if scale is None:
            return None
        per_row = scale.numel() // (num_experts * full_rows)
        view = scale.reshape(num_experts, full_rows, per_row)
        return view[:, partition_id * rows : (partition_id + 1) * rows].reshape(-1)

    shards = SimpleNamespace(fc1=[], fc2=[], fc1_scale=[], fc2_scale=[])
    for partition_id in range(PARTITION_COUNT):
        shards.fc1.append(
            place(partition_id, row_shard(fc1_weight, partition_id, intermediate_size))
        )
        shards.fc2.append(
            place(partition_id, row_shard(fc2_weight, partition_id, hidden_size // 2))
        )
        fc1_scale_shard = scale_shard(
            fc1_scale, partition_id, 2 * intermediate_size, intermediate_size
        )
        fc2_scale_shard = scale_shard(
            fc2_scale, partition_id, hidden_size, hidden_size // 2
        )
        if fc1_scale_shard is not None:
            shards.fc1_scale.append(place(partition_id, fc1_scale_shard))
        if fc2_scale_shard is not None:
            shards.fc2_scale.append(place(partition_id, fc2_scale_shard))
    if fc1_scale is None:
        shards.fc1_scale = None
    if fc2_scale is None:
        shards.fc2_scale = None
    return shards


class _FakeFlashInfer:
    """``sys.modules`` entries for the FlashInfer pieces the module imports lazily."""

    def __init__(self):
        self.domains = _FakeLocalizedDomains()
        self.partitioned_calls: list[dict] = []

        locality_domain = types.ModuleType("flashinfer.locality_domain")
        locality_domain.LocalityDomainError = type(
            "LocalityDomainError", (RuntimeError,), {}
        )
        locality_domain.get_localized_domains = mock.Mock(return_value=self.domains)
        self.get_localized_domains = locality_domain.get_localized_domains

        partition = types.ModuleType("flashinfer.prims_ts.moe.partition")
        partition.MoePartitionResources = _FakePartitionResources
        partition.prepared_weight_layouts = _fake_prepared_weight_layouts
        partition.prepared_block_major_k_bytes = _fake_prepared_block_major_k_bytes
        partition.shard_prepared_moe_weights = _fake_shard_prepared_moe_weights

        fused_moe = types.ModuleType("flashinfer.fused_moe")

        def partitioned_op(**kwargs):
            self.partitioned_calls.append(kwargs)
            if kwargs["do_finalize"]:
                return [kwargs["output"]]
            # Unfinalized ABI: permuted GEMM2 output, routing weights, indices.
            num_tokens = kwargs["hidden_states"].shape[0]
            return [
                torch.zeros(num_tokens * kwargs["top_k"], HIDDEN, dtype=torch.bfloat16),
                torch.ones(num_tokens, kwargs["top_k"], dtype=torch.bfloat16),
                torch.zeros(num_tokens, kwargs["top_k"], dtype=torch.int32),
            ]

        fused_moe.prims_ts_fp4_block_scale_moe_partitioned = partitioned_op

        self.modules = {
            "flashinfer": types.ModuleType("flashinfer"),
            "flashinfer.locality_domain": locality_domain,
            "flashinfer.prims_ts": types.ModuleType("flashinfer.prims_ts"),
            "flashinfer.prims_ts.moe": types.ModuleType("flashinfer.prims_ts.moe"),
            "flashinfer.prims_ts.moe.partition": partition,
            "flashinfer.fused_moe": fused_moe,
        }


def _param(tensor):
    return torch.nn.Parameter(tensor, requires_grad=False)


def _nvfp4_trtllm_layer(*, activation="silu", is_gated=True):
    """A layer as align_fp4_moe_weights_for_flashinfer_trtllm leaves it."""
    return SimpleNamespace(
        moe_runner_config=MoeRunnerConfig(activation=activation, is_gated=is_gated),
        moe_locality_shards=None,
        w13_weight=_param(
            torch.randint(
                0, 255, (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8
            )
        ),
        w2_weight=_param(
            torch.randint(
                0, 255, (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8
            )
        ),
        w13_weight_scale=_param(
            torch.randint(
                0, 255, (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
        ),
        w2_weight_scale=_param(
            torch.randint(
                0, 255, (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
        ),
    )


class _LocalityTestCase(CustomTestCase):
    def setUp(self):
        self.fi = _FakeFlashInfer()
        self._modules = mock.patch.dict(sys.modules, self.fi.modules)
        self._modules.start()
        self._config = get_context().override_server_args(
            enable_moe_locality_partition=True,
            moe_runner_backend="flashinfer_trtllm",
        )
        self._config.install()

    def tearDown(self):
        self._config.restore()
        self._modules.stop()
        reset_context()


class TestMoeLocalityWeightSharding(_LocalityTestCase):
    def test_disabled_without_published_config(self):
        """Bare unit tests and non-publishing entries never see the flag as on."""
        self._config.restore()
        reset_context()
        self.assertFalse(locality.is_moe_locality_partition_enabled())
        layer = _nvfp4_trtllm_layer()
        before = layer.w13_weight.data
        locality.partition_fp4_moe_weights_for_locality(layer, scale_k_group_size=16)
        self.assertIsNone(layer.moe_locality_shards)
        self.assertIs(layer.w13_weight.data, before)

    def test_fp4_shards_land_in_their_domain_and_full_weights_are_released(self):
        layer = _nvfp4_trtllm_layer()
        full = {
            name: getattr(layer, name).data.clone()
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
        }

        locality.partition_fp4_moe_weights_for_locality(layer, scale_k_group_size=16)

        shards = layer.moe_locality_shards
        self.assertEqual(shards.strategy, "tp_n_tp_n")
        for partition_id in range(PARTITION_COUNT):
            # Each shard was allocated through the domain's pool allocator.
            for tensor in (
                shards.fc1[partition_id],
                shards.fc2[partition_id],
                shards.fc1_scale[partition_id],
                shards.fc2_scale[partition_id],
            ):
                self.assertEqual(self.fi.domains.pointer_domain(tensor), partition_id)
            # tp_n_tp_n: FC1 rows split at I, FC2 rows split at H/2.
            torch.testing.assert_close(
                shards.fc1[partition_id],
                full["w13_weight"][
                    :, partition_id * INTERMEDIATE : (partition_id + 1) * INTERMEDIATE
                ],
            )
            torch.testing.assert_close(
                shards.fc2[partition_id],
                full["w2_weight"][
                    :, partition_id * (HIDDEN // 2) : (partition_id + 1) * (HIDDEN // 2)
                ],
            )
            self.assertEqual(shards.fc1_scale[partition_id].dtype, torch.float8_e4m3fn)

        # No second copy: the full-size parameters keep their shape for size
        # readers but hold a single element, and a reload write fails loudly.
        for name, tensor in full.items():
            param = getattr(layer, name)
            self.assertEqual(tuple(param.shape), tuple(tensor.shape))
            self.assertEqual(param.dtype, tensor.dtype)
            self.assertEqual(
                param.data.untyped_storage().nbytes(), tensor.element_size()
            )
            with self.assertRaises(RuntimeError):
                param.data.copy_(tensor)

    def test_scale_attrs_select_which_scale_params_are_sharded_and_released(self):
        """FP8 checkpoints with MXFP4 experts keep scales under *_weight_scale_inv;
        sharding must read and release exactly the named pair."""
        layer = _nvfp4_trtllm_layer()
        layer.w13_weight_scale_inv = layer.w13_weight_scale
        layer.w2_weight_scale_inv = layer.w2_weight_scale
        untouched = _param(torch.zeros(2, 2, dtype=torch.float8_e4m3fn))
        layer.w13_weight_scale = untouched
        layer.w2_weight_scale = untouched

        locality.partition_fp4_moe_weights_for_locality(
            layer,
            scale_k_group_size=32,
            scale_attrs=("w13_weight_scale_inv", "w2_weight_scale_inv"),
        )

        shards = layer.moe_locality_shards
        self.assertEqual(len(shards.fc1_scale), PARTITION_COUNT)
        self.assertEqual(layer.w13_weight_scale_inv.data.untyped_storage().nbytes(), 1)
        self.assertIs(layer.w13_weight_scale, untouched)
        self.assertEqual(untouched.data.untyped_storage().nbytes(), 4)

    def test_partition_handle_is_process_wide(self):
        """Two layers share one LocalizedDomains / MoePartitionResources pair.

        CUDA graph replay requires every partitioned layer to fork/join through
        the same long-lived events, and the pools are process-lifetime.
        """
        first = _nvfp4_trtllm_layer()
        second = _nvfp4_trtllm_layer()
        locality.partition_fp4_moe_weights_for_locality(first, scale_k_group_size=16)
        locality.partition_fp4_moe_weights_for_locality(second, scale_k_group_size=16)

        self.assertIs(
            first.moe_locality_shards.partition, second.moe_locality_shards.partition
        )
        self.assertIs(
            get_resources().moe_locality_partition, first.moe_locality_shards.partition
        )
        self.fi.get_localized_domains.assert_called_once()
        self.assertEqual(
            first.moe_locality_shards.resources.streams, self.fi.domains.streams
        )
        # Second call on an already-sharded layer is a no-op.
        shards = first.moe_locality_shards
        locality.partition_fp4_moe_weights_for_locality(first, scale_k_group_size=16)
        self.assertIs(first.moe_locality_shards, shards)

    def test_sm_split_is_forwarded_to_flashinfer(self):
        self._config.restore()
        self._config = get_context().override_server_args(
            enable_moe_locality_partition=True,
            moe_runner_backend="flashinfer_trtllm",
            moe_locality_sm_split="strict",
        )
        self._config.install()
        locality.partition_fp4_moe_weights_for_locality(
            _nvfp4_trtllm_layer(), scale_k_group_size=16
        )
        _, kwargs = self.fi.get_localized_domains.call_args
        self.assertEqual(kwargs["sm_split"], "strict")

    def test_unsupported_layouts_are_refused_before_sharding(self):
        """Prims-TS partitioning has no non-gated, SiTU, biased or DeepSeek-FP8 path."""
        for layer in (
            _nvfp4_trtllm_layer(activation="relu2", is_gated=False),
            _nvfp4_trtllm_layer(activation="situ"),
        ):
            with self.assertRaises(NotImplementedError):
                locality.partition_fp4_moe_weights_for_locality(
                    layer, scale_k_group_size=16
                )
            self.assertIsNone(layer.moe_locality_shards)

        biased = _nvfp4_trtllm_layer()
        biased.w13_weight_bias = _param(torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE))
        biased.w2_weight_bias = _param(torch.zeros(NUM_EXPERTS, HIDDEN))
        biased.w2_weight_bias.data[0, 0] = 1.0
        with self.assertRaises(NotImplementedError):
            locality.partition_mxfp4_trtllm_gen_moe_weights_for_locality(biased)
        self.assertIsNone(biased.moe_locality_shards)

        zero_bias = _nvfp4_trtllm_layer()
        zero_bias.w13_weight_bias = _param(torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE))
        zero_bias.w2_weight_bias = _param(torch.zeros(NUM_EXPERTS, HIDDEN))
        locality.partition_mxfp4_trtllm_gen_moe_weights_for_locality(zero_bias)
        self.assertIsNotNone(zero_bias.moe_locality_shards)

        with self.assertRaises(NotImplementedError):
            locality.reject_deepseek_fp8_block_locality_partition()
        self.assertEqual(self.fi.get_localized_domains.call_count, 1)


class TestMoeLocalityPartitionedForward(_LocalityTestCase):
    def _shards(self):
        layer = _nvfp4_trtllm_layer()
        locality.partition_fp4_moe_weights_for_locality(layer, scale_k_group_size=16)
        return layer.moe_locality_shards

    def _quant_info(self, shards):
        ones = torch.ones(NUM_EXPERTS, dtype=torch.float32)
        return FlashInferTrtllmFp4MoeQuantInfo(
            w13_weight=torch.empty(0),
            w2_weight=torch.empty(0),
            w13_weight_scale=torch.empty(0),
            w2_weight_scale=torch.empty(0),
            g1_scale_c=ones,
            g1_alphas=ones * 2,
            g2_alphas=ones * 3,
            w13_input_scale_quant=torch.ones(()),
            global_num_experts=NUM_EXPERTS,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            intermediate_size_per_partition=INTERMEDIATE,
            routing_method_type=2,
            locality_shards=shards,
        )

    def _run(self, *, defer_finalize=False, per_token_scale=None):
        shards = self._shards()
        num_tokens = 3
        hidden_states = torch.zeros(num_tokens, HIDDEN // 2, dtype=torch.uint8)
        output = torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16)
        topk_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=torch.zeros(num_tokens, NUM_EXPERTS),
            topk_config=TopKConfig(top_k=2, num_expert_group=1, topk_group=1),
        )
        with mock.patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm.trtllm_moe_enable_pdl",
            return_value=False,
        ):
            result = locality.run_locality_partitioned_fp4_moe(
                shards=shards,
                quant_info=self._quant_info(shards),
                runner_config=MoeRunnerConfig(routed_scaling_factor=1.5),
                topk_output=topk_output,
                use_routed_topk=False,
                defer_finalize=defer_finalize,
                hidden_states=hidden_states,
                hidden_states_scale=torch.zeros(
                    num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn
                ),
                per_token_scale=per_token_scale,
                activation_type=3,
                output=output,
            )
        return shards, output, result

    def test_fp4_forward_passes_both_shards_and_the_shared_handle(self):
        shards, output, result = self._run()

        self.assertIs(result, output)
        (call,) = self.fi.partitioned_calls
        self.assertEqual(call["partition_strategy"], "tp_n_tp_n")
        self.assertIs(call["partition_resources"], shards.resources)
        self.assertIs(call["output"], output)
        self.assertTrue(call["do_finalize"])
        for key, expected in (
            ("gemm1_weights", shards.fc1),
            ("gemm1_weights_scale", shards.fc1_scale),
            ("gemm2_weights", shards.fc2),
            ("gemm2_weights_scale", shards.fc2_scale),
        ):
            self.assertEqual(len(call[key]), PARTITION_COUNT)
            for actual, want in zip(call[key], expected):
                self.assertIs(actual, want)
        # Logits routing keeps the layer's routing method and scaling factor.
        self.assertEqual(call["routing_method_type"], 2)
        self.assertEqual(call["routed_scaling_factor"], 1.5)
        self.assertEqual(call["top_k"], 2)
        self.assertIsNone(call["topk_ids"])

    def test_fp4_deferred_finalize_returns_the_unfinalized_triple(self):
        """DeepSeek-style fused finalize (shared-expert add) runs the MoE with
        do_finalize=False and consumes trtllm-gen's triple unchanged."""
        _, _, result = self._run(defer_finalize=True)

        (call,) = self.fi.partitioned_calls
        self.assertFalse(call["do_finalize"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].dtype, torch.bfloat16)
        self.assertEqual(result[2].dtype, torch.int32)

    def test_fp4_forward_refuses_per_token_scale(self):
        with self.assertRaises(NotImplementedError):
            self._run(per_token_scale=torch.ones(3))
        self.assertEqual(self.fi.partitioned_calls, [])


if __name__ == "__main__":
    unittest.main()
