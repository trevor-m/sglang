"""MegaMoE locality-domain weight localization: bookkeeping, guard and EPLB.

deep_gemm.localize() is replaced by a CPU copy with the same (num_domains, *B,
N / num_domains, K) domain-major layout, so the repointing, the refusal paths,
the over-cap guard and EPLB's per-domain slabs run on CPU. The EPLB case moves
localized slabs with the real updater between two gloo ranks. The kernel itself
and physical placement are covered by the GPU test.
"""

import os
import socket
import sys
import traceback
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import sglang.srt.layers.moe.mega_moe as mega
import sglang.srt.runtime_context as runtime_context
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

D = 2  # locality domains
E = 4  # local experts
N1, K1 = 8, 6  # L1 rows (gate+up) and packed K bytes
N2, K2 = 6, 4  # L2 rows (hidden) and packed K bytes


def _domain_major(t: torch.Tensor) -> torch.Tensor:
    # deep_gemm.localize(t, dim=-2): (*B, N, K) -> (D, *B, N / D, K)
    return t.unflatten(-2, (D, -1)).movedim(-3, 0).contiguous()


def _logical(t: torch.Tensor) -> torch.Tensor:
    # Inverse of _domain_major
    return t.movedim(0, -3).flatten(-3, -2)


def _fake_deep_gemm(available=True, num_domains=D):
    dg = ModuleType("deep_gemm")
    dg.calls = []
    dg.localize = lambda t, dim=-2: dg.calls.append("localize") or _domain_major(t)
    dg.destroy_localizer = lambda: dg.calls.append("destroy")
    dg.is_localization_available = lambda: available
    dg.get_num_locality_domains = lambda: num_domains
    return dg


def _experts(seed=0):
    g = torch.Generator().manual_seed(seed)
    experts = nn.Module()

    def param(*shape, dtype=torch.int8):
        t = torch.randint(-100, 100, shape, generator=g).to(dtype)
        return nn.Parameter(t, requires_grad=False)

    experts.w13_weight = param(E, N1, K1)
    experts.w2_weight = param(E, N2, K2)
    experts.w13_weight_scale_inv = param(E, N1, 2, dtype=torch.int32)
    experts.w2_weight_scale_inv = param(E, N2, 2, dtype=torch.int32)
    experts.num_local_experts = E
    experts._mega_moe_weights_built = True
    experts._mega_moe_nvfp4 = False
    experts._mega_moe_w4a4 = False
    experts._mega_moe_trtllm_layout = False
    experts.mega_l1_weights = (
        experts.w13_weight.data,
        experts.w13_weight_scale_inv.data,
    )
    experts.mega_l2_weights = (experts.w2_weight.data, experts.w2_weight_scale_inv.data)
    return experts


def _moe(seed=0, shared=True):
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.experts = _experts(seed)
    moe.num_fused_shared_experts = 0
    moe.mega_shared_l1_weights = moe.mega_shared_l2_weights = None
    if shared:
        moe.mega_shared_l1_weights = (
            torch.arange(N1 * K1, dtype=torch.int8).view(N1, K1),
            torch.zeros(1),
        )
        moe.mega_shared_l2_weights = (
            torch.arange(N2 * K2, dtype=torch.int8).view(N2, K2),
            torch.zeros(1),
        )
    return moe


@pytest.fixture
def localizable(monkeypatch):
    state = SimpleNamespace(mode="decode", elastic=None, dg=_fake_deep_gemm())
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_WEIGHTS", "1")
    monkeypatch.setitem(sys.modules, "deep_gemm", state.dg)
    monkeypatch.setattr(mega, "_device_sm", 107)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runtime_context,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode=state.mode),
    )
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=state.elastic)),
    )
    return state


def test_localize_repoints_routed_and_shared(localizable):
    model = nn.ModuleList([_moe(0), _moe(1, shared=False)])
    before = [
        (
            m.experts.w13_weight.clone(),
            m.experts.w2_weight.clone(),
            m.mega_shared_l1_weights,
        )
        for m in model
    ]
    sfs = [(m.experts.mega_l1_weights[1], m.experts.mega_l2_weights[1]) for m in model]

    assert mega.localize_mega_moe_weights(model)

    assert localizable.dg.calls == ["localize"] * 6 + ["destroy"]
    for moe, (w13, w2, shared_l1), (sf1, sf2) in zip(model, before, sfs):
        experts = moe.experts
        assert mega.is_mega_moe_localized(experts)
        assert experts.w13_weight.shape == (D, E, N1 // D, K1)
        assert torch.equal(_logical(experts.w13_weight.data), w13)
        assert torch.equal(_logical(experts.w2_weight.data), w2)
        # The kernel's tuples name the parameters' storage, and keep their SFs.
        assert experts.mega_l1_weights[0].data_ptr() == experts.w13_weight.data_ptr()
        assert experts.mega_l2_weights[0].data_ptr() == experts.w2_weight.data_ptr()
        assert experts.mega_l1_weights[1] is sf1 and experts.mega_l2_weights[1] is sf2
        if shared_l1 is None:
            assert moe.mega_shared_l1_weights is None
        else:
            assert moe.mega_shared_l1_weights[0].shape == (D, N1 // D, K1)
            assert torch.equal(_logical(moe.mega_shared_l1_weights[0]), shared_l1[0])
            assert moe.mega_shared_l1_weights[1] is shared_l1[1]
    # Already-localized layers are skipped.
    assert not mega.localize_mega_moe_weights(model)


def test_get_moe_weights_exposes_domain_slabs(localizable):
    moe = _moe(0)
    plain = moe.get_moe_weights()
    assert [t.shape for t in plain] == [
        (E, N1, K1),
        (E, N2, K2),
        (E, N1, 2),
        (E, N2, 2),
    ]

    assert mega.localize_mega_moe_weights(nn.ModuleList([moe]))
    weights = moe.get_moe_weights()
    w13, w2 = moe.experts.w13_weight.data, moe.experts.w2_weight.data
    assert [t.shape for t in weights] == [
        (E, N1 // D, K1),
        (E, N1 // D, K1),
        (E, N2 // D, K2),
        (E, N2 // D, K2),
        (E, N1, 2),
        (E, N2, 2),
    ]
    for d in range(D):
        assert weights[d].data_ptr() == w13[d].data_ptr()
        assert weights[D + d].data_ptr() == w2[d].data_ptr()
    # Every expert's slice of a slab is contiguous, as NCCL/gloo P2P requires.
    assert all(t[j].is_contiguous() for t in weights for j in range(E))


@pytest.mark.parametrize(
    "case",
    ["env_off", "prefill", "elastic", "unavailable", "one_domain", "nvfp4", "odd_n"],
)
def test_refusals_leave_weights_untouched(localizable, monkeypatch, case, caplog):
    moe = _moe(0)
    if case == "env_off":
        monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_WEIGHTS", "0")
    elif case == "prefill":
        localizable.mode = "prefill"
    elif case == "elastic":
        localizable.elastic = "mooncake"
    elif case == "unavailable":
        monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm(available=False))
    elif case == "one_domain":
        monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm(num_domains=1))
    elif case == "nvfp4":
        moe.experts._mega_moe_nvfp4 = True
    elif case == "odd_n":
        moe.mega_shared_l2_weights = (torch.zeros(N2 + 1, K2, dtype=torch.int8), None)
    w13 = moe.experts.w13_weight.data

    assert not mega.localize_mega_moe_weights(nn.ModuleList([moe]))
    assert moe.experts.w13_weight.data_ptr() == w13.data_ptr()
    assert not mega.is_mega_moe_localized(moe.experts)
    assert "localize" not in localizable.dg.calls
    if case != "env_off":
        assert "not localizing" in caplog.text


def test_prefill_override_allows_localization(localizable, monkeypatch):
    localizable.mode = "null"
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_ANY_MODE", "1")
    assert mega.localize_mega_moe_weights(nn.ModuleList([_moe(0)]))


def test_over_cap_raises_only_when_localized(localizable, monkeypatch):
    tokens = SimpleNamespace(per_rank=None)
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK", "1024")
    monkeypatch.setattr(
        mega, "get_moe_a2a_backend", lambda: SimpleNamespace(is_megamoe=lambda: True)
    )
    monkeypatch.setattr(mega, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(mega, "get_dp_global_num_tokens", lambda: tokens.per_rank)
    monkeypatch.setattr(mega, "is_dsa_enable_prefill_cp", lambda: False)
    moe = _moe(0)
    hidden = torch.zeros(4, 8)

    tokens.per_rank = [10, 1024]
    assert mega.should_use_mega_moe(moe, hidden)
    tokens.per_rank = [10, 1025]
    assert not mega.should_use_mega_moe(moe, hidden)  # DeepEP fallback

    assert mega.localize_mega_moe_weights(nn.ModuleList([moe]))
    tokens.per_rank = [10, 1024]
    assert mega.should_use_mega_moe(moe, hidden)
    tokens.per_rank = [10, 1025]
    with pytest.raises(RuntimeError, match="cannot read the localized"):
        mega.should_use_mega_moe(moe, hidden)


# ------------------------------------------------------------------------------------
# EPLB: the real updater moves the per-domain slabs of localized weights (2 gloo ranks)

WORLD = 2
NUM_PHYSICAL = WORLD * E
NUM_LOGICAL = 6
ROUNDS = 20


def _expert_values(param_idx, logical, shape):
    base = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.int32)
    return (base.reshape(shape) + logical * 37 + param_idx * 11).to(torch.int8)


def _random_map(gen):
    extra = torch.randint(0, NUM_LOGICAL, (NUM_PHYSICAL - NUM_LOGICAL,), generator=gen)
    ans = torch.cat([torch.arange(NUM_LOGICAL), extra])
    return ans[torch.randperm(NUM_PHYSICAL, generator=gen)]


def _logical_params(moe):
    # Every routed parameter in its logical (E, N, K) view
    experts = moe.experts
    return [
        _logical(experts.w13_weight.data),
        _logical(experts.w2_weight.data),
        experts.w13_weight_scale_inv.data,
        experts.w2_weight_scale_inv.data,
    ]


def _eplb_worker(rank, port, result_q):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        dist.init_process_group(
            "gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=60)
        )
        from sglang.srt.eplb.expert_location_updater import (
            create_temp_buffers,
            update_expert_weights_single_layer,
        )

        moe = _moe(rank, shared=False)
        experts = moe.experts
        # Localize the way localize_mega_moe_weights does (deep_gemm.localize layout)
        experts.w13_weight.data = _domain_major(experts.w13_weight.data)
        experts.w2_weight.data = _domain_major(experts.w2_weight.data)
        experts._mega_moe_localized_params = ("w13_weight", "w2_weight")
        storage = [p.data_ptr() for p in experts.parameters()]

        gen = torch.Generator().manual_seed(0)  # the same maps on every rank
        old_map = _random_map(gen)
        for p_idx, param in enumerate(_logical_params(moe)):
            for j in range(E):
                logical = int(old_map[rank * E + j])
                want = _expert_values(p_idx, logical, param.shape[1:]).to(param.dtype)
                if p_idx < 2:  # write through the domain-major storage
                    src = want.unflatten(0, (D, -1))
                    for d in range(D):
                        getattr(experts, ("w13_weight", "w2_weight")[p_idx]).data[
                            d, j
                        ].copy_(src[d])
                else:
                    param[j].copy_(want)

        for _ in range(ROUNDS):
            new_map = _random_map(gen)
            weights = moe.get_moe_weights()
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=create_temp_buffers(weights),
                old_physical_to_logical_map=old_map.tolist(),
                new_physical_to_logical_map=new_map.tolist(),
                num_local_physical_experts=E,
                num_gpu_per_node=WORLD,
                rank=rank,
                world_size=WORLD,
            )
            for p_idx, param in enumerate(_logical_params(moe)):
                for j in range(E):
                    logical = int(new_map[rank * E + j])
                    want = _expert_values(p_idx, logical, param.shape[1:]).to(
                        param.dtype
                    )
                    assert torch.equal(param[j], want), (
                        f"rank {rank} param {p_idx} slot {j}: want logical {logical}"
                    )
            assert [p.data_ptr() for p in experts.parameters()] == storage
            old_map = new_map
        dist.barrier()
        dist.destroy_process_group()
        result_q.put((rank, None))
    except BaseException as e:
        result_q.put((rank, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        result_q.close()
        result_q.join_thread()
        os._exit(1)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_eplb_moves_localized_slabs():
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_eplb_worker, args=(rank, port, result_q))
        for rank in range(WORLD)
    ]
    for p in procs:
        p.start()
    errors = []
    for _ in range(WORLD):
        rank, err = result_q.get(timeout=180)
        if err is not None:
            errors.append(f"rank {rank}: {err}")
            break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    assert errors == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
