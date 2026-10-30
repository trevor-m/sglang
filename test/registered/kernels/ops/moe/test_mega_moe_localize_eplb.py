"""MegaMoE with locality-domain weights, before and after EPLB rebalances.

Each rank runs DeepGEMM fp8_fp4_mega_moe on two copies of the same experts at
DeepSeek-V4.1-Flash shapes (hidden 5120, intermediate 2304, top-6, one fused
shared expert, redundant expert slots): ordinary weights, and weights moved into
the locality domains by localize_mega_moe_weights(). Then the real EPLB updater
remaps the physical experts a few times, using get_moe_weights() for both
copies, and every token keeps its logical experts. Checks, at every step:
  - localized output == ordinary output (bitwise: same schedule, other memory);
  - output == the output before any rebalance (bitwise: EPLB moved the right
    bytes, so each token still meets the same logical experts);
  - the localized weights keep their storage and stay localized.
Skips without 2 GPUs or without localized memory on this GPU and driver.
"""

import hashlib
import os
import socket
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

H, I = 5120, 2304
TOPK = 6
NUM_LOGICAL = 192
REDUNDANT_PER_RANK = 1
MAX_TOKENS = 1024
TOKENS = (768, 330, 1)
ROUNDS = 3


def _localization_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        callable(getattr(deep_gemm, "localize", None))
        and deep_gemm.is_localization_available()
        and deep_gemm.get_num_locality_domains() > 1
    )


def _seed(*key) -> int:
    return int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:8], 16)


def _worker(local_rank, num_ranks, port, result_q):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        # MegaMoE's MN-major scale slices are strided: keep the nccl backend
        os.environ.setdefault("TORCH_DIST_USE_NCCL2", "0")
        os.environ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_WEIGHTS"] = "1"
        import deep_gemm as dg
        import torch.distributed as dist
        from deep_gemm import utils
        from deep_gemm.utils.dist import init_dist
        from torch import nn

        # init_dist sets the device (and makes CUDA the default) before SGLang
        # imports touch CUDA; CPU generators below name their device.
        rank, world, group = init_dist(local_rank, num_ranks)

        import sglang.srt.layers.moe.mega_moe as mega
        import sglang.srt.runtime_context as runtime_context
        from sglang.srt.eplb.expert_location_updater import (
            create_temp_buffers,
            update_expert_weights_single_layer,
        )
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        cpu_group = dist.new_group(list(range(world)), backend="gloo")
        runtime_context.get_disagg = lambda: SimpleNamespace(
            disaggregation_mode="decode"
        )
        runtime_context.get_exec = lambda: SimpleNamespace(
            moe=SimpleNamespace(elastic_ep_backend=None)
        )
        runtime_context.get_parallel = lambda: SimpleNamespace(
            moe_ep_group=SimpleNamespace(world_size=world, cpu_group=cpu_group)
        )

        epr = NUM_LOGICAL // world + REDUNDANT_PER_RANK
        num_physical = epr * world

        def random_map(gen):
            extra = torch.randint(
                0,
                NUM_LOGICAL,
                (num_physical - NUM_LOGICAL,),
                generator=gen,
                device="cpu",
            )
            ans = torch.cat([torch.arange(NUM_LOGICAL, device="cpu"), extra])
            return ans[
                torch.randperm(num_physical, generator=gen, device="cpu")
            ].tolist()

        def cast_fp4(w):
            q, sf = utils.per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
            return q, sf

        def logical_expert(e):
            g = torch.Generator(device="cuda").manual_seed(_seed("w", e))
            w1 = torch.randn(
                (2 * I, H), generator=g, device="cuda", dtype=torch.bfloat16
            )
            w2 = torch.randn((H, I), generator=g, device="cuda", dtype=torch.bfloat16)
            return cast_fp4(w1), cast_fp4(w2)

        def routed_weights(phys_map):
            local = [logical_expert(phys_map[rank * epr + j]) for j in range(epr)]
            l1 = torch.stack([x[0][0] for x in local])
            l1_sf = torch.stack([x[0][1] for x in local])
            l2 = torch.stack([x[1][0] for x in local])
            l2_sf = torch.stack([x[1][1] for x in local])
            l1_sf = dg.transform_sf_into_required_layout(l1_sf, 2 * I, H, (1, 32), epr)
            l2_sf = dg.transform_sf_into_required_layout(l2_sf, H, I, (1, 32), epr)
            return dg.transform_weights_for_mega_moe(
                (l1, l1_sf), (l2, l2_sf), mma_type="fp8xfp4"
            )

        def cast_fp8_shared(w):
            x, sf = utils.per_token_cast_to_fp8(
                w, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            mn, packed_k = sf.shape
            sf_tma = torch.empty_strided(
                (mn, packed_k), (1, utils.align(mn, 4)), dtype=sf.dtype, device="cuda"
            )
            sf_tma.copy_(sf)
            return x, sf_tma

        g = torch.Generator(device="cuda").manual_seed(_seed("shared"))
        shared = dg.transform_weights_for_mega_moe(
            cast_fp8_shared(
                torch.randn(
                    (2 * I, H), generator=g, device="cuda", dtype=torch.bfloat16
                )
            ),
            cast_fp8_shared(
                torch.randn((H, I), generator=g, device="cuda", dtype=torch.bfloat16)
            ),
            mma_type="fp8xfp4",
        )

        def make_moe(l1, l2):
            experts = nn.Module()
            experts.w13_weight = nn.Parameter(l1[0].clone(), requires_grad=False)
            experts.w2_weight = nn.Parameter(l2[0].clone(), requires_grad=False)
            experts.mega_w13_weight_scale_utccp = nn.Parameter(
                l1[1].clone(), requires_grad=False
            )
            experts.mega_w2_weight_scale_utccp = nn.Parameter(
                l2[1].clone(), requires_grad=False
            )
            experts.num_local_experts = epr
            experts._mega_moe_weights_built = True
            experts._mega_moe_nvfp4 = experts._mega_moe_w4a4 = False
            experts._mega_moe_trtllm_layout = False
            experts.mega_l1_weights = (
                experts.w13_weight.data,
                experts.mega_w13_weight_scale_utccp.data,
            )
            experts.mega_l2_weights = (
                experts.w2_weight.data,
                experts.mega_w2_weight_scale_utccp.data,
            )
            moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
            nn.Module.__init__(moe)
            moe.experts = experts
            moe.num_fused_shared_experts = 0
            moe.mega_shared_l1_weights = (shared[0][0].clone(), shared[0][1])
            moe.mega_shared_l2_weights = (shared[1][0].clone(), shared[1][1])
            return moe

        gen = torch.Generator(device="cpu").manual_seed(
            _seed("maps")
        )  # the same maps on every rank
        phys_map = random_map(gen)
        l1, l2 = routed_weights(phys_map)
        plain, localized = make_moe(l1, l2), make_moe(l1, l2)
        del l1, l2
        assert mega.localize_mega_moe_weights(nn.ModuleList([localized]))
        assert mega.is_mega_moe_localized(localized.experts)
        loc = localized.experts
        assert dg.is_localized(loc.mega_l1_weights[0]) and dg.is_localized(
            loc.mega_l2_weights[0]
        )
        assert dg.is_localized(localized.mega_shared_l1_weights[0])
        storage = (loc.w13_weight.data_ptr(), loc.w2_weight.data_ptr())

        buffer = dg.get_symm_buffer_for_mega_moe(
            group,
            num_physical,
            MAX_TOKENS,
            TOPK,
            H,
            I,
            num_shared_experts=1,
            mma_type="fp8xfp4",
        )

        def shared_sf_layout(sf, block_m):
            rows = buffer.shared_l1_acts_sf.shape[0]
            num_tokens, packed_k = sf.shape
            out = torch.empty_strided(
                (rows, packed_k), (1, rows), dtype=sf.dtype, device="cuda"
            )
            out.zero_()
            m = torch.arange(num_tokens, device="cuda")
            block, m_in = m // block_m, m % block_m
            dst = (
                block * utils.align(block_m, 128)
                + (m_in // 128) * 128
                + (m_in % 32) * 4
                + (m_in % 128) // 32
            )
            out[dst] = sf
            return out

        def inputs(T):
            cg = torch.Generator().manual_seed(_seed("x", rank, T))
            x = torch.randn(
                (max(T, 1), H), generator=cg, dtype=torch.bfloat16, device="cpu"
            ).cuda()[:T]
            x_fp8, x_sf = utils.per_token_cast_to_fp8(
                x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            scores = torch.randn((max(T, 1), NUM_LOGICAL), generator=cg, device="cpu")[
                :T
            ]
            topk_w, topk_logical = torch.topk(scores, TOPK, dim=-1, sorted=False)
            return x_fp8, x_sf, topk_logical, topk_w.float().cuda()

        def to_physical(topk_logical, phys_map):
            replicas = [[] for _ in range(NUM_LOGICAL)]
            for p, e in enumerate(phys_map):
                replicas[e].append(p)
            t = torch.arange(topk_logical.shape[0], device="cpu").unsqueeze(1)
            out = torch.empty_like(topk_logical)
            for e in range(NUM_LOGICAL):
                mask = topk_logical == e
                if mask.any():
                    r = torch.tensor(replicas[e], device="cpu")
                    out[mask] = r[(t.expand_as(topk_logical)[mask] + e) % len(r)]
            return out.cuda()

        def run(moe, inp, phys_map):
            x_fp8, x_sf, topk_logical, topk_w = inp
            T = x_fp8.shape[0]
            bm = dg.get_block_m_for_mega_moe(
                world,
                num_physical,
                buffer.num_max_tokens_per_rank,
                max(T, 1),
                TOPK,
                "fp8xfp4",
            )
            if T:
                buffer.x[:T].copy_(x_fp8)
                buffer.x_sf[:T].copy_(x_sf)
                buffer.topk_idx[:T].copy_(to_physical(topk_logical, phys_map))
                buffer.topk_weights[:T].copy_(topk_w.to(buffer.topk_weights.dtype))
            buffer.shared_l1_acts_sf.copy_(shared_sf_layout(x_sf, bm))
            y = torch.zeros((max(T, 1), H), dtype=torch.bfloat16, device="cuda")
            dg.fp8_fp4_mega_moe(
                y[:T],
                moe.experts.mega_l1_weights,
                moe.experts.mega_l2_weights,
                buffer,
                shared_l1_weights=moe.mega_shared_l1_weights,
                shared_l2_weights=moe.mega_shared_l2_weights,
                recipe=(1, 1, 32),
                activation="swiglu",
                activation_clamp=10.0,
                fast_math=True,
            )
            torch.cuda.synchronize()
            return y[:T].clone()

        def rebalance(moe, old_map, new_map):
            weights = moe.get_moe_weights()
            assert len(weights) == (6 if moe is localized else 4)
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=create_temp_buffers(weights),
                old_physical_to_logical_map=old_map,
                new_physical_to_logical_map=new_map,
                num_local_physical_experts=epr,
                num_gpu_per_node=world,
                rank=rank,
                world_size=world,
            )
            torch.cuda.synchronize()

        def check(tag, got, want):
            ok = torch.equal(got, want)
            flag = torch.tensor([0 if ok else 1], device="cuda")
            dist.all_reduce(flag)
            if not ok:
                rel = (
                    (got.float() - want.float()).norm()
                    / want.float().norm().clamp(min=1e-9)
                ).item()
                raise AssertionError(
                    f"rank {rank} {tag}: not bitwise equal (rel L2 {rel:.3e})"
                )
            assert flag.item() == 0, f"{tag}: another rank differs"

        reference = {}
        for T in TOKENS:
            inp = inputs(T)
            reference[T] = run(plain, inp, phys_map)
            check(
                f"T={T} round 0 localized vs plain",
                run(localized, inp, phys_map),
                reference[T],
            )
        for r in range(1, ROUNDS + 1):
            new_map = random_map(gen)
            rebalance(plain, phys_map, new_map)
            rebalance(localized, phys_map, new_map)
            phys_map = new_map
            assert (loc.w13_weight.data_ptr(), loc.w2_weight.data_ptr()) == storage
            assert dg.is_localized(loc.mega_l1_weights[0]) and dg.is_localized(
                loc.mega_l2_weights[0]
            )
            for T in TOKENS:
                inp = inputs(T)
                check(
                    f"T={T} round {r} plain vs before",
                    run(plain, inp, phys_map),
                    reference[T],
                )
                check(
                    f"T={T} round {r} localized vs before",
                    run(localized, inp, phys_map),
                    reference[T],
                )
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        result_q.put((rank, None))
    except BaseException as e:
        result_q.put((local_rank, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        result_q.close()
        result_q.join_thread()
        os._exit(1)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not _localization_ready(),
    reason="Needs 2+ SM100-family GPUs with localized memory (sgl-deep-gemm >= 0.2.5)",
)
def test_localized_megamoe_survives_eplb():
    import torch.multiprocessing as mp

    num_ranks = min(4, torch.cuda.device_count())
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, num_ranks, port, result_q))
        for r in range(num_ranks)
    ]
    for p in procs:
        p.start()
    errors = []
    for _ in range(num_ranks):
        rank, err = result_q.get(timeout=900)
        if err is not None:
            errors.append(f"rank {rank}: {err}")
            break
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    assert errors == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
