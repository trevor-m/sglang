from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import numpy.typing as npt
import requests
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

GUARD = "NixlMsgGuard".encode("ascii")


class ReceiveStagingBuffer:
    """Pre-allocated GPU staging buffer for receiving NIXL transfers on decode side."""

    def __init__(self, size_mb: int, kv_args: KVArgs, kv_buffers: list, device: int):
        import torch

        num_kv_ptrs = len(kv_args.kv_data_ptrs)
        item_len = kv_args.kv_item_lens[0]  # uniform across layers
        page_size = kv_args.page_size

        total_bytes = size_mb * 1024 * 1024
        bytes_per_ptr = total_bytes // num_kv_ptrs
        self.max_staging_pages = bytes_per_ptr // item_len
        self.page_size = page_size
        self.item_len = item_len
        self.num_kv_ptrs = num_kv_ptrs

        logger.info(
            f"Receive staging buffer: {size_mb}MB, {num_kv_ptrs} ptrs, "
            f"{self.max_staging_pages} pages/ptr, item_len={item_len}"
        )

        # Per-layer staging tensors matching KV buffer shapes (smaller first dim)
        num_staging_tokens = self.max_staging_pages * page_size
        self.staging_layers = []
        for kv_buf in kv_buffers:
            staging = torch.empty(
                num_staging_tokens,
                *kv_buf.shape[1:],
                dtype=kv_buf.dtype,
                device=f"cuda:{device}",
            )
            self.staging_layers.append(staging)

        # Pointers for NIXL registration and _register_kv_args
        self.layer_ptrs = [t.data_ptr() for t in self.staging_layers]
        self.layer_lens = [t.numel() * t.element_size() for t in self.staging_layers]

        # Page allocator (thread-safe)
        self._free_pages = list(range(self.max_staging_pages))
        self._lock = threading.Lock()

        # Stream for async copy-back
        self.stream = torch.cuda.Stream(device=device)

    def alloc_pages(self, n: int) -> list[int]:
        with self._lock:
            if len(self._free_pages) < n:
                raise RuntimeError(
                    f"Receive staging buffer full: need {n} pages, "
                    f"only {len(self._free_pages)} available"
                )
            allocated = self._free_pages[:n]
            self._free_pages = self._free_pages[n:]
            return allocated

    def release_pages(self, pages: list[int]):
        with self._lock:
            self._free_pages.extend(pages)

    def copy_to_kv_cache(
        self, kv_buffers: list, staging_indices: list[int], kv_indices, page_size: int
    ):
        """Copy received data from staging pages to actual KV cache pages. Synchronous."""
        import torch

        device = self.staging_layers[0].device
        staging_pages = torch.tensor(staging_indices, dtype=torch.long, device=device)
        kv_pages = torch.tensor(kv_indices, dtype=torch.long, device=device)
        offsets = torch.arange(page_size, dtype=torch.long, device=device)

        staging_tokens = (
            staging_pages[:, None] * page_size + offsets[None, :]
        ).reshape(-1)
        kv_tokens = (kv_pages[:, None] * page_size + offsets[None, :]).reshape(-1)

        with torch.cuda.stream(self.stream):
            for layer_idx in range(len(kv_buffers)):
                kv_buffers[layer_idx][kv_tokens] = self.staging_layers[layer_idx][
                    staging_tokens
                ]
        self.stream.synchronize()


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            gpu_id=int(msg[7].decode("ascii")),
            decode_tp_size=int(msg[8].decode("ascii")),
            decode_tp_rank=int(msg[9].decode("ascii")),
            dst_kv_item_len=int(msg[10].decode("ascii")),
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per pp_rank: {pp_rank: set of chunk_ids}
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per pp_rank (set when is_last=True): {pp_rank: expected_count}
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of PP ranks expected to send data.
    num_pp_ranks_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        # All PP ranks must have reported their expected count
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        # Each PP rank must have received all expected chunks
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True

    def is_failed(self):
        return self.is_failure


class StagingBuffer:
    """Pre-allocated GPU staging buffer for NIXL transfers."""

    def __init__(self, size_bytes: int, device: int):
        self.buffer = torch.empty(
            size_bytes, dtype=torch.uint8, device=f"cuda:{device}"
        )
        self.base_ptr = self.buffer.data_ptr()
        self.size = size_bytes
        self.offset = 0
        self.stream = torch.cuda.Stream(device=device)

    def reset(self):
        self.offset = 0

    def copy_pages(
        self, kv_buf: torch.Tensor, page_start: int, page_count: int, page_size: int
    ) -> tuple[int, int]:
        """Copy contiguous pages from KV buffer to staging buffer.
        Returns (staging_ptr, num_bytes)."""
        token_start = page_start * page_size
        token_end = token_start + page_count * page_size
        src_slice = kv_buf[token_start:token_end]
        nbytes = src_slice.numel() * src_slice.element_size()

        if self.offset + nbytes > self.size:
            raise RuntimeError(
                f"Staging buffer full: need {nbytes} bytes at offset {self.offset}, capacity {self.size}"
            )

        dst_view = self.buffer[self.offset : self.offset + nbytes]
        with torch.cuda.stream(self.stream):
            dst_view.copy_(src_slice.reshape(-1).view(torch.uint8))

        ptr = self.base_ptr + self.offset
        self.offset += nbytes
        return ptr, nbytes

    def synchronize(self):
        """Wait for all copies to complete."""
        self.stream.synchronize()


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.kv_buffers = args.kv_buffers

        # Staging buffer setup (sender-side only)
        staging_size_mb = envs.SGLANG_NIXL_SEND_STAGING_BUFFER_SIZE_MB.get()
        if staging_size_mb > 0 and disaggregation_mode == DisaggregationMode.PREFILL:
            self.staging_buffer = StagingBuffer(
                staging_size_mb * 1024 * 1024, args.gpu_id
            )
            self._last_staging_handles: list = []
            logger.info(
                f"Using staging buffer of {staging_size_mb} MB for NIXL transfers"
            )
        else:
            self.staging_buffer = None

        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        agent_config = nixl_agent_config(
            backends=[backend],
            num_threads=(8 if disaggregation_mode == DisaggregationMode.PREFILL else 0),
        )
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        # Receive staging buffer setup (decode side)
        recv_staging_size_mb = envs.SGLANG_NIXL_RECV_STAGING_BUFFER_SIZE_MB.get()
        if (
            recv_staging_size_mb > 0
            and disaggregation_mode == DisaggregationMode.DECODE
        ):
            self.recv_staging_buffer = ReceiveStagingBuffer(
                recv_staging_size_mb, args, args.kv_buffers, args.gpu_id
            )
        else:
            self.recv_staging_buffer = None

        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()

            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
            )
            self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _start_heartbeat_checker_thread(self):
        """
        Start the heartbeat checker thread for Decode worker.
        TODO (smor): unite nixl heartbeat checker with mooncake's.
        """

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """Handle failure of a prefill node."""
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_attn_tp_size_table:
                del self.prefill_attn_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_pp_size_table:
                del self.prefill_pp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Mark all pending transfers associated with the failed node as failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.transfer_statuses
                and not self.transfer_statuses[room].is_done()
            ):
                # Mark the transfer as failed
                self.transfer_statuses[room].is_failure = True
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} transfers affected"
        )
        for room in possible_affected_rooms:
            logger.error(f"Let room {room} be failed due to prefill down")
            self.update_status(room, KVPoll.Failed)

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        pass

    def register_buffer_to_engine(self):
        if self.staging_buffer is not None:
            # Prefill: register sender staging buffer instead of KV cache
            staging_addrs = [
                (
                    self.staging_buffer.base_ptr,
                    self.staging_buffer.size,
                    self.kv_args.gpu_id,
                    "",
                )
            ]
            self.kv_descs = self.agent.register_memory(staging_addrs, "VRAM")
            logger.debug(f"Register staging buffer, size= {self.staging_buffer.size}")
            if not self.kv_descs:
                raise Exception("NIXL memory registration failed for staging buffer")
        elif self.recv_staging_buffer is not None:
            # Decode: register receiver staging buffer instead of KV cache
            kv_addrs = [
                (ptr, length, self.kv_args.gpu_id, "")
                for ptr, length in zip(
                    self.recv_staging_buffer.layer_ptrs,
                    self.recv_staging_buffer.layer_lens,
                )
            ]
            self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
            logger.debug(f"Register recv staging buffer, len(kv_addr)= {len(kv_addrs)}")
            if not self.kv_descs:
                raise Exception(
                    "NIXL memory registration failed for recv staging buffer"
                )
        else:
            # Default: register KV cache directly
            kv_addrs = [
                (kv_data_ptr, kv_data_len, self.kv_args.gpu_id, "")
                for kv_data_ptr, kv_data_len in zip(
                    self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
                )
            ]
            self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
            logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
            if not self.kv_descs:
                raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)

    def _wait_staging_handles(self):
        """Wait for all in-flight NIXL transfers that use staging buffer data."""
        for handle in self._last_staging_handles:
            while self.agent.check_xfer_state(handle) not in ("DONE", "ERR"):
                pass
        self._last_staging_handles.clear()

    def _get_layer_buffers(self, layers_current_pp_stage: int) -> list | None:
        """Get KV buffer tensors matching layers_params order."""
        if not self.kv_buffers:
            return None
        if self.is_mla_backend:
            return self.kv_buffers[:layers_current_pp_stage]
        else:
            num_kv_layers = len(self.kv_args.kv_data_ptrs) // 2
            return (
                self.kv_buffers[:layers_current_pp_stage]
                + self.kv_buffers[
                    num_kv_layers : num_kv_layers + layers_current_pp_stage
                ]
            )

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )

            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]

        src_addrs = []
        dst_addrs = []

        layer_buffers = self._get_layer_buffers(layers_current_pp_stage)

        if self.staging_buffer is not None and layer_buffers is not None:
            # === STAGING BUFFER PATH ===
            # 1. Wait for previous transfer to finish reading from staging buffer
            self._wait_staging_handles()
            self.staging_buffer.reset()

            # 2. Copy KV pages to staging buffer
            page_size = self.kv_args.page_size
            for layer_idx, (_, dst_ptr, item_len) in enumerate(layers_params):
                kv_buf = layer_buffers[layer_idx]
                for prefill_index, decode_index in zip(
                    prefill_kv_blocks, dst_kv_blocks
                ):
                    page_start = int(prefill_index[0])
                    page_count = len(prefill_index)
                    staging_ptr, nbytes = self.staging_buffer.copy_pages(
                        kv_buf, page_start, page_count, page_size
                    )
                    src_addrs.append((staging_ptr, nbytes, self.kv_args.gpu_id))
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    dst_addrs.append((dst_addr, nbytes, dst_gpu_id))

            # 3. Synchronize CUDA copy stream before NIXL transfer
            self.staging_buffer.synchronize()
        else:
            # === ORIGINAL DIRECT PATH ===
            for src_ptr, dst_ptr, item_len in layers_params:
                for prefill_index, decode_index in zip(
                    prefill_kv_blocks, dst_kv_blocks
                ):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    src_addrs.append((src_addr, length, self.kv_args.gpu_id))
                    dst_addrs.append((dst_addr, length, dst_gpu_id))

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_kv_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")

        # Track handle for staging buffer lifecycle
        if self.staging_buffer is not None:
            self._last_staging_handles.append(xfer_handle)

        return xfer_handle

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_kv_item_len: int,
    ):
        # Get configuration from kv_args
        local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size
        num_kv_heads = self.kv_args.kv_head_num

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine which heads to send
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )
        # Create transfer descriptors
        src_addrs = []
        dst_addrs = []

        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        src_dst_ptr_pairs = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        src_addrs = []
        dst_addrs = []

        # Calculate strides for a single token slot
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size

        for src_ptr, dst_ptr in src_dst_ptr_pairs:
            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_kv_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_kv_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addrs.append(
                        (
                            src_slice_addr,
                            heads_bytes_per_token_to_send,
                            self.kv_args.gpu_id,
                        )
                    )
                    dst_addrs.append(
                        (dst_slice_addr, heads_bytes_per_token_to_send, dst_gpu_id)
                    )

        # Use NIXL agent for transfer
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("Failed to create sliced KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post sliced KV transfer")

        return xfer_handle

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.pp_rank}"
            decode_tp_size = self.decode_kv_args_table[req.agent_name].decode_tp_size

            if self.is_mla_backend or (decode_tp_size == self.attn_tp_size):
                kv_xfer_handle = self.send_kvcache(
                    req.agent_name,
                    kv_indices,
                    self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    self.decode_kv_args_table[req.agent_name].gpu_id,
                    notif,
                )
            else:
                kv_xfer_handle = self.send_kvcache_slice(
                    req.agent_name,
                    kv_indices,
                    self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    self.decode_kv_args_table[req.agent_name].gpu_id,
                    notif,
                    prefill_tp_size=self.attn_tp_size,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=self.decode_kv_args_table[
                        req.agent_name
                    ].decode_tp_rank,
                    dst_kv_item_len=self.decode_kv_args_table[
                        req.agent_name
                    ].dst_kv_item_len,
                )

            handles.append(kv_xfer_handle)
            # Only the last chunk we need to send the aux data.
            if is_last:
                assert aux_index is not None
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    self.decode_kv_args_table[req.agent_name].dst_aux_ptrs,
                    req.dst_aux_index,
                    str(req.room) + "_aux",
                )
                handles.append(aux_xfer_handle)
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return handles

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # We could also check that self.bootstrap_info['agent_name'] matches
            # the message sender. But the bootstrap room alone should be
            # sufficient to map the status.
            for msg in messages:
                components = msg.decode("ascii").split("_", 4)
                room = int(components[0])
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    # Track received chunks per pp_rank
                    self.transfer_statuses[room].received_kvs_per_pp[pp_rank].add(
                        chunk_id
                    )
                    if is_last:
                        # Record expected chunk count for this pp_rank
                        self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = (
                            chunk_id + 1
                        )
                        # Set num_pp_ranks_expected from table (or default to 1)
                        if self.transfer_statuses[room].num_pp_ranks_expected is None:
                            self.transfer_statuses[room].num_pp_ranks_expected = (
                                self.required_prefill_response_num_table.get(room, 1)
                            )
                elif components[1] == "aux":
                    self.transfer_statuses[room].received_aux = True

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.xfer_handles = []
        self.has_sent = False
        self.chunk_id = 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        new_xfer_handles = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise RuntimeError("NIXL KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        self.staging_indices = None  # Staging pages allocated for this request
        self.real_kv_indices = None  # Original KV indices for copy-back
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        # Track this room with its bootstrap address for heartbeat monitoring
        if hasattr(self.kv_mgr, "addr_to_rooms_tracker"):
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(
                self.bootstrap_room
            )
        self.init_time = None

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Staging: allocate staging pages and remap indices
        if self.kv_mgr.recv_staging_buffer is not None and kv_indices.size > 0:
            self.real_kv_indices = kv_indices.copy()
            self.staging_indices = self.kv_mgr.recv_staging_buffer.alloc_pages(
                len(kv_indices)
            )
            kv_indices = np.array(self.staging_indices, dtype=np.int32)

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        now = time.time()
        elapsed = now - self.init_time

        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            if self.kv_mgr.transfer_statuses[self.bootstrap_room].is_failed():
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"Transfer for room {self.bootstrap_room} failed due to node failure"
                )
                # Release staging pages on failure
                if self.staging_indices is not None:
                    self.kv_mgr.recv_staging_buffer.release_pages(self.staging_indices)
                    self.staging_indices = None
            else:
                # Copy from staging to KV cache before reporting success
                if self.staging_indices is not None:
                    self.kv_mgr.recv_staging_buffer.copy_to_kv_cache(
                        self.kv_mgr.kv_buffers,
                        self.staging_indices,
                        self.real_kv_indices,
                        self.kv_mgr.kv_args.page_size,
                    )
                    self.kv_mgr.recv_staging_buffer.release_pages(self.staging_indices)
                    self.staging_indices = None
                self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            # Use staging buffer pointers if available
            if self.kv_mgr.recv_staging_buffer is not None:
                kv_ptrs = self.kv_mgr.recv_staging_buffer.layer_ptrs
            else:
                kv_ptrs = self.kv_mgr.kv_args.kv_data_ptrs
            packed_kv_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in kv_ptrs)
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.kv_mgr.kv_args.decode_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                    ]
                )

    def clear(self):
        if self.staging_indices is not None:
            self.kv_mgr.recv_staging_buffer.release_pages(self.staging_indices)
            self.staging_indices = None

    def failure_exception(self):
        raise RuntimeError("NIXL KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
