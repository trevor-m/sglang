# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import logging
from contextlib import contextmanager
from typing import List, Optional, Union

# ===================== import region =====================
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
import threading

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from sglang.srt.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class PyNcclSymmetricMemory:
    def __init__(self, nccl: NCCLLibrary, comm: ncclComm_t, size: int):
        self.nccl = nccl
        self.comm = comm
        self.size = size
        self.ptr = self.nccl.ncclMemAlloc(size)
        win_flags = 1 # NCCL_WIN_COLL_SYMMETRIC
        self.window = self.nccl.ncclCommWindowRegister(self.comm, self.ptr, size, win_flags)

    def data_ptr(self):
        return self.ptr
    
    def __del__(self):
        self.nccl.ncclCommWindowDeregister(self.comm, self.window)
        self.nccl.ncclMemFree(self.ptr)


class PyNcclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
        device_group: Optional[Union[ProcessGroup, StatelessProcessGroup]] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.NCCL
            ), "PyNcclCommunicator should be attached to a non-NCCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            self.stream = None
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception as e:
            logger.warning("Error loading NCCL, disabling. Error reason: %s", e)
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            self.stream = None
            return

        self.available = True
        self.disabled = False

        if self.rank == 0:
            logger.info("sglang is using nccl==%s", self.nccl.ncclGetVersion())

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )
            self.stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            self.stream.synchronize()
            del data

            # Create symmetric memory pool
            # backend = device_group._get_backend(torch.device(device))
            # pool = torch.cuda.MemPool(backend.mem_allocator)#, symm_mem=True)
            # with torch.cuda.use_mem_pool(pool):
            #     self.symm_mem_workspace = torch.arange(1024 * 1024 * 2, device=device, dtype=torch.uint8)
            self.symm_mem_workspace = PyNcclSymmetricMemory(self.nccl, self.comm, 1024*1024*512) #29360128

        self.cuda_lib = CudaRTLibrary()
        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        self.disabled = True

    def all_reduce(
        self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream=None,
        sizes: Optional[List[int]] = None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = self.stream

        if sizes is not None:
            split_offset = 0

            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                dst_slice = output_tensor[split_offset : split_offset + split_size]

                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            self.nccl.ncclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
        sizes: Optional[List[int]] = None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = self.stream

        assert self.symm_mem_workspace.size >= input_tensor.nbytes, f"symmetric memory is too small - need {input_tensor.nbytes} bytes"
        self.cuda_lib.cudaMemcpyAsync(
            self.symm_mem_workspace.data_ptr(),
            buffer_type(input_tensor.data_ptr()),
            input_tensor.nbytes,
            cudaStream_t(stream.cuda_stream)
        )

        current_rank_output = 0
        if sizes is not None:
            numel_base = int(np.prod(input_tensor.shape[1:]))
            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                #chunk = input_tensor[split_offset : split_offset + split_size, ...]
                output_ptr = buffer_type(self.symm_mem_workspace.data_ptr().value + split_offset * numel_base * input_tensor.element_size())
                if root == self.rank:
                    current_rank_output = output_ptr

                self.nccl.ncclReduce(
                    output_ptr, #buffer_type(chunk.data_ptr()),
                    output_ptr, #buffer_type(output_tensor.data_ptr()),
                    split_size * numel_base, #chunk.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    ncclRedOpTypeEnum.from_torch(op),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            current_rank_output = buffer_type(self.symm_mem_workspace.data_ptr().value + self.rank * output_tensor.nbytes)
            self.nccl.ncclReduceScatter(
                self.symm_mem_workspace.data_ptr(), #buffer_type(input_tensor.data_ptr()),
                current_rank_output, #buffer_type(output_tensor.data_ptr()),
                output_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
        
        self.cuda_lib.cudaMemcpyAsync(
            buffer_type(output_tensor.data_ptr()),
            current_rank_output,
            output_tensor.nbytes,
            cudaStream_t(stream.cuda_stream)
        )


    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    @contextmanager
    def change_state(
        self, enable: Optional[bool] = None, stream: Optional[torch.cuda.Stream] = None
    ):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream
