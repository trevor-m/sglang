
import tempfile
import torch
from torch.cuda.memory import CUDAPluggableAllocator


nccl_allocator_source = """
#include <nccl.h>
extern "C" {

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return ptr;

}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""

_mem_pool = None

def get_nccl_mem_pool():
    global _mem_pool
    if _mem_pool is None:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        nccl_allocator = torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )

        allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so", "nccl_alloc_plug", "nccl_free_plug"
        ).allocator()
        _mem_pool = torch.cuda.MemPool(allocator)
    
    return _mem_pool
