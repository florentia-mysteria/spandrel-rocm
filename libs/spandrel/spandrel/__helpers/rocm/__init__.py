"""
ROCm acceleration: custom HIP kernels for operations that PyTorch and MIOpen
handle poorly on consumer Radeon GPUs, and the architecture code that uses them.

Everything here is optional. Without a ROCm build of PyTorch on Windows (or with
SPANDREL_HIP_KERNELS=0), every model runs its original code.
"""
