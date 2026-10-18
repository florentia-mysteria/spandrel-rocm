"""
ROCm acceleration controls.

On AMD's ROCm builds of PyTorch for Windows, spandrel runs some operations with
custom HIP kernels (compiled at runtime with HIPRTC, which ships with PyTorch).
Elsewhere nothing changes.

Environment variables:
- SPANDREL_HIP_KERNELS=0 disables all custom kernels.
- SPANDREL_HIP_WMMA=0 disables the WMMA attention kernel on RDNA4 (gfx12).
- SPANDREL_HIP_CACHE_DIR sets where compiled kernels are cached (empty: no
  cache). Default: %LOCALAPPDATA%/spandrel/hip_cache.
"""

from __future__ import annotations

from .__helpers.rocm.common import clear_caches
from .__helpers.rocm.kernels import hip_kernels_available as is_available

__all__ = ["clear_caches", "is_available"]
