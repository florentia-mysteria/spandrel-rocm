from __future__ import annotations

import logging
from typing import Callable, cast

import torch
from torch import nn

logger = logging.getLogger("spandrel.rocm")

# Functions that drop cached GPU tensors, see clear_caches.
_cache_clearers: list[Callable[[], None]] = []


def register_cache(clear: Callable[[], None]) -> None:
    _cache_clearers.append(clear)


def clear_caches() -> None:
    """
    Drops cached GPU tensors (attention masks, region ids, ...). A mask for a
    large tile is hundreds of MB, and torch.cuda.empty_cache() cannot release
    it while it is still referenced.
    """
    for clear in _cache_clearers:
        clear()


def version_of(t: torch.Tensor) -> int:
    """
    In-place modification counter, for cache keys. Inference tensors (e.g.
    weights converted to another dtype inside torch.inference_mode) have none,
    and they cannot be modified outside of inference mode anyway.
    """
    try:
        return t._version
    except RuntimeError:
        return -1


def is_depthwise_3x3(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels == m.out_channels
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
    )


_dwconv_failed = False


def patch_depthwise_convs(model: nn.Module) -> None:
    """
    Routes depthwise 3x3 convolutions through a custom HIP kernel on ROCm.

    MIOpen has no fast kernel for them on consumer Radeon cards: on an RX 9070 XT
    one takes ~4 ms on a 180x256x256 tensor, the custom kernel ~0.16 ms. DAT runs
    72 of them per tile, which made them its single biggest cost. Many other
    architectures (SAFMN, OmniSR, DRCT, RGT, ...) use them as well.

    Any failure (compile, launch, unsupported input) falls back to the original
    convolution. Set SPANDREL_HIP_KERNELS=0 to disable all custom HIP kernels.
    """
    if getattr(model, "_spandrel_dwconv_patched", False):
        return

    from .kernels import can_dwconv3x3, dwconv3x3, hip_kernels_available

    if _dwconv_failed or not hip_kernels_available():
        return
    model._spandrel_dwconv_patched = True  # type: ignore[attr-defined]

    count = 0
    for module in model.modules():
        if not is_depthwise_3x3(module):
            continue
        conv = cast(nn.Conv2d, module)
        original_forward = module.forward
        cache: dict[str, object] = {}

        def forward(
            x: torch.Tensor,
            m: nn.Conv2d = conv,
            original: Callable[[torch.Tensor], torch.Tensor] = original_forward,
            cache: dict[str, object] = cache,
        ) -> torch.Tensor:
            global _dwconv_failed
            if _dwconv_failed or not can_dwconv3x3(x):
                return original(x)
            # fp32 weights in the kernel's (C, 9) layout, rebuilt if the module
            # was moved to another device or dtype
            key = (m.weight.data_ptr(), m.weight.dtype, version_of(m.weight))
            if cache.get("key") != key:
                cache["key"] = key
                cache["w"] = (
                    m.weight.detach().float().reshape(m.in_channels, 9).contiguous()
                )
                cache["b"] = (
                    m.bias.detach().float().contiguous() if m.bias is not None else None
                )
            try:
                return dwconv3x3(x, cache["w"], cache["b"])  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                _dwconv_failed = True
                logger.warning(
                    "HIP depthwise conv kernel failed, falling back to MIOpen: %s", e
                )
                return original(x)

        module.forward = forward
        count += 1

    if count:
        logger.debug("Using the HIP kernel for %d depthwise 3x3 convolutions", count)
