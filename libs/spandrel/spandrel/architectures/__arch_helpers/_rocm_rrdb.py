"""
Faster inference path for ESRGAN's RRDB blocks on ROCm.

A residual dense block concatenates its input with every previous conv output
before the next conv (64, 96, 128, 160 and 192 channels), copying everything it
already has again each time, and runs LeakyReLU and the `* 0.2 + x` residuals
as separate passes. PyTorch's elementwise ops on ROCm run at a fraction of the
memory bandwidth, so with the convs on fast Winograd kernels this is about a
third of ESRGAN's run time.

Here every dense block works in one buffer [x | x1 | x2 | x3 | x4]: each conv
reads a channel prefix of it (no copy for a batch of 1), a HIP kernel applies
the activation while writing the conv output into the next slot, and another
one does the residuals, writing the block output straight into the first slot
of the next block's buffer. The convs themselves are unchanged. The results
are bit-identical to the original.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from spandrel.__helpers.rocm import kernels

logger = logging.getLogger("spandrel.rocm")

_failed = False


def _is_conv3x3(m: nn.Module, in_ch: int, out_ch: int) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.in_channels == in_ch
        and m.out_channels == out_ch
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.groups == 1
        and m.padding_mode == "zeros"
    )


def _activation(m: nn.Module) -> tuple[float, bool] | None:
    """(negative slope, is ReLU) of a supported activation."""
    if isinstance(m, nn.LeakyReLU):
        return float(m.negative_slope), False
    if isinstance(m, nn.ReLU):
        return 0.0, True
    return None


def _rdb_layout(rdb: nn.Module) -> tuple[int, int, list] | None:
    """(nf, gc, [(conv, slope, relu)] * 4 + [(conv5, 0, False)]) or None."""
    try:
        if getattr(rdb, "conv1x1", None) is not None:
            return None  # ESRGAN+ extra residual paths
        first = rdb.conv1
        if not isinstance(first, nn.Sequential) or len(first) != 2:
            return None
        nf, gc = first[0].in_channels, first[0].out_channels
        layers = []
        for i, name in enumerate(("conv1", "conv2", "conv3", "conv4")):
            seq = getattr(rdb, name)
            if not isinstance(seq, nn.Sequential) or len(seq) != 2:
                return None
            act = _activation(seq[1])
            if act is None or not _is_conv3x3(seq[0], nf + i * gc, gc):
                return None
            layers.append((seq[0], act[0], act[1]))
        conv5 = rdb.conv5
        if isinstance(conv5, nn.Sequential) and len(conv5) == 1:
            conv5 = conv5[0]
        if not _is_conv3x3(conv5, nf + 4 * gc, nf):
            return None
        layers.append((conv5, 0.0, False))
        return nf, gc, layers
    except (AttributeError, TypeError, IndexError):
        return None


def _layouts(module: nn.Module) -> list | None:
    cached = module.__dict__.get("_spandrel_rrdb_layout")
    if cached is None:
        layouts = [_rdb_layout(r) for r in (module.RDB1, module.RDB2, module.RDB3)]
        ok = (
            all(lay is not None for lay in layouts)
            and len(
                {(lay[0], lay[1]) for lay in layouts}  # type: ignore[index]
            )
            == 1
        )
        cached = (layouts if ok else None,)
        module.__dict__["_spandrel_rrdb_layout"] = cached
    return cached[0]


def _input_buffer(x: torch.Tensor, channels: int) -> torch.Tensor:
    """
    A (B, channels, H, W) buffer whose first channels hold x. The output of the
    previous RRDB already is the first slot of such a buffer, so it is reused.
    """
    base = x._base
    if (
        base is not None
        and base.dim() == 4
        and base.shape[0] == x.shape[0]
        and base.shape[1] == channels
        and base.shape[2:] == x.shape[2:]
        and base.is_contiguous()
        and x.data_ptr() == base.data_ptr()
        and x.stride() == base.stride()
    ):
        return base
    b, c, h, w = x.shape
    buf = torch.empty((b, channels, h, w), device=x.device, dtype=x.dtype)
    buf[:, :c].copy_(x)
    return buf


def rrdb_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """RRDB.forward, or None if this path cannot be used."""
    global _failed
    if (
        _failed
        or module.training
        or not kernels.hip_kernels_available()
        or not kernels.can_rrdb(x)
    ):
        return None
    layouts = _layouts(module)
    if layouts is None:
        return None
    nf, gc, _ = layouts[0]
    if x.shape[1] != nf:
        return None
    channels = nf + 4 * gc
    try:
        b, _, h, w = x.shape
        cur = _input_buffer(x, channels)
        for k, (_, _, layers) in enumerate(layouts):
            for i, (conv, slope, relu) in enumerate(layers[:4]):
                y = conv(cur[:, : nf + i * gc])
                kernels.act_into(y, cur, nf + i * gc, slope, relu)
            y5 = layers[4][0](cur)
            nxt = torch.empty((b, channels, h, w), device=x.device, dtype=x.dtype)
            if k < 2:
                # RDB: x5 * 0.2 + x
                kernels.residual_add(y5, cur[:, :nf], nxt[:, :nf], 0.2)
            else:
                # last RDB and the RRDB around it: (x5 * 0.2 + x) * 0.2 + x_rrdb
                kernels.residual_add(y5, cur[:, :nf], nxt[:, :nf], 0.2, x, 0.2)
            cur = nxt
        return cur[:, :nf]
    except Exception as e:  # noqa: BLE001
        _failed = True
        logger.warning("HIP RRDB kernels failed, falling back: %s", e)
        return None
