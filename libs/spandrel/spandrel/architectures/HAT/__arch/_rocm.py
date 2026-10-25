"""
Faster inference path for HAT's hybrid attention block (HAB) on ROCm.

- The (shifted) window attention runs as one fused HIP kernel that reads q/k/v
  in place and writes the image layout directly: no torch.roll, no window
  partition/reverse, and the 256x256 attention matrices of every window and
  head never reach memory. The shift mask is passed as one region id per token.
- `shortcut + attn + CAB(x) * conv_scale` and the second LayerNorm are one HIP
  kernel, which reads the CAB conv output in its image layout and applies the
  channel attention on the fly: no permute copy and no separate passes.
- The first LayerNorm uses the HIP kernel too.

The CAB convolutions stay on MIOpen. Returns None when it cannot be used, and
HAT.py runs its original code. OCAB is unchanged.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from spandrel.__helpers.rocm import kernels
from spandrel.__helpers.rocm.common import register_cache, version_of

logger = logging.getLogger("spandrel.rocm")

_failed = False


def _cached(module: nn.Module, name: str, tensors: list[torch.Tensor], build):  # noqa: ANN001
    key = tuple((t.data_ptr(), version_of(t), t.dtype) for t in tensors)
    cached = module.__dict__.get(name)
    if cached is None or cached[0] != key:
        cached = (key, build())
        module.__dict__[name] = cached
    return cached[1]


def _ln_params(norm: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    return _cached(
        norm,
        "_spandrel_ln",
        [norm.weight, norm.bias],
        lambda: (
            norm.weight.detach().float().contiguous(),
            norm.bias.detach().float().contiguous(),
        ),
    )


def _bias_t(attn: nn.Module, rpi: torch.Tensor) -> torch.Tensor:
    """The relative position bias as float32 (heads, key, query)."""
    table = attn.relative_position_bias_table
    n = attn.window_size[0] * attn.window_size[1]

    def build() -> torch.Tensor:
        rpb = table[rpi.view(-1)].view(n, n, -1)  # q k h
        return rpb.permute(2, 1, 0).float().contiguous()  # h k q

    return _cached(attn, "_spandrel_bias_t", [table, rpi], build)


_label_cache: dict[tuple, torch.Tensor] = {}
register_cache(_label_cache.clear)


def _shift_labels(H: int, W: int, ws: int, shift: int, device: torch.device):  # noqa: N803
    """
    The shift region id of every token per window (nW, ws*ws): the regions of
    HAT's (Swin's) calculate_mask, which are -100 apart exactly where the region
    ids differ. Several tile sizes are cached; each is only H*W ints.
    """
    key = (H, W, ws, shift, device)
    labels = _label_cache.get(key)
    if labels is None:
        img = torch.zeros(H, W, dtype=torch.int32)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                img[hs, wsl] = cnt
                cnt += 1
        windows = img.view(H // ws, ws, W // ws, ws).permute(0, 2, 1, 3)
        labels = windows.reshape(-1, ws * ws).contiguous().to(device)
        if len(_label_cache) >= 16:
            _label_cache.pop(next(iter(_label_cache)))
        _label_cache[key] = labels
    return labels


def _is_conv3x3(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.groups == 1
    )


def _supported(block: nn.Module, c: int) -> bool:
    try:
        cab = block.conv_block.cab
        attn = block.attn
        return (
            all(
                isinstance(n, nn.LayerNorm)
                and n.elementwise_affine
                and n.bias is not None
                and tuple(n.normalized_shape) == (c,)
                for n in (block.norm1, block.norm2)
            )
            and 0 < c <= 256
            and len(cab) == 4
            and _is_conv3x3(cab[0])
            and _is_conv3x3(cab[2])
            and cab[2].out_channels == c
            and hasattr(cab[3], "attention")
            and attn.window_size[0] == attn.window_size[1] == block.window_size
        )
    except (AttributeError, TypeError, IndexError):
        return False


def hab_forward(module: nn.Module, x, x_size, rpi_sa):  # noqa: ANN001
    """HAB.forward, or None if this path cannot be used."""
    global _failed
    # drop_path is a no-op outside training
    if _failed or module.training or not kernels.hip_kernels_available():
        return None
    x = x.contiguous()  # the first block of a group gets a transposed view
    if not kernels.can_layer_norm(x):
        return None
    h, w = x_size
    b, n, c = x.shape
    ws, shift = module.window_size, module.shift_size
    supported = module.__dict__.get("_spandrel_supported")
    if supported is None:
        supported = _supported(module, c)
        module.__dict__["_spandrel_supported"] = supported
    if not supported or n != h * w or h % ws or w % ws:
        return None
    attn = module.attn
    try:
        g1, b1 = _ln_params(module.norm1)
        _, n1 = kernels.layer_norm(x, g1, b1, module.norm1.eps)

        qkv = attn.qkv(n1).view(b, n, 3, c).permute(2, 0, 1, 3)
        if not kernels.can_window_attention(qkv, h, w, ws, ws, attn.num_heads):
            return None

        # CAB: the convs as in the original, the channel attention is applied
        # by the residual kernel below
        cab = module.conv_block.cab
        conv = cab[2](cab[1](cab[0](n1.view(b, h, w, c).permute(0, 3, 1, 2))))
        ca = cab[3].attention(conv).view(b, c).float()

        labels = _shift_labels(h, w, ws, shift, x.device) if shift > 0 else None
        out = torch.empty((b, h, w, c), device=x.device, dtype=x.dtype)
        kernels.window_attention(
            qkv,
            h,
            w,
            ws,
            ws,
            attn.num_heads,
            float(attn.scale),
            _bias_t(attn, rpi_sa),
            labels,
            out=out,
            shift=(shift, shift),
        )
        attn_x = attn.proj(out.view(b, n, c))

        g2, b2 = _ln_params(module.norm2)
        s, n2 = kernels.hab_layer_norm(
            x, attn_x, conv, ca, float(module.conv_scale), g2, b2, module.norm2.eps
        )
        return s + module.mlp(n2)
    except Exception as e:  # noqa: BLE001
        _failed = True
        logger.warning("HIP HAT block failed, falling back: %s", e)
        return None


_ocab_failed = False


def _ocab_bias_t(block: nn.Module, rpi: torch.Tensor) -> torch.Tensor:
    """OCAB's relative position bias as float32 (heads, key, query)."""
    table = block.relative_position_bias_table
    nq, nk = block.window_size**2, block.overlap_win_size**2

    def build() -> torch.Tensor:
        rpb = table[rpi.view(-1)].view(nq, nk, -1)  # q k h
        return rpb.permute(2, 1, 0).float().contiguous()  # h k q

    return _cached(block, "_spandrel_bias_t", [table, rpi], build)


def ocab_forward(module: nn.Module, x, x_size, rpi):  # noqa: ANN001
    """
    OCAB.forward, or None if this path cannot be used. The overlapping cross
    attention is one WMMA kernel (RDNA4, 16-bit) that reads the overlapping k/v
    windows in place instead of unfolding them; the LayerNorms use the HIP
    kernel with the attention residual fused into the second one.
    """
    global _ocab_failed
    if _ocab_failed or module.training or not kernels.hip_kernels_available():
        return None
    x = x.contiguous()
    h, w = x_size
    b, n, c = x.shape
    if n != h * w or not kernels.can_layer_norm(x) or c > 256:
        return None
    if not all(
        isinstance(norm, nn.LayerNorm)
        and norm.elementwise_affine
        and norm.bias is not None
        and tuple(norm.normalized_shape) == (c,)
        for norm in (module.norm1, module.norm2)
    ):
        return None
    try:
        g1, b1 = _ln_params(module.norm1)
        _, n1 = kernels.layer_norm(x, g1, b1, module.norm1.eps)
        qkv = module.qkv(n1).view(b, n, 3, c).permute(2, 0, 1, 3)
        ws, ows = module.window_size, module.overlap_win_size
        if not kernels.can_ocab_attention(qkv, h, w, ws, ows, module.num_heads):
            return None
        out = kernels.ocab_attention(
            qkv,
            h,
            w,
            ws,
            ows,
            module.num_heads,
            float(module.scale),
            _ocab_bias_t(module, rpi),
        )
        g2, b2 = _ln_params(module.norm2)
        # the original adds proj + shortcut; addition is commutative
        s, n2 = kernels.layer_norm(
            x, g2, b2, module.norm2.eps, residual=module.proj(out.view(b, n, c))
        )
        return s + module.mlp(n2)
    except Exception as e:  # noqa: BLE001
        _ocab_failed = True
        logger.warning("HIP OCAB kernel failed, falling back: %s", e)
        return None
