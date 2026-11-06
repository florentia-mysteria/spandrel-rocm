"""
Faster inference path for FDAT on ROCm.

FDAT's block moves its tokens into the (B, C, H, W) image layout and back for
every depthwise conv and 1x1 conv (conv branch, FFN, interaction), each time
with a full copy, and runs its window attention as separate matmuls on
partitioned windows. PyTorch's transposes and elementwise ops on ROCm run at a
fraction of the memory bandwidth, so these dominate.

Here the whole block stays in the token layout: the depthwise convs run with a
HIP kernel straight on the tokens (the FFN's GELU applied while loading), the
1x1 convs become linear layers, the average pool a mean over tokens, the window
attention is one fused kernel reading q/k/v in place, and the LayerNorms use
the HIP kernel with the first residual add fused into the second one.

Returns None when it cannot be used, and fdat.py runs its original code.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from spandrel.__helpers.rocm import kernels
from spandrel.__helpers.rocm.common import is_depthwise_3x3, version_of

logger = logging.getLogger("spandrel.rocm")

_failed = False
# for comparisons: SPANDREL_FDAT_CHAN_ATTN=0 runs the original channel attention
_chan_attn = os.environ.get("SPANDREL_FDAT_CHAN_ATTN", "").strip() != "0"
# also for comparisons: False restores the separate residual add after fc2 and
# the separate interaction before the second LayerNorm
_fused_res = True
_fused_aim = True


def _cached(module: nn.Module, name: str, tensors: list[torch.Tensor], build):  # noqa: ANN001
    key = tuple((t.data_ptr(), version_of(t), t.dtype) for t in tensors)
    cached = module.__dict__.get(name)
    if cached is None or cached[0] != key:
        cached = (key, build())
        module.__dict__[name] = cached
    return cached[1]


def _dw_weight(conv: nn.Conv2d) -> torch.Tensor:
    return _cached(
        conv,
        "_spandrel_dw_w",
        [conv.weight],
        lambda: conv.weight.detach().float().reshape(conv.in_channels, 9).contiguous(),
    )


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


def _is_gelu(m: nn.Module) -> bool:
    return isinstance(m, nn.GELU) and m.approximate == "none"


def _is_1x1(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups == 1


def _supported(block: nn.Module, c: int) -> bool:
    try:
        conv, inter, ffn = block.conv, block.inter, block.ffn
        return (
            all(
                isinstance(n, nn.LayerNorm)
                and n.elementwise_affine
                and n.bias is not None
                and tuple(n.normalized_shape) == (c,)
                for n in (block.n1, block.n2)
            )
            and len(conv) == 2
            and is_depthwise_3x3(conv[0])
            and conv[0].bias is None
            and _is_gelu(conv[1])
            and len(inter.sg) == 2
            and _is_1x1(inter.sg[0])
            and inter.sg[0].bias is None
            and isinstance(inter.sg[1], nn.Sigmoid)
            and len(inter.cg) == 5
            and isinstance(inter.cg[0], nn.AdaptiveAvgPool2d)
            and _is_1x1(inter.cg[1])
            and inter.cg[1].bias is None
            and _is_gelu(inter.cg[2])
            and _is_1x1(inter.cg[3])
            and inter.cg[3].bias is None
            and isinstance(inter.cg[4], nn.Sigmoid)
            and _is_gelu(ffn.act)
            and is_depthwise_3x3(ffn.smix)
            and ffn.smix.bias is None
        )
    except (AttributeError, TypeError):
        return False


def _window_attention(attn: nn.Module, x: torch.Tensor, H: int, W: int):  # noqa: N803
    """FastSpatialWindowAttention, or None if the kernel cannot be used."""
    B, L, C = x.shape  # noqa: N806
    ws = attn.ws
    hp, wp = -(-H // ws) * ws, -(-W // ws) * ws
    if (hp != H or wp != W) and attn.qkv.bias is not None:
        # padded tokens would get the qkv bias instead of zeros
        return None
    qkv = attn.qkv(x).view(B, L, 3, C).permute(2, 0, 1, 3)
    if not kernels.can_window_attention(qkv, hp, wp, ws, ws, attn.nh):
        return None
    bias_t = _cached(
        attn,
        "_spandrel_bias_t",
        [attn.bias],
        lambda: attn.bias.detach().float().permute(0, 2, 1).contiguous(),
    )
    out = torch.empty((B, H, W, C), device=x.device, dtype=x.dtype)
    kernels.window_attention(
        qkv, hp, wp, ws, ws, attn.nh, float(attn.scale), bias_t, None, out=out
    )
    return attn.proj(out.view(B, L, C))


def _channel_attention(attn: nn.Module, x: torch.Tensor):
    """FastChannelAttention, or None if the kernel cannot be used."""
    B, N, C = x.shape  # noqa: N806
    nh = attn.nh
    qkv = attn.qkv(x)
    if not kernels.can_channel_gram(qkv, nh):
        return None
    d = C // nh
    # softmax(normalize(q)^T normalize(k) * temp) from q^T k and the norms
    # (F.normalize divides by max(norm, 1e-12)), in float32
    gram, nq, nk = kernels.channel_gram(qkv, nh)
    nq = nq.clamp_min(1e-12).view(B, nh, d, 1)
    nk = nk.clamp_min(1e-12).view(B, nh, 1, d)
    a = (gram / (nq * nk) * attn.temp.float()).softmax(-1)
    # proj(attn @ v) = v @ (W_proj @ blockdiag(attn))^T + b_proj
    wp = attn.proj.weight.float().view(C, nh, d).transpose(0, 1)
    m = (wp @ a).permute(0, 2, 1, 3).reshape(B, C, C).to(x.dtype)
    v = qkv[:, :, 2 * C :]
    bias = attn.proj.bias
    if B == 1:
        return F.linear(v[0], m[0], bias).unsqueeze(0)
    out = v @ m.transpose(1, 2)
    return out if bias is None else out + bias


def _interaction(
    inter: nn.Module, a: torch.Tensor, c: torch.Tensor, spatial_gate: bool
):
    """SimplifiedAIM in the token layout."""
    if spatial_gate:
        # "spatial_modulates_channel": a + c * sigmoid(conv1x1(a))
        w = inter.sg[0].weight.view(1, -1).to(a.dtype)
        return torch.addcmul(a, c, torch.sigmoid(F.linear(a, w)))
    # "channel_modulates_spatial": a * channel_gate(c) + c
    pooled = torch.mean(c, dim=1, keepdim=True, dtype=torch.float32).to(c.dtype)
    w1 = inter.cg[1].weight.flatten(1).to(c.dtype)
    w2 = inter.cg[3].weight.flatten(1).to(c.dtype)
    gate = torch.sigmoid(F.linear(F.gelu(F.linear(pooled, w1)), w2))
    return torch.addcmul(c, a, gate)


def _aim_gate(
    inter: nn.Module, c: torch.Tensor, spatial_gate: bool
) -> dict[str, torch.Tensor]:
    """The gate argument of kernels.gated_mix_layer_norm for SimplifiedAIM."""
    if spatial_gate:
        # the conv weight rounded like F.linear in the model's dtype sees it
        w = _cached(
            inter.sg[0],
            "_spandrel_sw",
            [inter.sg[0].weight],
            lambda: inter.sg[0].weight.detach().to(c.dtype).float().flatten(),
        )
        return {"spatial_weight": w}
    pooled = torch.mean(c, dim=1, keepdim=True, dtype=torch.float32).to(c.dtype)
    w1 = inter.cg[1].weight.flatten(1).to(c.dtype)
    w2 = inter.cg[3].weight.flatten(1).to(c.dtype)
    gate = torch.sigmoid(F.linear(F.gelu(F.linear(pooled, w1)), w2))
    return {"channel_gate": gate.float().view(c.shape[0], -1)}


def fdat_block(module: nn.Module, x: torch.Tensor, H: int, W: int):  # noqa: N803
    """SimplifiedDATBlock.forward, or None if this path cannot be used."""
    global _failed
    if (
        _failed
        or module.training
        or not kernels.hip_kernels_available()
        or not kernels.can_layer_norm(x)
    ):
        return None
    c = x.shape[-1]
    supported = module.__dict__.get("_spandrel_supported")
    if supported is None:
        supported = _supported(module, c)
        module.__dict__["_spandrel_supported"] = supported
    if not supported:
        return None
    try:
        g1, b1 = _ln_params(module.n1)
        _, n1 = kernels.layer_norm(x, g1, b1, module.n1.eps)

        a = None
        if module.btype == "spatial":
            a = _window_attention(module.attn, n1, H, W)
        elif _chan_attn:
            a = _channel_attention(module.attn, n1)
        if a is None:
            a = module.attn(n1, H, W)

        conv = kernels.dwconv_tokens(
            n1, 0, c, H, W, _dw_weight(module.conv[0]), None, gelu=True
        )
        # spatial blocks use "channel_modulates_spatial" (channel gate) and
        # channel blocks "spatial_modulates_channel" (spatial gate)
        spatial_gate = module.btype != "spatial"
        g2, b2 = _ln_params(module.n2)
        if _fused_aim:
            # the gated mix inside the second LayerNorm's residual add
            s, n2 = kernels.gated_mix_layer_norm(
                x,
                a,
                conv,
                g2,
                b2,
                module.n2.eps,
                **_aim_gate(module.inter, conv, spatial_gate),
            )
        else:
            fused = _interaction(module.inter, a, conv, spatial_gate)
            s, n2 = kernels.layer_norm(x, g2, b2, module.n2.eps, residual=fused)

        ffn = module.ffn
        h = ffn.fc1(n2)
        h = kernels.dwconv_tokens(
            h,
            0,
            h.shape[-1],
            H,
            W,
            _dw_weight(ffn.smix),
            None,
            gelu=False,
            gelu_in=True,
        )
        if ffn.fc2.bias is None and _fused_res:
            # the residual add in the GEMM's epilogue (beta = 1)
            return torch.addmm(
                s.view(-1, c), h.view(-1, h.shape[-1]), ffn.fc2.weight.t()
            ).view(s.shape)
        return s + ffn.fc2(h)
    except Exception as e:  # noqa: BLE001
        _failed = True
        logger.warning("HIP FDAT block failed, falling back: %s", e)
        return None
