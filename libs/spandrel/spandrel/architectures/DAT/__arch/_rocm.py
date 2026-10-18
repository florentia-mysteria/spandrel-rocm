"""
Faster inference paths for DAT.

The mask and position bias caches apply everywhere. The rest runs only with
custom HIP kernels on ROCm (see spandrel.__helpers.rocm); every function then
returns None when it cannot be used, and DAT.py runs its original code. Any
kernel failure disables that path for the rest of the process.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch.nn.functional as F
from torch import nn

from spandrel.__helpers.rocm import kernels
from spandrel.__helpers.rocm.common import is_depthwise_3x3, register_cache, version_of

logger = logging.getLogger("spandrel.rocm")

# --- shift masks -------------------------------------------------------------

# A single entry: all blocks of a model share it, and consecutive tiles almost
# always have the same size.
_mask_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_mask_lock = threading.Lock()


def cached_mask(
    module: nn.Module, H: int, W: int, dtype: torch.dtype | None = None  # noqa: N803
):
    """
    DAT rebuilds its shifted-window attention masks on the CPU in every shifted
    block whenever the tile size differs from the training size (i.e. always),
    then copies them to the GPU. That is 18 CPU mask builds and host-to-device
    copies per tile, which dominates the run time on fast GPUs.

    The mask is also always float32, so adding it to bfloat16 attention scores
    promotes them to float32 and the following `attn @ v` fails. That is why DAT
    could not run in bf16 at all.

    The mask only depends on the padded size, split size and shift size, which
    are the same for every block. So it is built once per tile size, directly on
    the model's device and in the model's dtype. The forward pass still calls
    `.to(x.device)` on it, which is then a no-op.
    """
    param = next(module.parameters(), None)
    if param is None:
        return module._build_mask(H, W, dtype)
    dtype = dtype or param.dtype

    key = (
        H,
        W,
        tuple(module.split_size),
        tuple(module.shift_size),
        param.device,
        dtype,
    )
    with _mask_lock:
        masks = _mask_cache.get(key)
        if masks is None:
            mask_0, mask_1 = module._build_mask(H, W)
            masks = (
                mask_0.to(param.device, dtype),
                mask_1.to(param.device, dtype),
            )
            _mask_cache.clear()
            _mask_cache[key] = masks
        return masks


def _clear_masks() -> None:
    with _mask_lock:
        _mask_cache.clear()


register_cache(_clear_masks)

# --- position bias -----------------------------------------------------------


def cached_pos_bias(module: nn.Module, biases: torch.Tensor) -> torch.Tensor:
    """
    DynamicPosBias runs a small MLP over a constant table of relative
    coordinates to get the attention position bias. The result only depends on
    the weights, yet it is recomputed in every attention call (36 times per tile
    for DAT-2), about 5% of the run time. Cache it until the weights, device or
    dtype change.
    """
    if module.training or torch.is_grad_enabled():
        return module._compute(biases)
    key = (
        biases.data_ptr(),
        biases.dtype,
        version_of(biases),
        tuple((p.data_ptr(), version_of(p)) for p in module.parameters()),
    )
    cached = module.__dict__.get("_spandrel_pos_cache")
    if cached is None or cached[0] != key:
        cached = (key, module._compute(biases))
        module.__dict__["_spandrel_pos_cache"] = cached
    return cached[1]


# --- window attention ----------------------------------------------------------

_attention_failed = False
_label_cache: dict[tuple, torch.Tensor] = {}
register_cache(_label_cache.clear)


def _bias_t(attn: nn.Module) -> torch.Tensor | None:
    """A Spatial_Attention's position bias as float32 (heads, key, query)."""
    if not attn.position_bias:
        return None
    pos = attn.pos(attn.rpe_biases)  # cached, see cached_pos_bias
    cached = attn.__dict__.get("_spandrel_bias_t")
    key = (pos.data_ptr(), version_of(pos), pos.dtype)
    if cached is None or cached[0] != key:
        n = attn.H_sp * attn.W_sp
        rpb = pos[attn.relative_position_index.view(-1)].view(n, n, -1)  # q k h
        cached = (key, rpb.permute(2, 1, 0).float().contiguous())  # h k q
        attn.__dict__["_spandrel_bias_t"] = cached
    return cached[1]


def _mask_labels(mask: torch.Tensor) -> torch.Tensor:
    """
    DAT's shift mask (nW, N, N) as one region id per token (nW, N): the index
    of the first token it is not masked against. Two tokens share a region
    exactly when their ids are equal, which is all the kernel needs.
    """
    key = (mask.data_ptr(), version_of(mask), tuple(mask.shape))
    labels = _label_cache.get(key)
    if labels is None:
        labels = (mask == 0).int().argmax(dim=-1).int().contiguous()
        _label_cache.clear()
        _label_cache[key] = labels
    return labels


def spatial_attention(module: nn.Module, qkv, H, W, mask):  # noqa: N803
    """
    Spatial_Attention as one fused HIP kernel.

    The original computes q @ k^T, adds the position bias and the shift mask,
    applies softmax and multiplies by v, writing the full attention matrix of
    every window to memory and reading it back several times (~100 MB per call
    on a 256px tile), plus permutes into and out of the window layout. The
    kernel keeps everything in registers and shared memory, reads q/k/v in place
    and writes the result straight into the image layout.

    The shift mask is passed as one region id per token instead of an N x N
    matrix: DAT's mask is -100 exactly where two tokens of a window belong to
    different shift regions, which the kernel reproduces by comparing ids.
    """
    global _attention_failed
    if (
        _attention_failed
        or module.training
        or not kernels.hip_kernels_available()
        or not kernels.can_window_attention(
            qkv, H, W, module.H_sp, module.W_sp, module.num_heads
        )
    ):
        return None
    try:
        return kernels.window_attention(
            qkv,
            H,
            W,
            module.H_sp,
            module.W_sp,
            module.num_heads,
            float(module.scale),
            _bias_t(module),
            _mask_labels(mask) if mask is not None else None,
        )
    except Exception as e:  # noqa: BLE001
        _attention_failed = True
        logger.warning("HIP window attention kernel failed, falling back: %s", e)
        return None


# --- spatial gate --------------------------------------------------------------

_spatial_gate_failed = False


def _sg_params(module: nn.Module):
    norm, conv = module.norm, module.conv
    tensors = [norm.weight, norm.bias, conv.weight] + (
        [conv.bias] if conv.bias is not None else []
    )
    key = tuple((t.data_ptr(), version_of(t), t.dtype) for t in tensors)
    cached = module.__dict__.get("_spandrel_sg_params")
    if cached is None or cached[0] != key:
        c = conv.in_channels
        params = (
            norm.weight.detach().float().contiguous(),
            norm.bias.detach().float().contiguous(),
            conv.weight.detach().float().reshape(c, 9).contiguous(),
            conv.bias.detach().float().contiguous() if conv.bias is not None else None,
        )
        cached = (key, params)
        module.__dict__["_spandrel_sg_params"] = cached
    return cached[1]


def spatial_gate(module: nn.Module, x, H, W):  # noqa: N803
    """
    SpatialGate (x1 * dwconv3x3(LayerNorm(x2))) as two HIP kernels.

    The original normalizes x2, transposes it to (B, C, H, W) with a copy,
    convolves, transposes back with another copy and multiplies: five passes
    over the tensor. The kernels work directly in the (B, H*W, 2C) token layout:
    one computes the LayerNorm statistics, the other normalizes on the fly while
    loading, convolves and applies the gate.
    """
    global _spatial_gate_failed
    if (
        _spatial_gate_failed
        or module.training
        or module.norm.weight is None
        or module.norm.bias is None
        or not is_depthwise_3x3(module.conv)
        or not kernels.hip_kernels_available()
        or not kernels.can_spatial_gate(x, H, W)
    ):
        return None
    try:
        gamma, beta, weight, bias = _sg_params(module)
        return kernels.spatial_gate(x, H, W, gamma, beta, module.norm.eps, weight, bias)
    except Exception as e:  # noqa: BLE001
        _spatial_gate_failed = True
        logger.warning("HIP spatial gate kernel failed, falling back: %s", e)
        return None


# --- adaptive interaction ------------------------------------------------------

_aim_failed = False


def _bn_scale_shift(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    inv = (bn.running_var.float() + bn.eps).rsqrt()  # type: ignore[union-attr]
    scale = bn.weight.float() * inv if bn.affine else inv
    shift = (bn.bias.float() if bn.affine else 0.0) - bn.running_mean.float() * scale  # type: ignore[union-attr]
    return scale, shift


def _aim_supported(module: nn.Module) -> bool:
    """Checks the layout of DAT's conv branch and Adaptive Interaction Module."""
    try:
        dw, ci, si = (
            module.dwconv,
            module.channel_interaction,
            module.spatial_interaction,
        )
        return (
            len(dw) == 3
            and is_depthwise_3x3(dw[0])
            and isinstance(dw[1], nn.BatchNorm2d)
            and isinstance(dw[2], nn.GELU)
            and dw[2].approximate == "none"
            and len(ci) == 5
            and isinstance(ci[0], nn.AdaptiveAvgPool2d)
            and len(si) == 4
            and isinstance(si[0], nn.Conv2d)
            and si[0].kernel_size == (1, 1)
            and isinstance(si[1], nn.BatchNorm2d)
            and isinstance(si[2], nn.GELU)
            and si[2].approximate == "none"
            and isinstance(si[3], nn.Conv2d)
            and si[3].kernel_size == (1, 1)
        )
    except (AttributeError, TypeError):
        return False


def _aim_params(module: nn.Module, dtype: torch.dtype):
    """
    The conv branch's BatchNorm folded into the depthwise conv, and the spatial
    interaction's first 1x1 conv + BatchNorm folded into a linear layer. In eval
    mode this is an exact rewrite.
    """
    dw, si = module.dwconv, module.spatial_interaction
    tensors = [
        t for m in (dw[0], dw[1], si[0], si[1], si[3]) for t in m.state_dict().values()
    ]
    key = (dtype, tuple((t.data_ptr(), version_of(t)) for t in tensors))
    cached = module.__dict__.get("_spandrel_aim_params")
    if cached is not None and cached[0] == key:
        return cached[1]

    conv = dw[0]
    c = conv.in_channels
    scale, shift = _bn_scale_shift(dw[1])
    dw_w = (conv.weight.detach().float().reshape(c, 9) * scale[:, None]).contiguous()
    dw_b = (
        (conv.bias.detach().float() if conv.bias is not None else 0.0) * scale + shift
    ).contiguous()

    s_scale, s_shift = _bn_scale_shift(si[1])
    w1 = si[0].weight.detach().float().reshape(si[0].out_channels, -1) * s_scale[:, None]
    b1 = (si[0].bias.detach().float() if si[0].bias is not None else 0.0) * s_scale + s_shift
    w2 = si[3].weight.detach().float().reshape(1, -1)
    b2 = si[3].bias.detach().float() if si[3].bias is not None else None
    params = (
        dw_w,
        dw_b,
        w1.to(dtype),
        b1.to(dtype),
        w2.to(dtype),
        b2.to(dtype) if b2 is not None else None,
    )
    module.__dict__["_spandrel_aim_params"] = (key, params)
    return params


def _channel_map(module: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """channel_interaction(t as an image) as (B, 1, C), from the token layout."""
    B, _, C = t.shape
    pooled = torch.mean(t, dim=1, dtype=torch.float32).to(t.dtype).view(B, C, 1, 1)
    out = pooled
    for m in list(module.channel_interaction)[1:]:
        out = m(out)
    return out.view(B, 1, C)


def _spatial_map(t: torch.Tensor, params) -> torch.Tensor:
    """spatial_interaction(t as an image) as (B, L, 1), from the token layout."""
    _, _, w1, b1, w2, b2 = params
    return F.linear(F.gelu(F.linear(t, w1, b1)), w2, b2)


def _aim_usable(module: nn.Module, x: torch.Tensor) -> bool:
    return (
        not _aim_failed
        and not module.training
        and x.is_cuda
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and not x.requires_grad
        and kernels.hip_kernels_available()
        and _aim_supported(module)
    )


def _asa_masks(module: nn.Module, H: int, W: int, device: torch.device):  # noqa: N803
    """The two shift masks DAT uses for a padded H x W, as in the original forward."""
    if module.patches_resolution != H or module.patches_resolution != W:
        mask_tmp = module.calculate_mask(H, W)
        return mask_tmp[0].to(device), mask_tmp[1].to(device)
    return module.attn_mask_0, module.attn_mask_1


_shift_label_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
register_cache(_shift_label_cache.clear)


def _shift_labels(module: nn.Module, H: int, W: int, device: torch.device):  # noqa: N803
    """
    The shift-window region id of every token, per window, for both branches:
    exactly the `mask_windows` that DAT's calculate_mask builds before turning
    them into N x N masks. The kernel only compares ids, so the masks are never
    built. Tiles of a few different sizes alternate, so several are cached; they
    are only H*W ints each.
    """
    key = (H, W, tuple(module.split_size), tuple(module.shift_size), device)
    labels = _shift_label_cache.get(key)
    if labels is None:

        def regions(sh: int, sw: int, th: int, tw: int) -> torch.Tensor:
            img = torch.zeros(H, W, dtype=torch.int32)
            cnt = 0
            for hs in (slice(0, -sh), slice(-sh, -th), slice(-th, None)):
                for ws in (slice(0, -sw), slice(-sw, -tw), slice(-tw, None)):
                    img[hs, ws] = cnt
                    cnt += 1
            windows = img.view(H // sh, sh, W // sw, sw).permute(0, 2, 1, 3)
            return windows.reshape(-1, sh * sw).contiguous().to(device)

        s0, s1 = module.split_size[0], module.split_size[1]
        t0, t1 = module.shift_size[0], module.shift_size[1]
        labels = (regions(s0, s1, t0, t1), regions(s1, s0, t1, t0))
        if len(_shift_label_cache) >= 16:
            _shift_label_cache.pop(next(iter(_shift_label_cache)))
        _shift_label_cache[key] = labels
    return labels


def _asa_fused_attention(module, qkv, B, H, W, _H, _W, C, shifted, device):  # noqa: N803
    """
    Both window attention branches straight into one (B, H*W, C) tensor, or
    None if the fused kernel cannot be used. Shifted windows are handled by the
    kernel's read/write offsets instead of torch.roll, the window padding by
    treating positions outside H x W as zero instead of F.pad, and the two
    branches write their channel halves directly: no roll, pad, crop or cat.

    qkv: (3, B, H*W, C), unpadded. _H x _W: the window grid (padded size).
    """
    attn0, attn1 = module.attns[0], module.attns[1]
    q0, q1 = qkv[..., : C // 2], qkv[..., C // 2 :]
    if _attention_failed or not all(
        kernels.can_window_attention(q, _H, _W, a.H_sp, a.W_sp, a.num_heads)
        for q, a in ((q0, attn0), (q1, attn1))
    ):
        return None
    if shifted:
        labels = _shift_labels(module, _H, _W, device)
        s0, s1 = module.shift_size[0], module.shift_size[1]
        shifts = ((s0, s1), (s1, s0))
    else:
        labels = (None, None)
        shifts = ((0, 0), (0, 0))
    out = torch.empty((B, H, W, C), device=qkv.device, dtype=qkv.dtype)
    for i, (q, a) in enumerate(((q0, attn0), (q1, attn1))):
        kernels.window_attention(
            q,
            _H,
            _W,
            a.H_sp,
            a.W_sp,
            a.num_heads,
            float(a.scale),
            _bias_t(a),
            labels[i],
            out=out,
            out_offset=i * (C // 2),
            shift=shifts[i],
        )
    return out.view(B, H * W, C)


def adaptive_spatial_attention(module: nn.Module, x, H, W):  # noqa: N803
    """
    Adaptive_Spatial_Attention and Adaptive_Channel_Attention both run a
    depthwise conv branch next to their attention and mix the two through the
    Adaptive Interaction Module. The original does that in the (B, C, H, W)
    image layout, which costs four full transposes per module (each ~1 ms on a
    256px tile with PyTorch on ROCm) for work that is mostly elementwise.

    This runs the conv branch with a HIP kernel straight from the qkv buffer in
    the token layout, with the BatchNorm folded in, and does the interaction in
    the token layout too: the average pool becomes a mean over tokens, and the
    1x1 convs become linear layers. The attention part is unchanged.
    """
    global _aim_failed
    if not _aim_usable(module, x):
        return None
    try:
        B, L, C = x.shape
        qkv_lin = module.qkv(x)  # B, L, 3C
        qkv = qkv_lin.reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, L, C

        max_split_size = max(module.split_size[0], module.split_size[1])
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size
        _H = pad_b + H  # noqa: N806
        _W = pad_r + W  # noqa: N806
        _L = _H * _W  # noqa: N806

        shifted = (
            module.rg_idx % 2 == 0 and module.b_idx > 0 and (module.b_idx - 2) % 4 == 0
        ) or (module.rg_idx % 2 != 0 and module.b_idx % 4 == 0)

        attened_x = _asa_fused_attention(module, qkv, B, H, W, _H, _W, C, shifted, x.device)
        if attened_x is None:
            # same as the original
            if pad_r or pad_b:
                qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
                qkv = (
                    F.pad(qkv, (0, pad_r, 0, pad_b))
                    .reshape(3, B, C, -1)
                    .transpose(-2, -1)
                )
            if shifted:
                qkv = qkv.view(3, B, _H, _W, C)
                qkv_0 = torch.roll(
                    qkv[:, :, :, :, : C // 2],
                    shifts=(-module.shift_size[0], -module.shift_size[1]),
                    dims=(2, 3),
                ).view(3, B, _L, C // 2)
                qkv_1 = torch.roll(
                    qkv[:, :, :, :, C // 2 :],
                    shifts=(-module.shift_size[1], -module.shift_size[0]),
                    dims=(2, 3),
                ).view(3, B, _L, C // 2)
                mask_0, mask_1 = _asa_masks(module, _H, _W, x.device)
                x1_shift = module.attns[0](qkv_0, _H, _W, mask=mask_0)
                x2_shift = module.attns[1](qkv_1, _H, _W, mask=mask_1)
                x1 = torch.roll(
                    x1_shift,
                    shifts=(module.shift_size[0], module.shift_size[1]),
                    dims=(1, 2),
                )
                x2 = torch.roll(
                    x2_shift,
                    shifts=(module.shift_size[1], module.shift_size[0]),
                    dims=(1, 2),
                )
            else:
                x1 = module.attns[0](qkv[:, :, :, : C // 2], _H, _W)
                x2 = module.attns[1](qkv[:, :, :, C // 2 :], _H, _W)
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)

        params = _aim_params(module, x.dtype)
        conv_x = kernels.dwconv_tokens(
            qkv_lin, 2 * C, C, H, W, params[0], params[1], gelu=True
        )
        channel_map = _channel_map(module, conv_x)  # B, 1, C
        spatial_map = _spatial_map(attened_x, params)  # B, L, 1
        x = kernels.gate_mix(attened_x, conv_x, spatial_map, channel_map, a_uses_tok=False)
        return module.proj_drop(module.proj(x))
    except Exception as e:  # noqa: BLE001
        _aim_failed = True
        logger.warning("HIP adaptive interaction failed, falling back: %s", e)
        return None


def adaptive_channel_attention(module: nn.Module, x, H, W):  # noqa: N803
    """See adaptive_spatial_attention."""
    global _aim_failed
    if not _aim_usable(module, x):
        return None
    try:
        B, N, C = x.shape
        heads = module.num_heads
        qkv_lin = module.qkv(x)  # B, N, 3C
        q = qkv_lin[..., :C].view(B, N, heads, C // heads).permute(0, 2, 3, 1)
        k = qkv_lin[..., C : 2 * C].view(B, N, heads, C // heads).permute(0, 2, 3, 1)
        v = qkv_lin[..., 2 * C :].view(B, N, heads, C // heads).permute(0, 2, 3, 1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * module.temperature
        attn = module.attn_drop(attn.softmax(dim=-1))
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        params = _aim_params(module, x.dtype)
        conv_x = kernels.dwconv_tokens(
            qkv_lin, 2 * C, C, H, W, params[0], params[1], gelu=True
        )
        channel_map = _channel_map(module, attened_x)  # B, 1, C
        spatial_map = _spatial_map(conv_x, params)  # B, N, 1
        x = kernels.gate_mix(attened_x, conv_x, spatial_map, channel_map, a_uses_tok=True)
        return module.proj_drop(module.proj(x))
    except Exception as e:  # noqa: BLE001
        _aim_failed = True
        logger.warning("HIP adaptive interaction failed, falling back: %s", e)
        return None


# --- DAT block -------------------------------------------------------------------

_ln_failed = False


def _ln_params(norm: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    key = tuple((t.data_ptr(), version_of(t)) for t in (norm.weight, norm.bias))
    cached = norm.__dict__.get("_spandrel_ln_params")
    if cached is None or cached[0] != key:
        cached = (
            key,
            (
                norm.weight.detach().float().contiguous(),
                norm.bias.detach().float().contiguous(),
            ),
        )
        norm.__dict__["_spandrel_ln_params"] = cached
    return cached[1]


def _block_usable(module: nn.Module, x: torch.Tensor) -> bool:
    c = x.shape[-1]
    return (
        not _ln_failed
        and not module.training
        and kernels.hip_kernels_available()
        and kernels.can_layer_norm(x)
        and all(
            isinstance(n, nn.LayerNorm)
            and n.elementwise_affine
            and n.bias is not None
            and tuple(n.normalized_shape) == (c,)
            for n in (module.norm1, module.norm2)
        )
    )


def dat_block(module: nn.Module, x, x_size):
    """
    DATB: x + attn(norm1(x)), then + ffn(norm2(.)). PyTorch's LayerNorm on ROCm
    runs at a fraction of memory bandwidth here (~0.36 ms on a 256px tile, twice
    per block), and the residual add before norm2 is one more full pass. A HIP
    kernel does the LayerNorms, with the first residual add fused into the
    second one.
    """
    global _ln_failed
    if not _block_usable(module, x):
        return None
    H, W = x_size  # noqa: N806
    try:
        g1, b1 = _ln_params(module.norm1)
        g2, b2 = _ln_params(module.norm2)
        _, n1 = kernels.layer_norm(x, g1, b1, module.norm1.eps)
        a = module.drop_path(module.attn(n1, H, W))
        s, n2 = kernels.layer_norm(x, g2, b2, module.norm2.eps, residual=a)
        return s + module.drop_path(module.ffn(n2, H, W))
    except Exception as e:  # noqa: BLE001
        _ln_failed = True
        logger.warning("HIP LayerNorm kernel failed, falling back: %s", e)
        return None
