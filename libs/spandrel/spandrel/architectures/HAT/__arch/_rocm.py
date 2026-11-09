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

The CAB convolutions stay on MIOpen. When PyTorch runs channels_last
convolutions in NHWC (PYTORCH_MIOPEN_SUGGEST_NHWC=1), they read the token rows
in place; otherwise the tokens are transposed to NCHW by a tiled HIP kernel,
not PyTorch's strided copy. The same goes for the 3x3 conv at the end of every
RHAG. Returns None when it cannot be used, and HAT.py runs its original code.
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


def _to_image(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Contiguous token rows (B, H*W, C) as a conv input (B, C, H, W)."""
    b, _, c = tokens.shape
    if kernels.nhwc_conv_available(tokens.device, tokens.dtype):
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2)  # channels_last view
    return kernels.transpose_last2(tokens).view(b, c, h, w)


def _to_tokens(img: torch.Tensor) -> torch.Tensor:
    """A conv output (B, C, H, W) as contiguous token rows (B, H*W, C)."""
    b, c, h, w = img.shape
    if kernels.is_channels_last(img):
        return img.permute(0, 2, 3, 1).reshape(b, h * w, c)  # a view
    return kernels.transpose_last2(img.contiguous().view(b, c, h * w))


def _prepare_convs(convs: list[nn.Conv2d], x: torch.Tensor) -> None:
    """Store conv weights channels_last once when convs run in NHWC."""
    if not kernels.nhwc_conv_available(x.device, x.dtype):
        return
    for conv in convs:
        if not conv.weight.is_contiguous(memory_format=torch.channels_last):
            conv.weight.data = conv.weight.data.contiguous(
                memory_format=torch.channels_last
            )


def _gemm_width(n: int) -> int:
    """
    n rounded up to a multiple of 64 when that adds at most 1/8: hipBLASLt is
    25-35% faster on RDNA4 with K = 192 than with HAT's K = 180, and zero
    channels do not change the products.
    """
    p = -(-n // 64) * 64
    return p if (p - n) * 8 <= n else n


def _plain_mlp(mlp: nn.Module) -> bool:
    """fc1 -> exact GELU -> fc2 with biases (dropout is a no-op here)."""
    return (
        isinstance(getattr(mlp, "fc1", None), nn.Linear)
        and isinstance(getattr(mlp, "fc2", None), nn.Linear)
        and isinstance(getattr(mlp, "act", None), nn.GELU)
        and mlp.act.approximate == "none"
        and mlp.fc1.bias is not None
        and mlp.fc2.bias is not None
    )


def _padded(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = t.new_zeros(shape)
    out[tuple(slice(0, n) for n in t.shape)] = t
    return out


def _linear(lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """lin(x[..., :in_features]) for x padded with zero channels."""
    k = x.shape[-1]
    if k == lin.in_features:
        return lin(x)
    w = _cached(
        lin,
        "_spandrel_wk",
        [lin.weight],
        lambda: _padded(lin.weight.detach(), (lin.out_features, k)),
    )
    return nn.functional.linear(x, w, lin.bias)


def _mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    mlp(x) for HAT's Mlp. With x padded to a wider K, fc1 is also padded to
    _gemm_width(hidden) outputs with zero weights and bias, so GELU(0) = 0
    gives fc2 a zero-padded K as well.
    """
    if x.shape[-1] == mlp.fc1.in_features:
        return mlp(x)
    fc1, fc2 = mlp.fc1, mlp.fc2
    k, hp = x.shape[-1], _gemm_width(fc1.out_features)

    def build() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            _padded(fc1.weight.detach(), (hp, k)),
            _padded(fc1.bias.detach(), (hp,)),
            _padded(fc2.weight.detach(), (fc2.out_features, hp)),
        )

    w1, b1, w2 = _cached(
        mlp, "_spandrel_padded", [fc1.weight, fc1.bias, fc2.weight], build
    )
    hidden = nn.functional.gelu(nn.functional.linear(x, w1, b1))
    return nn.functional.linear(hidden, w2, fc2.bias)


def _conv_width(n: int) -> int:
    """
    n rounded up to a multiple of 32 when that adds at most 1/8: MIOpen's NHWC
    kernels on RDNA4 are much faster for such channel counts (HAT-L's CAB
    conv 60 -> 180 takes 0.44 ms, 64 -> 192 0.26 ms; 180 -> 180 1.24 ms,
    192 -> 192 0.69 ms at 256x256).
    """
    p = -(-n // 32) * 32
    return p if (p - n) * 8 <= n else n


def _conv_weight(
    conv: nn.Conv2d, in_channels: int | None = None, out_channels: int | None = None
) -> torch.Tensor:
    """The weight channels_last, zero padded to in/out channels; cached."""
    cin = in_channels or conv.in_channels
    cout = out_channels or conv.out_channels

    def build() -> torch.Tensor:
        w = conv.weight.detach()
        if (cout, cin) != tuple(w.shape[:2]):
            w = _padded(w, (cout, cin, *w.shape[2:]))
        return w.contiguous(memory_format=torch.channels_last)

    return _cached(conv, f"_spandrel_w{cin}x{cout}", [conv.weight], build)


def _float_param(module: nn.Module, name: str, size: int | None = None) -> torch.Tensor:
    """A parameter as float32 contiguous, zero padded to size; cached."""
    t = getattr(module, name)

    def build() -> torch.Tensor:
        f = t.detach().float()
        if size is not None and size != f.shape[0]:
            f = _padded(f, (size,))
        return f.contiguous()

    return _cached(module, f"_spandrel_f32_{name}{size or ''}", [t], build)


def _fusable_cab(cab: nn.Module) -> bool:
    """conv3x3 (bias) -> exact GELU -> conv3x3 (bias) -> avg pool attention."""
    try:
        att = cab[3].attention
        return (
            cab[0].bias is not None
            and cab[2].bias is not None
            and isinstance(cab[1], nn.GELU)
            and cab[1].approximate == "none"
            and isinstance(att[0], nn.AdaptiveAvgPool2d)
            and att[0].output_size in (1, (1, 1))
        )
    except (AttributeError, TypeError, IndexError):
        return False


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
    r = _hab(module, x, None, x_size, rpi_sa)
    return None if r is None else r[0] + r[1]


def _hab(module: nn.Module, x, m, x_size, rpi_sa):  # noqa: ANN001
    """
    HAB.forward on the input x + m (m may be None) without its final residual
    add: returns (s, mlp_out) with HAB(x + m) = s + mlp_out, or None. The next
    block's first LayerNorm adds them (see blocks_forward).
    """
    global _failed
    # drop_path is a no-op outside training
    if _failed or module.training or not kernels.hip_kernels_available():
        return None
    x = x.contiguous()  # the first block of a group gets a transposed view
    if not kernels.can_layer_norm(x) or (m is not None and m.shape != x.shape):
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
    cab = module.conv_block.cab
    # The fused NHWC path: CAB convs on the token rows without bias kernels,
    # and LayerNorm outputs padded with zero channels to a GEMM-friendly width
    fused = kernels.nhwc_conv_available(x.device, x.dtype) and _fusable_cab(cab)
    kp = _gemm_width(c) if fused and _plain_mlp(module.mlp) else c
    try:
        g1, b1 = _ln_params(module.norm1)
        xs, n1 = kernels.layer_norm(
            x, g1, b1, module.norm1.eps, residual=m, out_channels=kp
        )
        if xs is not None:
            x = xs

        qkv = _linear(attn.qkv, n1).view(b, n, 3, c).permute(2, 0, 1, 3)
        if not kernels.can_window_attention(qkv, h, w, ws, ws, attn.num_heads):
            return None

        # CAB: the convs as in the original, the channel attention is applied
        # by the residual kernel below
        conv_bias = None
        if fused:
            # NHWC convs without bias: MIOpen would add it in separate
            # broadcast kernels. The first bias goes into one bias + GELU
            # pass, the second into the channel attention's average pool
            # (mean(y + b) = mean(y) + b) and into hab_ln below. The first
            # conv reads the padded rows, with zero weights for the padding.
            # The conv channels are padded too (60 -> 64 -> 192 for HAT-L):
            # zero weights and bias, GELU(0) = 0, so the padding stays zero.
            img = n1.view(b, h, w, kp).permute(0, 3, 1, 2)
            mid = _conv_width(cab[0].out_channels)
            cout = _conv_width(c)
            b0 = _float_param(cab[0], "bias", mid)
            conv_bias = _float_param(cab[2], "bias")
            y = nn.functional.conv2d(img, _conv_weight(cab[0], kp, mid), None, 1, 1)
            if not kernels.is_channels_last(y):
                return None
            kernels.bias_act_(y, b0, gelu=True)
            conv = nn.functional.conv2d(y, _conv_weight(cab[2], mid, cout), None, 1, 1)
            if not kernels.is_channels_last(conv):
                return None
            pooled = conv.mean((2, 3), keepdim=True, dtype=torch.float32)[:, :c]
            pooled = pooled + conv_bias.view(1, c, 1, 1)
            ca = cab[3].attention[1:](pooled.to(conv.dtype)).view(b, c).float()
        else:
            if not kernels.can_transpose(n1):
                return None
            _prepare_convs([cab[0], cab[2]], n1)
            conv = cab[2](cab[1](cab[0](_to_image(n1, h, w))))
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
            x,
            attn_x,
            conv,
            ca,
            float(module.conv_scale),
            g2,
            b2,
            module.norm2.eps,
            conv_bias=conv_bias,
            out_channels=kp,
        )
        return s, _mlp(module.mlp, n2)
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
    """OCAB.forward, or None if this path cannot be used (see _ocab)."""
    r = _ocab(module, x, None, x_size, rpi)
    return None if r is None else r[0] + r[1]


def _ocab(module: nn.Module, x, m, x_size, rpi):  # noqa: ANN001
    """
    Like _hab: OCAB on x + m, returning (s, mlp_out) or None.

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
    if m is not None and m.shape != x.shape:
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
        kp = _gemm_width(c) if _plain_mlp(module.mlp) else c
        g1, b1 = _ln_params(module.norm1)
        xs, n1 = kernels.layer_norm(
            x, g1, b1, module.norm1.eps, residual=m, out_channels=kp
        )
        if xs is not None:
            x = xs
        qkv = _linear(module.qkv, n1).view(b, n, 3, c).permute(2, 0, 1, 3)
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
            x,
            g2,
            b2,
            module.norm2.eps,
            residual=module.proj(out.view(b, n, c)),
            out_channels=kp,
        )
        assert s is not None
        return s, _mlp(module.mlp, n2)
    except Exception as e:  # noqa: BLE001
        _ocab_failed = True
        logger.warning("HIP OCAB kernel failed, falling back: %s", e)
        return None


_rhag_failed = False


def rhag_forward(module: nn.Module, x, x_size, params):  # noqa: ANN001
    """
    RHAG.forward, or None if this path cannot be used: the same ops, with the
    token <-> image conversions around its 3x3 conv done without PyTorch's
    strided copies (see _to_image).
    """
    global _rhag_failed
    if _rhag_failed or module.training or not kernels.hip_kernels_available():
        return None
    conv = module.conv
    if not isinstance(conv, nn.Conv2d) or not x.is_cuda:
        return None
    try:
        h, w = x_size
        c = x.shape[-1]
        cw = _conv_width(c)
        if (
            kernels.nhwc_conv_available(x.device, x.dtype)
            and conv.bias is not None
            and conv.padding_mode == "zeros"
            and conv.groups == 1
            and conv.in_channels == conv.out_channels == c
        ):
            # NHWC conv on the group output padded to cw channels (written
            # by the group's last residual add), bias and the group residual
            # in one pass
            pair = blocks_pair(module.residual_group, x, x_size, params)
            if pair is None:
                rows = kernels.pad_add(
                    module.residual_group(x, x_size, params), None, cw
                )
            else:
                rows = kernels.pad_add(pair[0], pair[1], cw)
            img = rows.view(x.shape[0], h, w, cw).permute(0, 3, 1, 2)
            y = nn.functional.conv2d(
                img,
                _conv_weight(conv, cw, cw),
                None,
                conv.stride,
                conv.padding,
                conv.dilation,
            )
            if kernels.is_channels_last(y):
                return kernels.bias_residual_rows(y, _float_param(conv, "bias"), x)
            return _to_tokens(y[:, :c] + conv.bias.view(1, c, 1, 1)) + x
        res = module.residual_group(x, x_size, params).contiguous()
        if not kernels.can_transpose(res):
            return None
        _prepare_convs([conv], res)
        img = _to_image(res, h, w)
        if kernels.is_channels_last(img) and conv.bias is not None:
            # NHWC conv: bias and the group residual in one pass
            y = nn.functional.conv2d(
                img,
                conv.weight,
                None,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
            if kernels.is_channels_last(y) and conv.padding_mode == "zeros":
                return kernels.bias_residual_rows(y, _float_param(conv, "bias"), x)
        return _to_tokens(conv(img)) + x
    except Exception as e:  # noqa: BLE001
        _rhag_failed = True
        logger.warning("HIP HAT group failed, falling back: %s", e)
        return None


def blocks_forward(module: nn.Module, x, x_size, params):  # noqa: ANN001
    """
    AttenBlocks.forward, or None if this path cannot be used: the HABs and the
    OCAB hand their last residual add to the next block's first LayerNorm.
    """
    pair = blocks_pair(module, x, x_size, params)
    if pair is None:
        return None
    return pair[0] if pair[1] is None else pair[0] + pair[1]


def blocks_pair(module: nn.Module, x, x_size, params):  # noqa: ANN001
    """Like blocks_forward, returning (s, m) with the result s + m, or None."""
    if module.training or not kernels.hip_kernels_available():
        return None
    if module.downsample is not None:
        return None
    s, m = x, None
    for blk in module.blocks:
        r = _hab(blk, s, m, x_size, params["rpi_sa"])
        if r is None:
            s, m = (
                blk(
                    s if m is None else s + m,
                    x_size,
                    params["rpi_sa"],
                    params["attn_mask"],
                ),
                None,
            )
        else:
            s, m = r
    r = _ocab(module.overlap_attn, s, m, x_size, params["rpi_oca"])
    if r is None:
        x_in = s if m is None else s + m
        return module.overlap_attn(x_in, x_size, params["rpi_oca"]), None
    return r
