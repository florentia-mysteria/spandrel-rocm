from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import sys
import threading
from pathlib import Path

import torch

logger = logging.getLogger("spandrel.rocm")

# Custom HIP kernels for operations that MIOpen handles poorly on consumer
# Radeon cards. They are compiled at runtime with HIPRTC, which ships with
# AMD's ROCm wheels for Windows and Linux, for whatever GPU architecture is
# present, and launched on
# PyTorch's current stream. No compiler or SDK has to be installed.

_DWCONV_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
    p[i] = (bf16_t)(u >> 16);
}

#define TW 32
#define TH 16

// Depthwise 3x3 convolution, stride 1, zero padding 1, NCHW layout.
// Each block loads a (TH+2) x (TW+2) tile of one channel into shared memory,
// so every input value is read from global memory about once. Accumulates
// in fp32 regardless of the storage type.
template <typename T>
__device__ void dwconv3x3(const T* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ b, T* __restrict__ y,
                          int C, int H, int W) {
    __shared__ float tile[TH + 2][TW + 2];
    const int bc = blockIdx.z;  // batch * C + channel
    const int c = bc % C;
    const long plane = (long)bc * H * W;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
    const int tid = threadIdx.y * TW + threadIdx.x;

    for (int i = tid; i < (TH + 2) * (TW + 2); i += TW * TH) {
        int ty = i / (TW + 2), tx = i % (TW + 2);
        int gy = y0 + ty - 1, gx = x0 + tx - 1;
        tile[ty][tx] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
            ? load_val(x, plane + (long)gy * W + gx) : 0.0f;
    }
    __syncthreads();

    const int ox = x0 + threadIdx.x, oy = y0 + threadIdx.y;
    if (ox >= W || oy >= H) return;
    const float* k = w + c * 9;
    float acc = b ? b[c] : 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            acc += tile[threadIdx.y + i][threadIdx.x + j] * k[i * 3 + j];
    store_val(y, plane + (long)oy * W + ox, acc);
}

extern "C" __global__ void dwconv3x3_f32(const float* x, const float* w, const float* b, float* y, int C, int H, int W) { dwconv3x3<float>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_f16(const _Float16* x, const float* w, const float* b, _Float16* y, int C, int H, int W) { dwconv3x3<_Float16>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_bf16(const bf16_t* x, const float* w, const float* b, bf16_t* y, int C, int H, int W) { dwconv3x3<bf16_t>(x, w, b, y, C, H, W); }
"""


_ATTN_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    p[i] = (bf16_t)(u >> 16);
}
__device__ __forceinline__ void copy_val(float* d, long di, const float* s, long si) { d[di] = s[si]; }
__device__ __forceinline__ void copy_val(_Float16* d, long di, const _Float16* s, long si) { d[di] = s[si]; }
__device__ __forceinline__ void copy_val(bf16_t* d, long di, const bf16_t* s, long si) { d[di] = s[si]; }

#define CH 16

typedef float f2v __attribute__((ext_vector_type(2)));
typedef unsigned int u2v __attribute__((ext_vector_type(2)));
typedef _Float16 h4v __attribute__((ext_vector_type(4)));

// 8-byte read from shared memory, unpacked to floats
__device__ __forceinline__ void load8(const float* p, float* o) {
    f2v v = *(const f2v*)p; o[0] = v.x; o[1] = v.y;
}
__device__ __forceinline__ void load8(const _Float16* p, float* o) {
    h4v v = *(const h4v*)p; o[0] = (float)v.x; o[1] = (float)v.y; o[2] = (float)v.z; o[3] = (float)v.w;
}
__device__ __forceinline__ void load8(const bf16_t* p, float* o) {
    u2v v = *(const u2v*)p;
    o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xFFFF0000u);
    o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xFFFF0000u);
}

// DAT window attention, fused: softmax(q*scale @ k^T + bias + shift_mask) @ v
// for one (window, head) per block and one query token per thread.
//
// qkv:    (3, B, L, heads*d) with arbitrary strides s0..s3 (elements), L = H*W
// biasT:  (heads, N, N) float, transposed: biasT[h][key][query], or null
// labels: (nW, N) int region id per token of each window, or null. Tokens of
//         different regions get -100 added, exactly like DAT's shift mask.
// out:    (B, H, W, heads*d) contiguous
//
// K and V rows are padded to DP (a multiple of the 8-byte vector width) and
// zero filled, so the inner loops read shared memory 8 bytes at a time.
template <typename T, int DMAX>
__device__ void win_attn(const T* __restrict__ qkv, long s0, long s1, long s2, long s3,
                         const float* __restrict__ biasT, const int* __restrict__ labels,
                         T* __restrict__ out, long out_stride, int out_off, int Hc, int Wc,
                         int sy, int sx, int H, int W, int H_sp, int W_sp,
                         int heads, int d, float scale) {
    constexpr int VEC = 8 / sizeof(T);
    extern __shared__ unsigned char lds_raw[];
    const int N = H_sp * W_sp;
    const int DP = (d + VEC - 1) / VEC * VEC;
    T* Ks = (T*)lds_raw;
    T* Vs = Ks + N * DP;
    int* labs = (int*)(Vs + N * DP);

    const int nw = W / W_sp, nh = H / H_sp, nW = nh * nw;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int ih = wimg / nw, iw = wimg % nw;
    const int n = threadIdx.x;
    // (y, x) in the (rolled) window grid; (ys, xs) where that token really is.
    // A roll by -s before attention and by +s after is the same as reading and
    // writing at (y + s) mod size, so DAT's torch.roll copies are not needed.
    const int y = ih * H_sp + n / W_sp, x = iw * W_sp + n % W_sp;
    const int ys = (y + sy) % H, xs = (x + sx) % W;
    // positions past the Hc x Wc image are the window padding: zero q/k/v,
    // exactly like DAT's F.pad, without materializing the padded copy
    const bool valid = ys < Hc && xs < Wc;
    const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + (long)(h * d) * s3;

    float q[DMAX];
    #pragma unroll
    for (int j = 0; j < DMAX; j++) {
        if (j < d && valid) {
            q[j] = load_val(qkv, base + j * s3) * scale;
            copy_val(Ks, n * DP + j, qkv, s0 + base + j * s3);
            copy_val(Vs, n * DP + j, qkv, 2 * s0 + base + j * s3);
        } else {
            q[j] = 0.0f;
            if (j < DP) { store_val(Ks, n * DP + j, 0.0f); store_val(Vs, n * DP + j, 0.0f); }
        }
    }
    if (labels) labs[n] = labels[wimg * N + n];
    __syncthreads();

    const int lab = labels ? labs[n] : 0;
    const float* bcol = biasT ? biasT + (long)h * N * N + n : nullptr;
    float m = -__builtin_inff(), lsum = 0.0f, acc[DMAX];
    #pragma unroll
    for (int j = 0; j < DMAX; j++) acc[j] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += CH) {
        float s[CH];
        float cm = -__builtin_inff();
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            const T* krow = Ks + (k0 + c) * DP;
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < DMAX; j += VEC) {
                if (j < DP) {
                    float kv[VEC];
                    load8(krow + j, kv);
                    #pragma unroll
                    for (int e = 0; e < VEC; e++) dot += q[j + e] * kv[e];
                }
            }
            if (bcol) dot += bcol[(long)(k0 + c) * N];
            if (labels && labs[k0 + c] != lab) dot -= 100.0f;
            s[c] = dot;
            cm = fmaxf(cm, dot);
        }
        const float mn = fmaxf(m, cm);
        const float corr = __expf(m - mn);
        lsum *= corr;
        #pragma unroll
        for (int j = 0; j < DMAX; j++) acc[j] *= corr;
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            const float p = __expf(s[c] - mn);
            lsum += p;
            const T* vrow = Vs + (k0 + c) * DP;
            #pragma unroll
            for (int j = 0; j < DMAX; j += VEC) {
                if (j < DP) {
                    float vv[VEC];
                    load8(vrow + j, vv);
                    #pragma unroll
                    for (int e = 0; e < VEC; e++) acc[j + e] += p * vv[e];
                }
            }
        }
        m = mn;
    }

    // write into (B, Hc, Wc, out_stride) at channel out_off, cropping padding
    if (ys >= Hc || xs >= Wc) return;
    const float inv = 1.0f / lsum;
    const long ob = (((long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
    #pragma unroll
    for (int j = 0; j < DMAX; j++)
        if (j < d) store_val(out, ob + j, acc[j] * inv);
}

#define ATTN_ARGS long s0, long s1, long s2, long s3, const float* biasT, const int* labels, long out_stride, int out_off, int Hc, int Wc, int sy, int sx, int H, int W, int H_sp, int W_sp, int heads, int d, float scale
#define ATTN_PASS s0, s1, s2, s3, biasT, labels
#define ATTN_TAIL out_stride, out_off, Hc, Wc, sy, sx, H, W, H_sp, W_sp, heads, d, scale
extern "C" __global__ void win_attn_f32_32(const float* qkv, float* out, ATTN_ARGS) { win_attn<float, 32>(qkv, ATTN_PASS, out, ATTN_TAIL); }
extern "C" __global__ void win_attn_f32_64(const float* qkv, float* out, ATTN_ARGS) { win_attn<float, 64>(qkv, ATTN_PASS, out, ATTN_TAIL); }
extern "C" __global__ void win_attn_f16_32(const _Float16* qkv, _Float16* out, ATTN_ARGS) { win_attn<_Float16, 32>(qkv, ATTN_PASS, out, ATTN_TAIL); }
extern "C" __global__ void win_attn_f16_64(const _Float16* qkv, _Float16* out, ATTN_ARGS) { win_attn<_Float16, 64>(qkv, ATTN_PASS, out, ATTN_TAIL); }
extern "C" __global__ void win_attn_bf16_32(const bf16_t* qkv, bf16_t* out, ATTN_ARGS) { win_attn<bf16_t, 32>(qkv, ATTN_PASS, out, ATTN_TAIL); }
extern "C" __global__ void win_attn_bf16_64(const bf16_t* qkv, bf16_t* out, ATTN_ARGS) { win_attn<bf16_t, 64>(qkv, ATTN_PASS, out, ATTN_TAIL); }
"""

_SG_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    p[i] = (bf16_t)(u >> 16);
}

// DAT's SpatialGate, in the token layout x: (B, L=H*W, 2C), rows contiguous.
//   x1, x2 = x[..., :C], x[..., C:]
//   out = x1 * dwconv3x3(LayerNorm(x2) as an image)
// without the two transposes to and from (B, C, H, W).

// Kernel A: LayerNorm statistics of x2, one (mean, rstd) pair per token.
// 8 tokens per 256-thread block, 32 threads per token.
template <typename T>
__device__ void sg_stats(const T* __restrict__ x, int C, long tokens, float eps,
                         float* __restrict__ stats) {
    __shared__ float s1[256], s2[256];
    const int lane = threadIdx.x & 31;
    const long tok = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    float a = 0.0f, q = 0.0f;
    if (tok < tokens) {
        const T* row = x + tok * (2L * C) + C;
        for (int c = lane; c < C; c += 32) {
            float v = load_val(row, c);
            a += v;
            q += v * v;
        }
    }
    s1[threadIdx.x] = a;
    s2[threadIdx.x] = q;
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) {
        if (lane < o) {
            s1[threadIdx.x] += s1[threadIdx.x + o];
            s2[threadIdx.x] += s2[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (lane == 0 && tok < tokens) {
        const float mean = s1[threadIdx.x] / C;
        const float var = fmaxf(s2[threadIdx.x] / C - mean * mean, 0.0f);
        stats[tok * 2] = mean;
        stats[tok * 2 + 1] = rsqrtf(var + eps);
    }
}

// Kernel B: normalize x2 on the fly, depthwise 3x3 conv (zero padded), gate.
// Block: 8x8 pixels x 32 channels, 256 threads = 32 channels x 8 columns.
#define SX 8
#define SY 8
#define CC 32
template <typename T>
__device__ void sg_gate(const T* __restrict__ x, const float* __restrict__ stats,
                        const float* __restrict__ gamma, const float* __restrict__ beta,
                        const float* __restrict__ w, const float* __restrict__ bias,
                        T* __restrict__ out, int C, int H, int W) {
    __shared__ float tile[(SY + 2) * (SX + 2) * CC];
    const int chunks = (C + CC - 1) / CC;
    const int b = blockIdx.z / chunks;
    const int c0 = (blockIdx.z % chunks) * CC;
    const int x0 = blockIdx.x * SX, y0 = blockIdx.y * SY;
    const long L = (long)H * W;
    const long C2 = 2L * C;

    for (int i = threadIdx.x; i < (SY + 2) * (SX + 2) * CC; i += 256) {
        const int c = i % CC, p = i / CC;
        const int gy = y0 + p / (SX + 2) - 1, gx = x0 + p % (SX + 2) - 1;
        float v = 0.0f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W && c0 + c < C) {
            const long tok = (long)b * L + (long)gy * W + gx;
            const float raw = load_val(x, tok * C2 + C + c0 + c);
            v = (raw - stats[tok * 2]) * stats[tok * 2 + 1] * gamma[c0 + c] + beta[c0 + c];
        }
        tile[i] = v;
    }
    __syncthreads();

    const int c = threadIdx.x % CC, tx = threadIdx.x / CC;
    if (c0 + c >= C) return;
    float k[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) k[i] = w[(c0 + c) * 9 + i];
    const float bc = bias ? bias[c0 + c] : 0.0f;
    const int ox = x0 + tx;
    if (ox >= W) return;
    #pragma unroll
    for (int ty = 0; ty < SY; ty++) {
        const int oy = y0 + ty;
        if (oy >= H) break;
        float acc = bc;
        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int j = 0; j < 3; j++)
                acc += k[i * 3 + j] * tile[((ty + i) * (SX + 2) + tx + j) * CC + c];
        const long tok = (long)b * L + (long)oy * W + ox;
        store_val(out, tok * C + c0 + c, acc * load_val(x, tok * C2 + c0 + c));
    }
}

extern "C" __global__ void sg_stats_f32(const float* x, int C, long tokens, float eps, float* stats) { sg_stats<float>(x, C, tokens, eps, stats); }
extern "C" __global__ void sg_stats_f16(const _Float16* x, int C, long tokens, float eps, float* stats) { sg_stats<_Float16>(x, C, tokens, eps, stats); }
extern "C" __global__ void sg_stats_bf16(const bf16_t* x, int C, long tokens, float eps, float* stats) { sg_stats<bf16_t>(x, C, tokens, eps, stats); }
#define SG_ARGS const float* stats, const float* gamma, const float* beta, const float* w, const float* bias
#define SG_PASS stats, gamma, beta, w, bias
extern "C" __global__ void sg_gate_f32(const float* x, float* out, SG_ARGS, int C, int H, int W) { sg_gate<float>(x, SG_PASS, out, C, H, W); }
extern "C" __global__ void sg_gate_f16(const _Float16* x, _Float16* out, SG_ARGS, int C, int H, int W) { sg_gate<_Float16>(x, SG_PASS, out, C, H, W); }
extern "C" __global__ void sg_gate_bf16(const bf16_t* x, bf16_t* out, SG_ARGS, int C, int H, int W) { sg_gate<bf16_t>(x, SG_PASS, out, C, H, W); }

// Depthwise 3x3 conv (zero padded) in the token layout, with an optional exact
// GELU on the output (gelu & 1) and/or on the input (gelu & 2, rounded to T
// like a separate PyTorch GELU before the conv). Reads channel c of token t at x[t * row_stride + ch_offset + c], so it
// can take e.g. the v part of a (B, L, 3C) qkv buffer directly, and writes a
// contiguous (B, L, C). Same tiling as sg_gate.
template <typename T>
__device__ void dw_tokens(const T* __restrict__ x, long row_stride, int ch_offset,
                          const float* __restrict__ w, const float* __restrict__ bias,
                          int gelu, T* __restrict__ out, int C, int H, int W) {
    __shared__ float tile[(SY + 2) * (SX + 2) * CC];
    const int chunks = (C + CC - 1) / CC;
    const int b = blockIdx.z / chunks;
    const int c0 = (blockIdx.z % chunks) * CC;
    const int x0 = blockIdx.x * SX, y0 = blockIdx.y * SY;
    const long L = (long)H * W;

    for (int i = threadIdx.x; i < (SY + 2) * (SX + 2) * CC; i += 256) {
        const int c = i % CC, p = i / CC;
        const int gy = y0 + p / (SX + 2) - 1, gx = x0 + p % (SX + 2) - 1;
        float v = 0.0f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W && c0 + c < C) {
            const long tok = (long)b * L + (long)gy * W + gx;
            v = load_val(x, tok * row_stride + ch_offset + c0 + c);
            if (gelu & 2) {
                T r;
                store_val(&r, 0, 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
                v = load_val(&r, 0);
            }
        }
        tile[i] = v;
    }
    __syncthreads();

    const int c = threadIdx.x % CC, tx = threadIdx.x / CC;
    if (c0 + c >= C) return;
    float k[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) k[i] = w[(c0 + c) * 9 + i];
    const float bc = bias ? bias[c0 + c] : 0.0f;
    const int ox = x0 + tx;
    if (ox >= W) return;
    #pragma unroll
    for (int ty = 0; ty < SY; ty++) {
        const int oy = y0 + ty;
        if (oy >= H) break;
        float acc = bc;
        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int j = 0; j < 3; j++)
                acc += k[i * 3 + j] * tile[((ty + i) * (SX + 2) + tx + j) * CC + c];
        if (gelu & 1) acc = 0.5f * acc * (1.0f + erff(acc * 0.70710678118654752f));
        const long tok = (long)b * L + (long)oy * W + ox;
        store_val(out, tok * C + c0 + c, acc);
    }
}

#define DWT_ARGS long row_stride, int ch_offset, const float* w, const float* bias, int gelu
#define DWT_PASS row_stride, ch_offset, w, bias, gelu
extern "C" __global__ void dw_tokens_f32(const float* x, float* out, DWT_ARGS, int C, int H, int W) { dw_tokens<float>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void dw_tokens_f16(const _Float16* x, _Float16* out, DWT_ARGS, int C, int H, int W) { dw_tokens<_Float16>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void dw_tokens_bf16(const bf16_t* x, bf16_t* out, DWT_ARGS, int C, int H, int W) { dw_tokens<bf16_t>(x, DWT_PASS, out, C, H, W); }

// LayerNorm over the channels of contiguous (tokens, C) rows, optionally after
// a residual add: sum = x + a (stored, rounded to T like PyTorch's add), out =
// LayerNorm(sum). One wave32 per token, 8 tokens per 256-thread block, up to
// 8 * 32 = 256 channels kept in registers.
__device__ __forceinline__ float round_t(float v, const float*) { return v; }
__device__ __forceinline__ float round_t(float v, const _Float16*) { return (float)(_Float16)v; }
__device__ __forceinline__ float round_t(float v, const bf16_t*) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}

template <typename T>
__device__ void add_ln(const T* __restrict__ x, const T* __restrict__ a, T* __restrict__ sum_out,
                       T* __restrict__ out, const float* __restrict__ gamma,
                       const float* __restrict__ beta, int C, long tokens, float eps,
                       int out_ld) {
    const int lane = threadIdx.x & 31;
    const long tok = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long row = tok * C;
    float v[8];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        float t = 0.0f;
        if (c < C) {
            t = load_val(x, row + c);
            if (a) t = round_t(t + load_val(a, row + c), x);
        }
        v[k] = t;
        s += t;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    const float mean = s / C;
    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        if (c < C) { const float dv = v[k] - mean; q += dv * dv; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor(q, o, 32);
    const float rstd = rsqrtf(q / C + eps);
    const long orow = tok * out_ld;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        if (c < C) {
            if (sum_out) store_val(sum_out, row + c, v[k]);
            store_val(out, orow + c, (v[k] - mean) * rstd * gamma[c] + beta[c]);
        }
    }
    for (int c = C + lane; c < out_ld; c += 32) store_val(out, orow + c, 0.0f);
}

#define LN_ARGS const float* gamma, const float* beta, int C, long tokens, float eps, int out_ld
#define LN_PASS gamma, beta, C, tokens, eps, out_ld
extern "C" __global__ void add_ln_f32(const float* x, const float* a, float* sum_out, float* out, LN_ARGS) { add_ln<float>(x, a, sum_out, out, LN_PASS); }
extern "C" __global__ void add_ln_f16(const _Float16* x, const _Float16* a, _Float16* sum_out, _Float16* out, LN_ARGS) { add_ln<_Float16>(x, a, sum_out, out, LN_PASS); }
extern "C" __global__ void add_ln_bf16(const bf16_t* x, const bf16_t* a, bf16_t* sum_out, bf16_t* out, LN_ARGS) { add_ln<bf16_t>(x, a, sum_out, out, LN_PASS); }

// HAT's HAB residual and second LayerNorm in one pass:
//   sum = (x + a) + (conv * ca[b, c]) * scale     (each op rounded to T, as in
//         `shortcut + attn_x + conv_x * conv_scale` with conv_x = conv * ca)
//   out = LayerNorm(sum)
// x and a are (B, L, C) token rows; conv is the CAB conv output in the image
// layout (B, C, L), or in token rows when conv_nhwc (a channels_last conv), so
// a block first stages 32 tokens of all channels in shared memory (coalesced
// either way) and then works on token rows like add_ln.
// One token row of hab_ln: v[k] for channels lane + 32k, given a loader for
// the conv value (already including its bias) of channel c.
template <typename T, typename ConvAt>
__device__ __forceinline__ void hab_ln_row(const T* __restrict__ x, const T* __restrict__ a,
                                           const float* __restrict__ ca, float scale,
                                           T* __restrict__ sum_out, T* __restrict__ out,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta, int C, long b,
                                           long row, long orow, int out_ld, float eps,
                                           int lane, ConvAt conv_at) {
    float v[8];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        float t = 0.0f;
        if (c < C) {
            const float cv = round_t(round_t(conv_at(c) * ca[b * C + c], x) * scale, x);
            t = round_t(round_t(load_val(x, row + c) + load_val(a, row + c), x) + cv, x);
        }
        v[k] = t;
        s += t;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    const float mean = s / C;
    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        if (c < C) { const float dv = v[k] - mean; q += dv * dv; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor(q, o, 32);
    const float rstd = rsqrtf(q / C + eps);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        if (c < C) {
            store_val(sum_out, row + c, v[k]);
            store_val(out, orow + c, (v[k] - mean) * rstd * gamma[c] + beta[c]);
        }
    }
    for (int c = C + lane; c < out_ld; c += 32) store_val(out, orow + c, 0.0f);
}

// hab_ln for a channels_last conv output (token rows): no staging needed, one
// wave per token like add_ln. cbias (C,) is the conv bias, or null.
template <typename T>
__device__ void hab_ln_rows(const T* __restrict__ x, const T* __restrict__ a,
                            const T* __restrict__ conv, const float* __restrict__ cbias,
                            const float* __restrict__ ca, float scale,
                            T* __restrict__ sum_out, T* __restrict__ out,
                            const float* __restrict__ gamma, const float* __restrict__ beta,
                            int C, long L, long tokens, float eps, int out_ld, int conv_ld) {
    const int lane = threadIdx.x & 31;
    const long tok = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long row = tok * C, crow = tok * conv_ld;
    hab_ln_row(x, a, ca, scale, sum_out, out, gamma, beta, C, tok / L, row, tok * out_ld,
               out_ld, eps, lane,
               [&](int c) {
                   const float cv = load_val(conv, crow + c);
                   return cbias ? round_t(cv + cbias[c], x) : cv;
               });
}

template <typename T>
__device__ void hab_ln(const T* __restrict__ x, const T* __restrict__ a,
                       const T* __restrict__ conv, const float* __restrict__ ca,
                       float scale, T* __restrict__ sum_out, T* __restrict__ out,
                       const float* __restrict__ gamma, const float* __restrict__ beta,
                       int C, long L, long tokens, float eps, int conv_nhwc) {
    __shared__ float tile[256 * 33];
    const long t0 = (long)blockIdx.x * 32;
    for (int i = threadIdx.x; i < C * 32; i += 256) {
        const int c = conv_nhwc ? i % C : i / 32, j = conv_nhwc ? i / C : i % 32;
        const long tok = t0 + j;
        float v = 0.0f;
        if (tok < tokens) {
            if (conv_nhwc) {
                v = load_val(conv, tok * C + c);
            } else {
                const long b = tok / L, p = tok - b * L;
                v = load_val(conv, (b * C + c) * L + p);
            }
        }
        tile[c * 33 + j] = v;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int j = wave; j < 32; j += 8) {
        const long tok = t0 + j;
        if (tok >= tokens) break;
        const long b = tok / L;
        const long row = tok * C;
        float v[8];
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = lane + 32 * k;
            float t = 0.0f;
            if (c < C) {
                const float cv = round_t(round_t(tile[c * 33 + j] * ca[b * C + c], x) * scale, x);
                t = round_t(round_t(load_val(x, row + c) + load_val(a, row + c), x) + cv, x);
            }
            v[k] = t;
            s += t;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
        const float mean = s / C;
        float q = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = lane + 32 * k;
            if (c < C) { const float dv = v[k] - mean; q += dv * dv; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) q += __shfl_xor(q, o, 32);
        const float rstd = rsqrtf(q / C + eps);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = lane + 32 * k;
            if (c < C) {
                store_val(sum_out, row + c, v[k]);
                store_val(out, row + c, (v[k] - mean) * rstd * gamma[c] + beta[c]);
            }
        }
    }
}

#define HAB_SIG(T, S) extern "C" __global__ void hab_ln_##S(const T* x, const T* a, const T* conv, const float* ca, float scale, T* sum_out, T* out, const float* gamma, const float* beta, int C, long L, long tokens, float eps, int conv_nhwc) { hab_ln<T>(x, a, conv, ca, scale, sum_out, out, gamma, beta, C, L, tokens, eps, conv_nhwc); }
HAB_SIG(float, f32)
HAB_SIG(_Float16, f16)
HAB_SIG(bf16_t, bf16)
#define HAB_ROWS_SIG(T, S) extern "C" __global__ void hab_ln_rows_##S(const T* x, const T* a, const T* conv, const float* cbias, const float* ca, float scale, T* sum_out, T* out, const float* gamma, const float* beta, int C, long L, long tokens, float eps, int out_ld, int conv_ld) { hab_ln_rows<T>(x, a, conv, cbias, ca, scale, sum_out, out, gamma, beta, C, L, tokens, eps, out_ld, conv_ld); }
HAB_ROWS_SIG(float, f32)
HAB_ROWS_SIG(_Float16, f16)
HAB_ROWS_SIG(bf16_t, bf16)

// Per-channel bias for channels_last conv outputs (rows of C), which MIOpen's
// NHWC path leaves to a separate broadcast add:
//   bias_act: y = act(round(y + bias[c]))  in place, act = exact GELU or none
//   bias_res: out = round(round(y + bias[c]) + r)
template <typename T>
__device__ void bias_act(T* __restrict__ y, const float* __restrict__ bias, int C, long total,
                         int gelu) {
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
        float v = round_t(load_val(y, i) + bias[i % C], y);
        if (gelu) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        store_val(y, i, v);
    }
}

template <typename T>
__device__ void bias_res(const T* __restrict__ y, const float* __restrict__ bias,
                         const T* __restrict__ r, T* __restrict__ out, int C, long total,
                         int y_ld) {
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
        const long t = i / C;
        const int c = (int)(i - t * C);
        const float v = round_t(load_val(y, t * y_ld + c) + bias[c], y);
        store_val(out, i, v + load_val(r, i));
    }
}

// out (rows of ld) = round(x + a) for c < C, 0 for the padding channels
template <typename T>
__device__ void pad_add(const T* __restrict__ x, const T* __restrict__ a, T* __restrict__ out,
                        int C, int ld, long total) {
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
        const long t = i / ld;
        const int c = (int)(i - t * ld);
        float v = 0.0f;
        if (c < C) v = load_val(x, t * C + c) + (a ? load_val(a, t * C + c) : 0.0f);
        store_val(out, i, v);
    }
}
extern "C" __global__ void pad_add_f32(const float* x, const float* a, float* out, int C, int ld, long total) { pad_add<float>(x, a, out, C, ld, total); }
extern "C" __global__ void pad_add_f16(const _Float16* x, const _Float16* a, _Float16* out, int C, int ld, long total) { pad_add<_Float16>(x, a, out, C, ld, total); }
extern "C" __global__ void pad_add_bf16(const bf16_t* x, const bf16_t* a, bf16_t* out, int C, int ld, long total) { pad_add<bf16_t>(x, a, out, C, ld, total); }

extern "C" __global__ void bias_act_f32(float* y, const float* bias, int C, long total, int gelu) { bias_act<float>(y, bias, C, total, gelu); }
extern "C" __global__ void bias_act_f16(_Float16* y, const float* bias, int C, long total, int gelu) { bias_act<_Float16>(y, bias, C, total, gelu); }
extern "C" __global__ void bias_act_bf16(bf16_t* y, const float* bias, int C, long total, int gelu) { bias_act<bf16_t>(y, bias, C, total, gelu); }
extern "C" __global__ void bias_res_f32(const float* y, const float* bias, const float* r, float* out, int C, long total, int y_ld) { bias_res<float>(y, bias, r, out, C, total, y_ld); }
extern "C" __global__ void bias_res_f16(const _Float16* y, const float* bias, const _Float16* r, _Float16* out, int C, long total, int y_ld) { bias_res<_Float16>(y, bias, r, out, C, total, y_ld); }
extern "C" __global__ void bias_res_bf16(const bf16_t* y, const float* bias, const bf16_t* r, bf16_t* out, int C, long total, int y_ld) { bias_res<bf16_t>(y, bias, r, out, C, total, y_ld); }

// DAT's Adaptive Interaction gating, in one pass over (B, N, C):
//   out = a * sigmoid(tok[b, n]) + b * sigmoid(ch[b, c])   (a_uses_tok = 1)
//   out = a * sigmoid(ch[b, c]) + b * sigmoid(tok[b, n])   (a_uses_tok = 0)
// PyTorch runs this as four broadcast kernels with temporaries.
template <typename T>
__device__ void gate_mix(const T* __restrict__ a, const T* __restrict__ bt,
                         const T* __restrict__ tok, const T* __restrict__ ch,
                         T* __restrict__ out, int C, long N, long total, int a_uses_tok) {
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
        const long t = i / C;
        const int c = (int)(i - t * C);
        const long bi = t / N;
        const float ts = 1.0f / (1.0f + __expf(-load_val(tok, t)));
        const float cs = 1.0f / (1.0f + __expf(-load_val(ch, bi * C + c)));
        const float va = load_val(a, i), vb = load_val(bt, i);
        store_val(out, i, a_uses_tok ? va * ts + vb * cs : va * cs + vb * ts);
    }
}

extern "C" __global__ void gate_mix_f32(const float* a, const float* b, const float* tok, const float* ch, float* out, int C, long N, long total, int a_uses_tok) { gate_mix<float>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
extern "C" __global__ void gate_mix_f16(const _Float16* a, const _Float16* b, const _Float16* tok, const _Float16* ch, _Float16* out, int C, long N, long total, int a_uses_tok) { gate_mix<_Float16>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
extern "C" __global__ void gate_mix_bf16(const bf16_t* a, const bf16_t* b, const bf16_t* tok, const bf16_t* ch, bf16_t* out, int C, long N, long total, int a_uses_tok) { gate_mix<bf16_t>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
"""

_ATTN_WMMA_SRC = r"""
// DAT window attention on RDNA4 matrix cores (gfx12 WMMA, wave32).
// Same contract as win_attn (see there), for 16-bit inputs.
//
// Block: 256 threads = 8 waves, one (window, head) per block. K and V of the
// window are staged in shared memory (V transposed), each wave walks query
// tiles of 16 rows. Per 16-key tile: S = Q K^T via WMMA (fp32 accumulate),
// scale + bias + shift mask + online softmax in registers, P goes through a
// small per-wave LDS buffer into WMMA operand layout, O += P V via WMMA.
//
// gfx12 wave32 layouts (verified): A lane l -> row l%16, k = 8*(l/16)+i;
// B lane l -> col l%16, k = 8*(l/16)+i; D lane l -> col l%16, row 8*(l/16)+i.

typedef unsigned short u16;
typedef short v8s __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u16 f2bf(float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ u16 f2h(float v) {
    _Float16 h = (_Float16)v;
    return *(u16*)&h;
}
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((unsigned int)v) << 16); }
__device__ __forceinline__ float h2f(u16 v) { return (float)*(_Float16*)&v; }

template <bool BF16>
__device__ __forceinline__ v8f mma(v8s a, v8s b, v8f c) {
    if constexpr (BF16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            __builtin_bit_cast(v8h, a), __builtin_bit_cast(v8h, b), c);
    }
}

template <bool BF16, int KD>
__device__ void win_attn_wmma(const u16* __restrict__ qkv, long s0, long s1, long s2, long s3,
                              const float* __restrict__ biasT, const int* __restrict__ labels,
                              u16* __restrict__ out, long out_stride, int out_off, int Hc, int Wc,
                              int sy, int sx, int H, int W, int H_sp, int W_sp,
                              int heads, int d, float scale) {
    constexpr int KS = KD / 16;  // k steps over the head dim, and O column tiles
    extern __shared__ unsigned char lds_raw[];
    const int N = H_sp * W_sp;
    u16* Ks = (u16*)lds_raw;           // [N][KD]
    u16* Vt = Ks + N * KD;             // [KD][N]
    u16* Pb = Vt + N * KD;             // [8 waves][16][16]
    int* labs = (int*)(Pb + 8 * 256);  // [N]

    const int nw = W / W_sp, nh = H / H_sp, nW = nh * nw;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int ih = wimg / nw, iw = wimg % nw;

    // stage K and V^T (zero padded to KD) and the region labels
    for (int t = threadIdx.x; t < N; t += 256) {
        const int y = ih * H_sp + t / W_sp, x = iw * W_sp + t % W_sp;
        const int ys = (y + sy) % H, xs = (x + sx) % W;
        const bool valid = ys < Hc && xs < Wc;  // window padding is zero
        const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + (long)(h * d) * s3;
        for (int j = 0; j < KD; j++) {
            const bool in = j < d && valid;
            Ks[t * KD + j] = in ? qkv[s0 + base + j * s3] : (u16)0;
            Vt[j * N + t] = in ? qkv[2 * s0 + base + j * s3] : (u16)0;
        }
        if (labels) labs[t] = labels[wimg * N + t];
    }
    __syncthreads();

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    u16* P = Pb + wave * 256;

    for (int qt = wave; qt < N / 16; qt += 8) {
        // A operands for Q: row = query qt*16 + col, k = s*16 + 8*half + i
        v8s qa[KS];
        {
            const int n = qt * 16 + col;
            const int y = ih * H_sp + n / W_sp, x = iw * W_sp + n % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            const bool valid = ys < Hc && xs < Wc;
            const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + (long)(h * d) * s3;
            #pragma unroll
            for (int s = 0; s < KS; s++)
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int j = s * 16 + 8 * half + i;
                    qa[s][i] = (j < d && valid) ? (short)qkv[base + j * s3] : (short)0;
                }
        }
        // this lane's 8 query rows (D layout)
        const int q0 = qt * 16 + 8 * half;
        int qlab[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) qlab[i] = labels ? labs[q0 + i] : 0;

        float m[8], l[8];
        v8f o[KS];
        #pragma unroll
        for (int i = 0; i < 8; i++) { m[i] = -__builtin_inff(); l[i] = 0.0f; }
        #pragma unroll
        for (int t = 0; t < KS; t++) o[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

        for (int kt = 0; kt < N / 16; kt++) {
            const int key = kt * 16 + col;
            v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int s = 0; s < KS; s++) {
                const v8s kb = *(const v8s*)(Ks + key * KD + s * 16 + 8 * half);
                sacc = mma<BF16>(qa[s], kb, sacc);
            }
            float sv[8];
            const int klab = labels ? labs[key] : 0;
            const float* brow = biasT ? biasT + ((long)h * N + key) * N + q0 : nullptr;
            v4f b0 = {0, 0, 0, 0}, b1 = {0, 0, 0, 0};
            if (brow) { b0 = *(const v4f*)brow; b1 = *(const v4f*)(brow + 4); }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float v = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                if (labels && klab != qlab[i]) v -= 100.0f;
                sv[i] = v;
            }
            // row max / sum across the 16 key columns of this half wave
            float corr[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float mx = sv[i];
                mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
                const float mn = fmaxf(m[i], mx);
                corr[i] = __expf(m[i] - mn);
                const float p = __expf(sv[i] - mn);
                float ps = p;
                ps += __shfl_xor(ps, 1, 16);
                ps += __shfl_xor(ps, 2, 16);
                ps += __shfl_xor(ps, 4, 16);
                ps += __shfl_xor(ps, 8, 16);
                l[i] = l[i] * corr[i] + ps;
                m[i] = mn;
                // P[row][key] for the A-layout re-read below
                P[(8 * half + i) * 16 + col] = BF16 ? f2bf(p) : f2h(p);
            }
            #pragma unroll
            for (int t = 0; t < KS; t++)
                #pragma unroll
                for (int i = 0; i < 8; i++) o[t][i] *= corr[i];
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            // A operand: row = query col, k = key 8*half + i
            const v8s pa = *(const v8s*)(P + col * 16 + 8 * half);
            #pragma unroll
            for (int t = 0; t < KS; t++) {
                // B operand: col = head dim t*16 + col, k = key kt*16 + 8*half + i
                const v8s vb = *(const v8s*)(Vt + (t * 16 + col) * N + kt * 16 + 8 * half);
                o[t] = mma<BF16>(pa, vb, o[t]);
            }
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        }

        // write O / l: lane holds head dim t*16 + col for rows q0 + i
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = q0 + i;
            const int y = ih * H_sp + n / W_sp, x = iw * W_sp + n % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            if (ys >= Hc || xs >= Wc) continue;
            const float inv = 1.0f / l[i];
            const long ob = (((long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
            #pragma unroll
            for (int t = 0; t < KS; t++) {
                const int j = t * 16 + col;
                if (j < d) out[ob + j] = BF16 ? f2bf(o[t][i] * inv) : f2h(o[t][i] * inv);
            }
        }
    }
}

#ifndef WMMAT_QT
#define WMMAT_QT 2
#endif

// The same attention computed transposed: S^T = K Q^T and O^T = V^T P^T.
// The D layout of S^T (lane -> query l%16, keys 8*(l/16)+i) is exactly the B
// operand layout P^T needs, so P never leaves the registers, and each lane
// owns one query: the softmax row max is a local max over 8 values plus one
// shuffle with the other half wave, the row sum is local until the end, and
// the O^T rescale is one scalar per lane. Two key tiles per softmax step.
template <bool BF16, int KD, int QT>
__device__ void win_attn_wmmat(const u16* __restrict__ qkv, long s0, long s1, long s2, long s3,
                               const float* __restrict__ biasT, const int* __restrict__ labels,
                               u16* __restrict__ out, long out_stride, int out_off, int Hc, int Wc,
                               int sy, int sx, int H, int W, int H_sp, int W_sp,
                               int heads, int d, float scale) {
    constexpr int KS = KD / 16;
    extern __shared__ unsigned char lds_raw[];
    const int N = H_sp * W_sp;
    u16* Ks = (u16*)lds_raw;           // [N][KD]
    u16* Vt = Ks + N * KD;             // [KD][N]
    int* labs = (int*)(Vt + N * KD);   // [N]

    const int nw = W / W_sp, nh = H / H_sp, nW = nh * nw;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int ih = wimg / nw, iw = wimg % nw;

    // Stage K and V^T. With unit channel stride and even sizes, consecutive
    // threads read consecutive 32-bit pairs of one token's channels, so the
    // loads are coalesced; otherwise one thread per token as in win_attn_wmma.
    const bool pairs = s3 == 1 && !(d & 1) && !(s0 & 1) && !(s1 & 1) && !(s2 & 1);
    if (pairs) {
        constexpr int KP = KD / 2;
        for (int i = threadIdx.x; i < N * KP; i += 256) {
            const int t = i / KP, j = 2 * (i % KP);
            const int y = ih * H_sp + t / W_sp, x = iw * W_sp + t % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            unsigned int kv = 0, vv = 0;
            if (j < d && ys < Hc && xs < Wc) {
                const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + h * d + j;
                kv = *(const unsigned int*)(qkv + s0 + base);
                vv = *(const unsigned int*)(qkv + 2 * s0 + base);
            }
            *(unsigned int*)(Ks + t * KD + j) = kv;
            Vt[j * N + t] = (u16)(vv & 0xFFFFu);
            Vt[(j + 1) * N + t] = (u16)(vv >> 16);
        }
    } else {
        for (int t = threadIdx.x; t < N; t += 256) {
            const int y = ih * H_sp + t / W_sp, x = iw * W_sp + t % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            const bool valid = ys < Hc && xs < Wc;
            const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + (long)(h * d) * s3;
            for (int j = 0; j < KD; j++) {
                const bool in = j < d && valid;
                Ks[t * KD + j] = in ? qkv[s0 + base + j * s3] : (u16)0;
                Vt[j * N + t] = in ? qkv[2 * s0 + base + j * s3] : (u16)0;
            }
        }
    }
    if (labels)
        for (int t = threadIdx.x; t < N; t += 256) labs[t] = labels[wimg * N + t];
    __syncthreads();

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    const int nkt = N / 16;

    // QT query tiles per wave at once: they share every K/V^T operand load
    // and give the scheduler independent WMMA/exp chains to interleave.
    for (int qt0 = wave * QT; qt0 < nkt; qt0 += 8 * QT) {
        const int nq = min(QT, nkt - qt0);
        int qn[QT], qlab[QT];
        const float* bq[QT];
        v8s qb[QT][KS];
        float m[QT], lsum[QT];
        v8f o[QT][KS];
        #pragma unroll
        for (int r = 0; r < QT; r++) {
            qn[r] = (qt0 + (r < nq ? r : 0)) * 16 + col;
            const int y = ih * H_sp + qn[r] / W_sp, x = iw * W_sp + qn[r] % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            const bool valid = ys < Hc && xs < Wc;
            if (pairs) {
                const unsigned int* qp = (const unsigned int*)(
                    qkv + (long)b * s1 + ((long)ys * Wc + xs) * s2 + h * d);
                #pragma unroll
                for (int s = 0; s < KS; s++)
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int j = s * 16 + 8 * half + 2 * i;
                        const unsigned int u = (j < d && valid) ? qp[j / 2] : 0u;
                        qb[r][s][2 * i] = (short)(u & 0xFFFFu);
                        qb[r][s][2 * i + 1] = (short)(u >> 16);
                    }
            } else {
                const long base = (long)b * s1 + ((long)ys * Wc + xs) * s2 + (long)(h * d) * s3;
                #pragma unroll
                for (int s = 0; s < KS; s++)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int j = s * 16 + 8 * half + i;
                        qb[r][s][i] = (j < d && valid) ? (short)qkv[base + j * s3] : (short)0;
                    }
            }
            qlab[r] = labels ? labs[qn[r]] : 0;
            // biasQ is (heads, query, key): this lane's 8 keys are contiguous
            bq[r] = biasT ? biasT + ((long)h * N + qn[r]) * N : nullptr;
            m[r] = -__builtin_inff(); lsum[r] = 0.0f;
            #pragma unroll
            for (int t = 0; t < KS; t++) o[r][t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
        }

        for (int kt = 0; kt < nkt; kt++) {
            v8s ka[KS], va[KS];
            #pragma unroll
            for (int s = 0; s < KS; s++)
                ka[s] = *(const v8s*)(Ks + (kt * 16 + col) * KD + s * 16 + 8 * half);
            #pragma unroll
            for (int t = 0; t < KS; t++)
                va[t] = *(const v8s*)(Vt + (t * 16 + col) * N + kt * 16 + 8 * half);
            const int k0 = kt * 16 + 8 * half;
            v4i l0 = {0, 0, 0, 0}, l1 = {0, 0, 0, 0};
            if (labels) { l0 = *(const v4i*)(labs + k0); l1 = *(const v4i*)(labs + k0 + 4); }
            #pragma unroll
            for (int r = 0; r < QT; r++) {
                v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int s = 0; s < KS; s++) sacc = mma<BF16>(ka[s], qb[r][s], sacc);
                v4f b0 = {0, 0, 0, 0}, b1 = {0, 0, 0, 0};
                if (bq[r]) { b0 = *(const v4f*)(bq[r] + k0); b1 = *(const v4f*)(bq[r] + k0 + 4); }
                float sv[8];
                float mx = -__builtin_inff();
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                    if (labels && (i < 4 ? l0[i] : l1[i - 4]) != qlab[r]) v -= 100.0f;
                    sv[i] = v;
                    mx = fmaxf(mx, v);
                }
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float mn = fmaxf(m[r], mx);
                const float corr = __expf(m[r] - mn);
                m[r] = mn;
                v8s pb;
                float ps = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float p = __expf(sv[i] - mn);
                    ps += p;
                    pb[i] = (short)(BF16 ? f2bf(p) : f2h(p));
                }
                lsum[r] = lsum[r] * corr + ps;
                #pragma unroll
                for (int t = 0; t < KS; t++) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) o[r][t][i] *= corr;
                    o[r][t] = mma<BF16>(va[t], pb, o[r][t]);
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < QT; r++) {
            if (r >= nq) break;
            const float ls = lsum[r] + __shfl_xor(lsum[r], 16, 32);
            const int y = ih * H_sp + qn[r] / W_sp, x = iw * W_sp + qn[r] % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            if (ys < Hc && xs < Wc) {
                const float inv = 1.0f / ls;
                const long ob = (((long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
                #pragma unroll
                for (int t = 0; t < KS; t++)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int j = t * 16 + 8 * half + i;
                        if (j < d) out[ob + j] = BF16 ? f2bf(o[r][t][i] * inv) : f2h(o[r][t][i] * inv);
                    }
            }
        }
    }
}

#define WMMA_ARGS long s0, long s1, long s2, long s3, const float* biasT, const int* labels, long out_stride, int out_off, int Hc, int Wc, int sy, int sx, int H, int W, int H_sp, int W_sp, int heads, int d, float scale
#define WMMA_PASS s0, s1, s2, s3, biasT, labels, out, out_stride, out_off, Hc, Wc, sy, sx, H, W, H_sp, W_sp, heads, d, scale
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_bf16_32(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<true, 32>(qkv, WMMA_PASS); }
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_bf16_64(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<true, 64>(qkv, WMMA_PASS); }
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_bf16_16(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<true, 16>(qkv, WMMA_PASS); }
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_f16_32(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<false, 32>(qkv, WMMA_PASS); }
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_f16_64(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<false, 64>(qkv, WMMA_PASS); }
extern "C" __global__ void __launch_bounds__(256) win_attn_wmma_f16_16(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmma<false, 16>(qkv, WMMA_PASS); }
#define WMMAT(S, B, KD) extern "C" __global__ void __launch_bounds__(256) win_attn_wmmat_##S##_##KD(const u16* qkv, u16* out, WMMA_ARGS) { win_attn_wmmat<B, KD, WMMAT_QT>(qkv, WMMA_PASS); }
WMMAT(bf16, true, 16) WMMAT(bf16, true, 32) WMMAT(bf16, true, 64)
WMMAT(f16, false, 16) WMMAT(f16, false, 32) WMMAT(f16, false, 64)
"""

# HAT OCAB: the helpers of _ATTN_WMMA_SRC (up to win_attn_wmma) + its own kernel
_OCAB_SRC = (
    _ATTN_WMMA_SRC[
        : _ATTN_WMMA_SRC.index(
            "template <bool BF16, int KD>\n__device__ void win_attn_wmma"
        )
    ]
    + r"""
// HAT's overlapping cross attention (OCAB) on RDNA4 matrix cores.
//
// Queries come from the non-overlapping ws x ws window at (iy*ws, ix*ws), keys
// and values from the larger ows x ows window starting `pad` pixels above and
// left of it, with zeros outside the image (exactly HAT's zero-padded unfold
// of the projected k/v). The k/v windows overlap their neighbours, so they are
// read in place from qkv instead of being cut out.
//
// Block: 256 threads = 8 waves, one (window, head) per block; each wave owns up
// to MAXT query tiles of 16 rows. The keys do not fit in shared memory at once
// (576 keys for ws=16), so they are staged `chunk` at a time; the online
// softmax state of every query row stays in registers across chunks. Key slots
// past ows*ows (padding to a multiple of 16) score -inf.

template <bool BF16, int KD>
__device__ void ocab_wmma(const u16* __restrict__ qkv, long s0, long s1, long s2, long s3,
                          const float* __restrict__ biasT, u16* __restrict__ out,
                          int H, int W, int ws, int ows, int heads, int d, float scale,
                          int chunk) {
    constexpr int KS = KD / 16;
    constexpr int MAXT = 2;  // query tiles per wave: ws*ws <= 8 * 16 * MAXT
    extern __shared__ unsigned char lds_raw[];
    const int NQ = ws * ws, NK = ows * ows, NKp = (NK + 15) / 16 * 16;
    const int pad = (ows - ws) / 2;
    u16* Ks = (u16*)lds_raw;      // [chunk][KD]
    u16* Vt = Ks + chunk * KD;    // [KD][chunk]
    u16* Pb = Vt + chunk * KD;    // [8 waves][16][16]

    const int nwx = W / ws, nW = (H / ws) * nwx;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int iy = wimg / nwx, ix = wimg % nwx;
    const int ky0 = iy * ws - pad, kx0 = ix * ws - pad;

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    u16* P = Pb + wave * 256;
    const int ntiles = NQ / 16;

    v8s qa[MAXT][KS];
    float m[MAXT][8], l[MAXT][8];
    v8f o[MAXT][KS];
    #pragma unroll
    for (int tt = 0; tt < MAXT; tt++) {
        const int qt = wave + 8 * tt;
        #pragma unroll
        for (int i = 0; i < 8; i++) { m[tt][i] = -__builtin_inff(); l[tt][i] = 0.0f; }
        #pragma unroll
        for (int t = 0; t < KS; t++) o[tt][t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < KS; s++) qa[tt][s] = (v8s){0, 0, 0, 0, 0, 0, 0, 0};
        if (qt < ntiles) {
            const int n = qt * 16 + col;
            const int y = iy * ws + n / ws, x = ix * ws + n % ws;
            const long base = (long)b * s1 + ((long)y * W + x) * s2 + (long)(h * d) * s3;
            #pragma unroll
            for (int s = 0; s < KS; s++)
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int j = s * 16 + 8 * half + i;
                    qa[tt][s][i] = j < d ? (short)qkv[base + j * s3] : (short)0;
                }
        }
    }

    for (int c0 = 0; c0 < NKp; c0 += chunk) {
        const int nc = min(chunk, NKp - c0);
        __syncthreads();  // everyone is done with the previous chunk
        for (int t = threadIdx.x; t < nc; t += 256) {
            const int key = c0 + t;
            const int y = ky0 + key / ows, x = kx0 + key % ows;
            const bool valid = key < NK && y >= 0 && y < H && x >= 0 && x < W;
            const long base = valid
                ? (long)b * s1 + ((long)y * W + x) * s2 + (long)(h * d) * s3 : 0;
            for (int j = 0; j < KD; j++) {
                const bool in = valid && j < d;
                Ks[t * KD + j] = in ? qkv[s0 + base + j * s3] : (u16)0;
                Vt[j * chunk + t] = in ? qkv[2 * s0 + base + j * s3] : (u16)0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tt = 0; tt < MAXT; tt++) {
            const int qt = wave + 8 * tt;
            if (qt >= ntiles) break;
            const int q0 = qt * 16 + 8 * half;
            for (int kt = 0; kt < nc / 16; kt++) {
                const int kl = kt * 16 + col, key = c0 + kl;
                v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int s = 0; s < KS; s++) {
                    const v8s kb = *(const v8s*)(Ks + kl * KD + s * 16 + 8 * half);
                    sacc = mma<BF16>(qa[tt][s], kb, sacc);
                }
                float sv[8];
                if (key < NK) {
                    const float* brow = biasT + ((long)h * NK + key) * NQ + q0;
                    const v4f b0 = *(const v4f*)brow, b1 = *(const v4f*)(brow + 4);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) sv[i] = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) sv[i] = -__builtin_inff();
                }
                float corr[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float mx = sv[i];
                    mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
                    const float mn = fmaxf(m[tt][i], mx);
                    corr[i] = __expf(m[tt][i] - mn);
                    const float p = __expf(sv[i] - mn);
                    float ps = p;
                    ps += __shfl_xor(ps, 1, 16);
                    ps += __shfl_xor(ps, 2, 16);
                    ps += __shfl_xor(ps, 4, 16);
                    ps += __shfl_xor(ps, 8, 16);
                    l[tt][i] = l[tt][i] * corr[i] + ps;
                    m[tt][i] = mn;
                    P[(8 * half + i) * 16 + col] = BF16 ? f2bf(p) : f2h(p);
                }
                #pragma unroll
                for (int t = 0; t < KS; t++)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) o[tt][t][i] *= corr[i];
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
                const v8s pa = *(const v8s*)(P + col * 16 + 8 * half);
                #pragma unroll
                for (int t = 0; t < KS; t++) {
                    const v8s vb = *(const v8s*)(Vt + (t * 16 + col) * chunk + kt * 16 + 8 * half);
                    o[tt][t] = mma<BF16>(pa, vb, o[tt][t]);
                }
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            }
        }
    }

    #pragma unroll
    for (int tt = 0; tt < MAXT; tt++) {
        const int qt = wave + 8 * tt;
        if (qt >= ntiles) break;
        const int q0 = qt * 16 + 8 * half;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = q0 + i;
            const int y = iy * ws + n / ws, x = ix * ws + n % ws;
            const float inv = 1.0f / l[tt][i];
            const long ob = (((long)b * H + y) * W + x) * (long)(heads * d) + h * d;
            #pragma unroll
            for (int t = 0; t < KS; t++) {
                const int j = t * 16 + col;
                if (j < d) out[ob + j] = BF16 ? f2bf(o[tt][t][i] * inv) : f2h(o[tt][t][i] * inv);
            }
        }
    }
}

// OCAB computed transposed, like win_attn_wmmat: S^T = K Q^T, O^T = V^T P^T,
// one query per lane, P kept in registers. biasQ is (heads, NQ, NKp) with the
// padding key slots set to -inf, so padding needs no test. K/V chunks are
// staged with coalesced 32-bit loads when the channel layout allows it.
template <bool BF16, int KD>
__device__ void ocab_wmmat(const u16* __restrict__ qkv, long s0, long s1, long s2, long s3,
                           const float* __restrict__ biasQ, u16* __restrict__ out,
                           int H, int W, int ws, int ows, int heads, int d, float scale,
                           int chunk) {
    constexpr int KS = KD / 16;
    constexpr int MAXT = 2;
    extern __shared__ unsigned char lds_raw[];
    const int NQ = ws * ws, NK = ows * ows, NKp = (NK + 15) / 16 * 16;
    const int pad = (ows - ws) / 2;
    u16* Ks = (u16*)lds_raw;      // [chunk][KD]
    u16* Vt = Ks + chunk * KD;    // [KD][chunk]

    const int nwx = W / ws, nW = (H / ws) * nwx;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int iy = wimg / nwx, ix = wimg % nwx;
    const int ky0 = iy * ws - pad, kx0 = ix * ws - pad;

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    const int ntiles = NQ / 16;
    const bool pairs = s3 == 1 && !(d & 1) && !(s0 & 1) && !(s1 & 1) && !(s2 & 1);

    v8s qb[MAXT][KS];
    float m[MAXT], lsum[MAXT];
    v8f o[MAXT][KS];
    #pragma unroll
    for (int tt = 0; tt < MAXT; tt++) {
        const int qt = wave + 8 * tt;
        m[tt] = -__builtin_inff(); lsum[tt] = 0.0f;
        #pragma unroll
        for (int t = 0; t < KS; t++) o[tt][t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < KS; s++) qb[tt][s] = (v8s){0, 0, 0, 0, 0, 0, 0, 0};
        if (qt < ntiles) {
            const int n = qt * 16 + col;
            const int y = iy * ws + n / ws, x = ix * ws + n % ws;
            const long base = (long)b * s1 + ((long)y * W + x) * s2 + (long)(h * d) * s3;
            #pragma unroll
            for (int s = 0; s < KS; s++)
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int j = s * 16 + 8 * half + i;
                    qb[tt][s][i] = j < d ? (short)qkv[base + j * s3] : (short)0;
                }
        }
    }

    for (int c0 = 0; c0 < NKp; c0 += chunk) {
        const int nc = min(chunk, NKp - c0);
        __syncthreads();
        if (pairs) {
            constexpr int KP = KD / 2;
            for (int i = threadIdx.x; i < nc * KP; i += 256) {
                const int t = i / KP, j = 2 * (i % KP);
                const int key = c0 + t;
                const int y = ky0 + key / ows, x = kx0 + key % ows;
                unsigned int kv = 0, vv = 0;
                if (j < d && key < NK && y >= 0 && y < H && x >= 0 && x < W) {
                    const long base = (long)b * s1 + ((long)y * W + x) * s2 + h * d + j;
                    kv = *(const unsigned int*)(qkv + s0 + base);
                    vv = *(const unsigned int*)(qkv + 2 * s0 + base);
                }
                *(unsigned int*)(Ks + t * KD + j) = kv;
                Vt[j * chunk + t] = (u16)(vv & 0xFFFFu);
                Vt[(j + 1) * chunk + t] = (u16)(vv >> 16);
            }
        } else {
            for (int t = threadIdx.x; t < nc; t += 256) {
                const int key = c0 + t;
                const int y = ky0 + key / ows, x = kx0 + key % ows;
                const bool valid = key < NK && y >= 0 && y < H && x >= 0 && x < W;
                const long base = valid
                    ? (long)b * s1 + ((long)y * W + x) * s2 + (long)(h * d) * s3 : 0;
                for (int j = 0; j < KD; j++) {
                    const bool in = valid && j < d;
                    Ks[t * KD + j] = in ? qkv[s0 + base + j * s3] : (u16)0;
                    Vt[j * chunk + t] = in ? qkv[2 * s0 + base + j * s3] : (u16)0;
                }
            }
        }
        __syncthreads();

        const int nkt = nc / 16;
        const int step = (nkt % 2 == 0) ? 2 : 1;
        #pragma unroll
        for (int tt = 0; tt < MAXT; tt++) {
            const int qt = wave + 8 * tt;
            if (qt >= ntiles) break;
            const float* bq = biasQ + ((long)h * NQ + qt * 16 + col) * NKp + c0;
            for (int kt0 = 0; kt0 < nkt; kt0 += step) {
                float sv[2][8];
                float mx = -__builtin_inff();
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    if (u >= step) break;
                    const int kt = kt0 + u;
                    v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (int s = 0; s < KS; s++) {
                        const v8s ka = *(const v8s*)(Ks + (kt * 16 + col) * KD + s * 16 + 8 * half);
                        sacc = mma<BF16>(ka, qb[tt][s], sacc);
                    }
                    const int k0 = kt * 16 + 8 * half;
                    const v4f b0 = *(const v4f*)(bq + k0), b1 = *(const v4f*)(bq + k0 + 4);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float v = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                        sv[u][i] = v;
                        mx = fmaxf(mx, v);
                    }
                }
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float mn = fmaxf(m[tt], mx);
                const float corr = __expf(m[tt] - mn);
                m[tt] = mn;
                lsum[tt] *= corr;
                #pragma unroll
                for (int t = 0; t < KS; t++)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) o[tt][t][i] *= corr;
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    if (u >= step) break;
                    const int kt = kt0 + u;
                    v8s pb;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float p = __expf(sv[u][i] - mn);
                        lsum[tt] += p;
                        pb[i] = (short)(BF16 ? f2bf(p) : f2h(p));
                    }
                    #pragma unroll
                    for (int t = 0; t < KS; t++) {
                        const v8s va = *(const v8s*)(Vt + (t * 16 + col) * chunk + kt * 16 + 8 * half);
                        o[tt][t] = mma<BF16>(va, pb, o[tt][t]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int tt = 0; tt < MAXT; tt++) {
        const int qt = wave + 8 * tt;
        if (qt >= ntiles) break;
        const float ls = lsum[tt] + __shfl_xor(lsum[tt], 16, 32);
        const float inv = 1.0f / ls;
        const int n = qt * 16 + col;
        const int y = iy * ws + n / ws, x = ix * ws + n % ws;
        const long ob = (((long)b * H + y) * W + x) * (long)(heads * d) + h * d;
        #pragma unroll
        for (int t = 0; t < KS; t++)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int j = t * 16 + 8 * half + i;
                if (j < d) out[ob + j] = BF16 ? f2bf(o[tt][t][i] * inv) : f2h(o[tt][t][i] * inv);
            }
    }
}

#define OCAB_ARGS long s0, long s1, long s2, long s3, const float* biasT, int H, int W, int ws, int ows, int heads, int d, float scale, int chunk
#define OCAB_PASS s0, s1, s2, s3, biasT, out, H, W, ws, ows, heads, d, scale, chunk
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_16(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 16>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_32(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 32>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_64(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 64>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_16(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 16>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_32(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 32>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_64(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 64>(qkv, OCAB_PASS); }
#define OCABT(S, B, KD) extern "C" __global__ void __launch_bounds__(256) ocab_wmmat_##S##_##KD(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmmat<B, KD>(qkv, OCAB_PASS); }
OCABT(bf16, true, 16) OCABT(bf16, true, 32) OCABT(bf16, true, 64)
OCABT(f16, false, 16) OCABT(f16, false, 32) OCABT(f16, false, 64)
"""
)

_RRDB_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ unsigned short bf16_bits(float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ void store_val(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long i, float v) { p[i] = bf16_bits(v); }

// Rounds a float to T and back, as PyTorch does after every elementwise op.
__device__ __forceinline__ float round_to(const float*, float v) { return v; }
__device__ __forceinline__ float round_to(const _Float16*, float v) { return (float)(_Float16)v; }
__device__ __forceinline__ float round_to(const bf16_t*, float v) {
    return __uint_as_float(((unsigned int)bf16_bits(v)) << 16);
}

// ESRGAN's dense blocks without torch.cat. Every block works in one buffer
// (B, nf + 4 * gc, H, W) holding [x | x1 | x2 | x3 | x4]; each conv reads a
// channel prefix of it and these kernels write its output into the next slot.
// All tensors are NCHW with H*W contiguous per channel; `*_stride` is the
// per-image element stride (channels of the underlying buffer * H * W).
// blockIdx.y is the image.

// dst[off:off+C] = act(src), act = LeakyReLU(slope) or ReLU.
template <typename T>
__device__ void act_into(const T* __restrict__ src, T* __restrict__ dst, long n,
                         long dst_stride, long dst_off, float slope, int relu) {
    const long b = blockIdx.y;
    const T* s = src + b * n;
    T* d = dst + b * dst_stride + dst_off;
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
        float v = load_val(s, i);
        store_val(d, i, v > 0.0f ? v : (relu ? 0.0f : v * slope));
    }
}

// dst = a * s1 + x1, then optionally * s2 + x2, rounding to T after every
// multiply and add like the separate PyTorch ops do.
template <typename T>
__device__ void res_add(const T* __restrict__ a, const T* __restrict__ x1, long x1_stride,
                        const T* __restrict__ x2, long x2_stride, T* __restrict__ dst,
                        long dst_stride, long n, float s1, float s2, int two) {
    const long b = blockIdx.y;
    const T* pa = a + b * n;
    const T* p1 = x1 + b * x1_stride;
    const T* p2 = two ? x2 + b * x2_stride : nullptr;
    T* d = dst + b * dst_stride;
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
        float t = round_to(d, load_val(pa, i) * s1);
        t = round_to(d, t + load_val(p1, i));
        if (two) {
            t = round_to(d, t * s2);
            t = round_to(d, t + load_val(p2, i));
        }
        store_val(d, i, t);
    }
}

#define RRDB_KERNELS(T, S)                                                                   \
    extern "C" __global__ void act_into_##S(const T* src, T* dst, long n, long dst_stride,   \
                                            long dst_off, float slope, int relu) {           \
        act_into<T>(src, dst, n, dst_stride, dst_off, slope, relu);                          \
    }                                                                                        \
    extern "C" __global__ void res_add_##S(const T* a, const T* x1, long x1_stride,          \
                                           const T* x2, long x2_stride, T* dst,              \
                                           long dst_stride, long n, float s1, float s2,      \
                                           int two) {                                        \
        res_add<T>(a, x1, x1_stride, x2, x2_stride, dst, dst_stride, n, s1, s2, two);        \
    }
RRDB_KERNELS(float, f32)
RRDB_KERNELS(_Float16, f16)
RRDB_KERNELS(bf16_t, bf16)
"""

_TW, _TH = 32, 16
_MAX_GRID_Z = 65535
# Batched transpose (B, R, S) -> (B, S, R) through 32x32 tiles in shared memory,
# so both the reads and the writes are coalesced. PyTorch's generic strided
# copy, which `.contiguous()` of a transposed view runs, reaches only a few
# percent of the memory bandwidth on RDNA for this pattern. Values are moved
# as raw bits, so one kernel per element size covers every dtype.
_TRANSPOSE_SRC = r"""
template <typename T>
__device__ void transpose(const T* __restrict__ x, T* __restrict__ y, long R, long S) {
    __shared__ T tile[32][33];
    const long b = blockIdx.z;
    const long r0 = (long)blockIdx.y * 32, s0 = (long)blockIdx.x * 32;
    const T* xb = x + b * R * S;
    T* yb = y + b * R * S;
    const int tx = threadIdx.x, ty = threadIdx.y;  // 32 x 8
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const long r = r0 + ty + k, c = s0 + tx;
        if (r < R && c < S) tile[ty + k][tx] = xb[r * S + c];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const long c = s0 + ty + k, r = r0 + tx;
        if (r < R && c < S) yb[c * R + r] = tile[tx][ty + k];
    }
}

extern "C" __global__ void transpose_16(const unsigned short* x, unsigned short* y, long R, long S) { transpose<unsigned short>(x, y, R, S); }
extern "C" __global__ void transpose_32(const unsigned int* x, unsigned int* y, long R, long S) { transpose<unsigned int>(x, y, R, S); }
"""

_DTYPE_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
# name -> (source, kernel names, extra compiler options)
_PROGRAMS: dict[str, tuple[str, list[str], list[str]]] = {
    "dwconv": (_DWCONV_SRC, [f"dwconv3x3_{s}" for s in _DTYPE_SUFFIX.values()], []),
    "transpose": (_TRANSPOSE_SRC, ["transpose_16", "transpose_32"], []),
    "win_attn": (
        _ATTN_SRC,
        [f"win_attn_{s}_{dm}" for s in _DTYPE_SUFFIX.values() for dm in (32, 64)],
        [],
    ),
    # gfx12 (RDNA4) only: uses the gfx12 WMMA builtins
    "win_attn_wmma": (
        _ATTN_WMMA_SRC,
        [
            f"win_attn_{v}_{s}_{kd}"
            for v in ("wmma", "wmmat")
            for s in ("bf16", "f16")
            for kd in (16, 32, 64)
        ],
        ["-std=c++17"],
    ),
    # gfx12 (RDNA4) only, like win_attn_wmma
    "ocab_wmma": (
        _OCAB_SRC,
        [
            f"ocab_{v}_{s}_{kd}"
            for v in ("wmma", "wmmat")
            for s in ("bf16", "f16")
            for kd in (16, 32, 64)
        ],
        ["-std=c++17"],
    ),
    "spatial_gate": (
        _SG_SRC,
        [f"sg_{k}_{s}" for k in ("stats", "gate") for s in _DTYPE_SUFFIX.values()]
        + [f"dw_tokens_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"add_ln_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"gate_mix_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"hab_ln_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"hab_ln_rows_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"bias_{k}_{s}" for k in ("act", "res") for s in _DTYPE_SUFFIX.values()]
        + [f"pad_add_{s}" for s in _DTYPE_SUFFIX.values()],
        [],
    ),
    # exact PyTorch rounding: no fused multiply-add
    "rrdb": (
        _RRDB_SRC,
        [f"{k}_{s}" for k in ("act_into", "res_add") for s in _DTYPE_SUFFIX.values()],
        ["-ffp-contract=off"],
    ),
}
_LDS_LIMIT = 64 * 1024


class _Hip:
    def __init__(self):
        root = self._find_sdk_root()
        if sys.platform == "win32":
            bin_dir = root / "bin"
            os.add_dll_directory(str(bin_dir))
            self.hip = ctypes.WinDLL(str(next(bin_dir.glob("amdhip64*.dll"))))
            self.rtc = ctypes.WinDLL(str(next(bin_dir.glob("hiprtc0*.dll"))))
        else:
            # torch has already loaded this libamdhip64, so this is the same
            # HIP runtime instance and its streams and pointers are valid here
            lib_dir = root / "lib"
            self.hip = ctypes.CDLL(str(next(lib_dir.glob("libamdhip64.so*"))))
            self.rtc = ctypes.CDLL(str(next(lib_dir.glob("libhiprtc.so*"))))
        self.funcs: dict[tuple[int, str], ctypes.c_void_p] = {}
        self.compiled: set[tuple[int, str]] = set()
        self.lock = threading.Lock()

    @staticmethod
    def _find_sdk_root() -> Path:
        import _rocm_sdk_core  # type: ignore

        return Path(_rocm_sdk_core.__file__).parent

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed with HIP error {err}")

    def _hiprtc_version(self) -> str:
        major, minor = ctypes.c_int(), ctypes.c_int()
        if self.rtc.hiprtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
            return "unknown"
        return f"{major.value}.{minor.value}"

    def _build(self, program: str, source: str, opts: list[bytes]) -> bytes:
        """Compiles a program with HIPRTC and returns the code object."""
        prog = ctypes.c_void_p()
        self._check(
            self.rtc.hiprtcCreateProgram(
                ctypes.byref(prog),
                source.encode(),
                f"{program}.hip".encode(),
                0,
                None,
                None,
            ),
            "hiprtcCreateProgram",
        )
        try:
            err = self.rtc.hiprtcCompileProgram(
                prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)
            )
            if err != 0:
                size = ctypes.c_size_t()
                self.rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.rtc.hiprtcGetProgramLog(prog, log)
                raise RuntimeError(
                    "HIPRTC compilation failed: " + log.value.decode(errors="replace")
                )
            size = ctypes.c_size_t()
            self._check(
                self.rtc.hiprtcGetCodeSize(prog, ctypes.byref(size)),
                "hiprtcGetCodeSize",
            )
            code = ctypes.create_string_buffer(size.value)
            self._check(self.rtc.hiprtcGetCode(prog, code), "hiprtcGetCode")
            return code.raw
        finally:
            self.rtc.hiprtcDestroyProgram(ctypes.byref(prog))

    def _load(self, device_index: int, names: list[str], code: bytes) -> None:
        with torch.cuda.device(device_index):
            torch.cuda.current_stream(device_index)  # ensure the context exists
            module = ctypes.c_void_p()
            self._check(
                self.hip.hipModuleLoadData(ctypes.byref(module), ctypes.c_char_p(code)),
                "hipModuleLoadData",
            )
            funcs = {}
            for name in names:
                fn = ctypes.c_void_p()
                self._check(
                    self.hip.hipModuleGetFunction(
                        ctypes.byref(fn), module, name.encode()
                    ),
                    "hipModuleGetFunction",
                )
                funcs[(device_index, name)] = fn
            self.funcs.update(funcs)

    def _compile(self, device_index: int, program: str) -> None:
        """
        Loads a program's kernels for a device, from the on-disk cache if
        possible. Compiled code objects are kept in the directory from _cache_dir(),
        keyed by everything that affects the result: kernel source, compiler
        options, GPU architecture and HIPRTC version. A changed kernel or driver
        simply gets a new file; stale files of the same program and GPU are
        removed.
        """
        source, names, extra_opts = _PROGRAMS[program]
        arch = torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
        opts = [f"--offload-arch={arch}".encode(), b"-O3"] + [
            o.encode() for o in extra_opts
        ]

        digest = hashlib.sha256(
            b"|".join([source.encode(), *opts, self._hiprtc_version().encode()])
        ).hexdigest()[:24]
        prefix = f"{program}-{arch}-"
        cache_dir = _cache_dir()
        path = cache_dir / f"{prefix}{digest}.co" if cache_dir else None

        if path is not None and path.is_file():
            try:
                self._load(device_index, names, path.read_bytes())
                logger.debug("Loaded cached HIP %s kernels for %s", program, arch)
                return
            except Exception as e:  # noqa: BLE001
                logger.info("Ignoring unusable cached HIP %s kernels: %s", program, e)
                try:
                    path.unlink()
                except OSError:
                    pass

        code = self._build(program, source, opts)
        self._load(device_index, names, code)
        logger.info("Compiled HIP %s kernels for %s", program, arch)

        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                for old in path.parent.glob(f"{prefix}*.co"):
                    if old != path:
                        old.unlink(missing_ok=True)
                # write a temp file first, so a concurrent reader never sees half a file
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_bytes(code)
                os.replace(tmp, path)
            except OSError as e:
                logger.debug("Could not cache HIP %s kernels: %s", program, e)

    def get(self, device_index: int, program: str, name: str) -> ctypes.c_void_p:
        fn = self.funcs.get((device_index, name))
        if fn is None:
            with self.lock:
                if (device_index, program) not in self.compiled:
                    self._compile(device_index, program)
                    self.compiled.add((device_index, program))
                fn = self.funcs[(device_index, name)]
        return fn

    def launch(
        self,
        fn: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list,
        stream: ctypes.c_void_p,
        shared_mem: int = 0,
    ) -> None:
        params = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
        )
        self._check(
            self.hip.hipModuleLaunchKernel(
                fn, *grid, *block, shared_mem, stream, params, None
            ),
            "hipModuleLaunchKernel",
        )


_hip: _Hip | None = None
_hip_lock = threading.Lock()


def _cache_dir() -> Path | None:
    """
    Where compiled kernels are kept between runs: SPANDREL_HIP_CACHE_DIR if set
    (an empty value disables the cache), otherwise %LOCALAPPDATA%/spandrel/hip_cache
    on Windows and $XDG_CACHE_HOME/spandrel/hip_cache (~/.cache) elsewhere.
    """
    custom = os.environ.get("SPANDREL_HIP_CACHE_DIR")
    if custom is not None:
        return Path(custom) if custom.strip() else None
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) / "spandrel" / "hip_cache" if local else None
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "spandrel" / "hip_cache"


def hip_kernels_available() -> bool:
    """
    Only for AMD's ROCm builds of PyTorch (Windows and Linux), where HIPRTC ships
    with torch in the _rocm_sdk_core package.
    """
    if os.environ.get("SPANDREL_HIP_KERNELS", "").strip() == "0":
        return False
    if sys.platform not in ("win32", "linux"):
        return False
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    return _has_rocm_sdk()


@functools.cache
def _has_rocm_sdk() -> bool:
    try:
        import _rocm_sdk_core  # type: ignore # noqa: F401
    except ImportError:
        return False
    return True


def _get_hip() -> _Hip:
    global _hip
    if _hip is None:
        with _hip_lock:
            if _hip is None:
                _hip = _Hip()
    return _hip


def _stream(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream(t.device).cuda_stream)


# --- depthwise 3x3 convolution ----------------------------------------------


def can_dwconv3x3(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 4
        and x.dtype in _DTYPE_SUFFIX
        and not x.requires_grad
        and x.shape[0] * x.shape[1] <= _MAX_GRID_Z
    )


def dwconv3x3(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """
    Depthwise 3x3 convolution, stride 1, zero padding 1.

    x: (B, C, H, W). weight: (C, 9) float32 contiguous. bias: (C,) float32 or None.
    """
    x = x.contiguous()
    B, C, H, W = x.shape  # noqa: N806
    hip = _get_hip()
    fn = hip.get(x.device.index or 0, "dwconv", f"dwconv3x3_{_DTYPE_SUFFIX[x.dtype]}")
    y = torch.empty_like(x)
    args = [
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(C),
        ctypes.c_int(H),
        ctypes.c_int(W),
    ]
    grid = ((W + _TW - 1) // _TW, (H + _TH - 1) // _TH, B * C)
    hip.launch(fn, grid, (_TW, _TH, 1), args, _stream(x))
    return y


# --- DAT window attention ------------------------------------------------------


def _attn_lds_bytes(n: int, d: int, dtype: torch.dtype) -> int:
    # must match the kernel: K/V rows padded to a multiple of the 8-byte vector
    elem = torch.tensor([], dtype=dtype).element_size()
    vec = 8 // elem
    dp = (d + vec - 1) // vec * vec
    return 2 * n * dp * elem + n * 4


def can_window_attention(
    qkv: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    H_sp: int,  # noqa: N803
    W_sp: int,  # noqa: N803
    heads: int,
) -> bool:
    n = H_sp * W_sp
    if not (qkv.is_cuda and qkv.dim() == 4 and qkv.dtype in _DTYPE_SUFFIX):
        return False
    if qkv.requires_grad or H % H_sp or W % W_sp or n % 16 or n > 1024:
        return False
    c = qkv.shape[-1]
    if c % heads:
        return False
    d = c // heads
    return d <= 64 and _attn_lds_bytes(n, d, qkv.dtype) <= _LDS_LIMIT


_gfx12: dict[int, bool] = {}
_wmma_failed = False


def _is_gfx12(device: int) -> bool:
    if device not in _gfx12:
        arch = torch.cuda.get_device_properties(device).gcnArchName
        _gfx12[device] = arch.startswith("gfx12")
    return _gfx12[device]


def _wmma_lds_bytes(n: int, kd: int) -> int:
    # must match win_attn_wmma: K [N][KD], V^T [KD][N], 8 P tiles, labels
    return 2 * n * kd * 2 + 8 * 256 * 2 + n * 4


def _wmmat_lds_bytes(n: int, kd: int) -> int:
    # must match win_attn_wmmat: K [N][KD], V^T [KD][N], labels
    return 2 * n * kd * 2 + n * 4


def _wmma_kd(qkv: torch.Tensor, d: int) -> int:
    """The head dim padded for WMMA (16/32/64), or 0 if WMMA does not apply."""
    if _wmma_failed or os.environ.get("SPANDREL_HIP_WMMA", "").strip() == "0":
        return 0
    if qkv.dtype not in (torch.bfloat16, torch.float16) or not _is_gfx12(
        qkv.device.index or 0
    ):
        return 0
    return 16 if d <= 16 else 32 if d <= 32 else 64 if d <= 64 else 0


def _wmma_head_dim(qkv: torch.Tensor, n: int, d: int) -> int:
    """The padded head dim for the WMMA kernel, or 0 if it does not apply."""
    kd = _wmma_kd(qkv, d)
    if not kd or n % 16 or _wmma_lds_bytes(n, kd) > _LDS_LIMIT:
        return 0
    return kd


_OCAB_MAX_QUERIES = 8 * 16 * 2  # 8 waves x 16 rows x MAXT tiles in ocab_wmma
_ocab_failed = False


def _ocab_chunk(n_keys: int, kd: int) -> int:
    """Keys staged in shared memory at a time: K and V^T, plus 8 P tiles."""
    fit = (_LDS_LIMIT - 8 * 256 * 2) // (2 * kd * 2) // 16 * 16
    return min(fit, (n_keys + 15) // 16 * 16)


def _ocab_chunk_t(n_keys: int, kd: int) -> int:
    """
    Keys per chunk for ocab_wmmat: the padded keys split evenly into as few
    chunks as fit in 40 KB of K and V^T.
    """
    nkp = (n_keys + 15) // 16 * 16
    fit = max(16, 40 * 1024 // (2 * kd * 2) // 16 * 16)
    chunks = -(-nkp // fit)
    return (-(-nkp // chunks) + 15) // 16 * 16


_ocab_bias_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _ocab_bias_query_major(bias_t: torch.Tensor, n_keys: int) -> torch.Tensor:
    """(heads, NK, NQ) -> (heads, NQ, NKp), padding key slots -inf; cached."""
    key = (bias_t.data_ptr(), _version_of(bias_t), n_keys)
    hit = _ocab_bias_cache.get(key)
    if hit is None or hit[0] is not bias_t:
        heads, _, nq = bias_t.shape
        nkp = (n_keys + 15) // 16 * 16
        q = torch.full(
            (heads, nq, nkp), float("-inf"), device=bias_t.device, dtype=torch.float32
        )
        q[:, :, :n_keys] = bias_t.transpose(1, 2)
        if len(_ocab_bias_cache) >= 64:
            _ocab_bias_cache.pop(next(iter(_ocab_bias_cache)))
        hit = (bias_t, q)
        _ocab_bias_cache[key] = hit
    return hit[1]


def can_ocab_attention(
    qkv: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    ws: int,
    ows: int,
    heads: int,
) -> bool:
    """Whether ocab_attention applies (16-bit qkv on gfx12)."""
    if _ocab_failed or not (qkv.is_cuda and qkv.dim() == 4) or qkv.requires_grad:
        return False
    c = qkv.shape[-1]
    if c % heads or H % ws or W % ws or (ows - ws) % 2 or ows < ws:
        return False
    nq = ws * ws
    if nq % 16 or nq > _OCAB_MAX_QUERIES:
        return False
    # keys are staged in chunks, so only the head dim limits shared memory
    return _wmma_kd(qkv, c // heads) > 0


def ocab_attention(
    qkv: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    ws: int,
    ows: int,
    heads: int,
    scale: float,
    bias_t: torch.Tensor,
) -> torch.Tensor:
    """
    HAT's overlapping cross attention in one kernel. qkv: (3, B, H*W, C), any
    strides; queries from the ws x ws windows, keys/values from the ows x ows
    windows around them (zero outside the image). bias_t: (heads, ows*ows,
    ws*ws) float32 (key, query), contiguous. Returns (B, H, W, C).
    """
    global _ocab_failed
    _, B, _L, C = qkv.shape  # noqa: N806
    d = C // heads
    kd = _wmma_kd(qkv, d)
    transposed = kd <= 32 and os.environ.get("SPANDREL_HIP_WMMA_T", "").strip() != "0"
    if transposed:
        chunk = _ocab_chunk_t(ows * ows, kd)
        shared = 2 * chunk * kd * 2
        bias_t = _ocab_bias_query_major(bias_t, ows * ows)
    else:
        chunk = _ocab_chunk(ows * ows, kd)
        shared = 2 * chunk * kd * 2 + 8 * 256 * 2
    out = torch.empty((B, H, W, C), device=qkv.device, dtype=qkv.dtype)
    hip = _get_hip()
    try:
        fn = hip.get(
            qkv.device.index or 0,
            "ocab_wmma",
            f"ocab_{'wmmat' if transposed else 'wmma'}_{_DTYPE_SUFFIX[qkv.dtype]}_{kd}",
        )
    except Exception:
        _ocab_failed = True
        raise
    s0, s1, s2, s3 = qkv.stride()
    blocks = B * (H // ws) * (W // ws) * heads
    hip.launch(
        fn,
        (blocks, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(s0),
            ctypes.c_int64(s1),
            ctypes.c_int64(s2),
            ctypes.c_int64(s3),
            ctypes.c_void_p(bias_t.data_ptr()),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(ws),
            ctypes.c_int(ows),
            ctypes.c_int(heads),
            ctypes.c_int(d),
            ctypes.c_float(scale),
            ctypes.c_int(chunk),
        ],
        _stream(qkv),
        shared_mem=shared,
    )
    return out


_bias_q_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _version_of(t: torch.Tensor) -> int:
    # like common.version_of (common imports this module)
    try:
        return t._version
    except RuntimeError:
        return -1


def clear_caches() -> None:
    """Drops the bias layouts cached here; called by common.clear_caches."""
    _bias_q_cache.clear()
    _ocab_bias_cache.clear()


def _bias_query_major(bias_t: torch.Tensor) -> torch.Tensor:
    """(heads, key, query) -> contiguous (heads, query, key), cached."""
    key = (bias_t.data_ptr(), _version_of(bias_t), bias_t.shape[-1])
    hit = _bias_q_cache.get(key)
    if hit is None or hit[0] is not bias_t:
        if len(_bias_q_cache) >= 64:
            _bias_q_cache.pop(next(iter(_bias_q_cache)))
        hit = (bias_t, bias_t.transpose(1, 2).contiguous())
        _bias_q_cache[key] = hit
    return hit[1]


def window_attention(
    qkv: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    H_sp: int,  # noqa: N803
    W_sp: int,  # noqa: N803
    heads: int,
    scale: float,
    bias_t: torch.Tensor | None,
    labels: torch.Tensor | None,
    out: torch.Tensor | None = None,
    out_offset: int = 0,
    shift: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """
    DAT window attention for one branch, fused into a single kernel.

    qkv: (3, B, Hc*Wc, heads*d), any strides. H and W are the window grid,
        which may be larger than the data (Hc x Wc, from `out`); positions
        outside the data count as zero, like DAT's zero padding.
    bias_t: (heads, N, N) float32, transposed (key, query), contiguous.
    labels: (nW, N) int32 contiguous.
    shift: (sy, sx) for shifted windows: the result equals rolling qkv by
        (-sy, -sx), attending, and rolling the result back by (sy, sx).

    Without `out`, returns (B, H, W, heads*d) like DAT's windows2img. With
    `out` (B, Hc, Wc, Cout) contiguous, writes into channels
    [out_offset, out_offset + heads*d) of it instead, dropping positions
    outside Hc x Wc (the window padding).
    """
    _, B, _L, C = qkv.shape  # noqa: N806
    d = C // heads
    n = H_sp * W_sp
    if out is None:
        out = torch.empty((B, H, W, C), device=qkv.device, dtype=qkv.dtype)
    _, Hc, Wc, out_c = out.shape  # noqa: N806
    hip = _get_hip()
    device = qkv.device.index or 0
    kd = _wmma_head_dim(qkv, n, d)
    fn: ctypes.c_void_p | None = None
    variant = ""
    if kd:
        try:
            # the transposed kernel is faster up to 32 channels per head
            variant = (
                "wmma"
                if kd > 32 or os.environ.get("SPANDREL_HIP_WMMA_T", "").strip() == "0"
                else "wmmat"
            )
            fn = hip.get(
                device,
                "win_attn_wmma",
                f"win_attn_{variant}_{_DTYPE_SUFFIX[qkv.dtype]}_{kd}",
            )
        except Exception as e:  # noqa: BLE001
            # keep the scalar kernel, only the matrix core path is lost
            global _wmma_failed
            _wmma_failed = True
            logger.warning(
                "WMMA attention kernel unavailable, using the scalar one: %s", e
            )
            kd = 0
    if fn is not None:
        block, shared = 256, _wmma_lds_bytes(n, kd)
        if variant == "wmmat":
            shared = _wmmat_lds_bytes(n, kd)
    else:
        name = f"win_attn_{_DTYPE_SUFFIX[qkv.dtype]}_{32 if d <= 32 else 64}"
        fn = hip.get(device, "win_attn", name)
        block, shared = n, _attn_lds_bytes(n, d, qkv.dtype)
    if variant == "wmmat" and bias_t is not None:
        bias_t = _bias_query_major(bias_t)
    s0, s1, s2, s3 = qkv.stride()
    args = [
        ctypes.c_void_p(qkv.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(s0),
        ctypes.c_int64(s1),
        ctypes.c_int64(s2),
        ctypes.c_int64(s3),
        ctypes.c_void_p(bias_t.data_ptr() if bias_t is not None else 0),
        ctypes.c_void_p(labels.data_ptr() if labels is not None else 0),
        ctypes.c_int64(out_c),
        ctypes.c_int(out_offset),
        ctypes.c_int(Hc),
        ctypes.c_int(Wc),
        ctypes.c_int(shift[0] % H),
        ctypes.c_int(shift[1] % W),
        ctypes.c_int(H),
        ctypes.c_int(W),
        ctypes.c_int(H_sp),
        ctypes.c_int(W_sp),
        ctypes.c_int(heads),
        ctypes.c_int(d),
        ctypes.c_float(scale),
    ]
    blocks = B * (H // H_sp) * (W // W_sp) * heads
    hip.launch(fn, (blocks, 1, 1), (block, 1, 1), args, _stream(qkv), shared)
    return out


# --- DAT spatial gate ----------------------------------------------------------

_SG_CC, _SG_SX, _SG_SY = 32, 8, 8


def can_spatial_gate(x: torch.Tensor, H: int, W: int) -> bool:  # noqa: N803
    return (
        x.is_cuda
        and x.dim() == 3
        and x.dtype in _DTYPE_SUFFIX
        and not x.requires_grad
        and x.shape[1] == H * W
        and x.shape[2] % 2 == 0
        and x.is_contiguous()
    )


def spatial_gate(
    x: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    DAT's SpatialGate: x1 * dwconv3x3(LayerNorm(x2)), in the (B, H*W, 2C) layout.

    gamma/beta: (C,) float32. weight: (C, 9) float32. bias: (C,) float32 or None.
    Returns (B, H*W, C).
    """
    B, L, C2 = x.shape  # noqa: N806
    C = C2 // 2  # noqa: N806
    hip = _get_hip()
    device = x.device.index or 0
    sfx = _DTYPE_SUFFIX[x.dtype]
    stream = _stream(x)
    tokens = B * L
    stats = torch.empty((tokens, 2), device=x.device, dtype=torch.float32)
    hip.launch(
        hip.get(device, "spatial_gate", f"sg_stats_{sfx}"),
        ((tokens + 7) // 8, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_void_p(stats.data_ptr()),
        ],
        stream,
    )
    out = torch.empty((B, L, C), device=x.device, dtype=x.dtype)
    chunks = (C + _SG_CC - 1) // _SG_CC
    hip.launch(
        hip.get(device, "spatial_gate", f"sg_gate_{sfx}"),
        ((W + _SG_SX - 1) // _SG_SX, (H + _SG_SY - 1) // _SG_SY, B * chunks),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(stats.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
            ctypes.c_int(C),
            ctypes.c_int(H),
            ctypes.c_int(W),
        ],
        stream,
    )
    return out


def dwconv_tokens(
    x: torch.Tensor,
    ch_offset: int,
    channels: int,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gelu: bool,
    gelu_in: bool = False,
) -> torch.Tensor:
    """
    Depthwise 3x3 conv on channels [ch_offset, ch_offset + channels) of a
    (B, H*W, row) token tensor with contiguous rows, optionally followed by GELU
    and/or preceded by GELU (gelu_in).

    weight: (channels, 9) float32. bias: (channels,) float32 or None.
    Returns (B, H*W, channels).
    """
    B, L, _ = x.shape  # noqa: N806
    if x.stride(2) != 1 or x.stride(0) != L * x.stride(1):
        raise ValueError("token tensor rows must be contiguous")
    hip = _get_hip()
    out = torch.empty((B, L, channels), device=x.device, dtype=x.dtype)
    chunks = (channels + _SG_CC - 1) // _SG_CC
    hip.launch(
        hip.get(
            x.device.index or 0, "spatial_gate", f"dw_tokens_{_DTYPE_SUFFIX[x.dtype]}"
        ),
        ((W + _SG_SX - 1) // _SG_SX, (H + _SG_SY - 1) // _SG_SY, B * chunks),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(x.stride(1)),
            ctypes.c_int(ch_offset),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
            ctypes.c_int((1 if gelu else 0) | (2 if gelu_in else 0)),
            ctypes.c_int(channels),
            ctypes.c_int(H),
            ctypes.c_int(W),
        ],
        _stream(x),
    )
    return out


# --- LayerNorm (+ residual add) -----------------------------------------------


def can_layer_norm(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in _DTYPE_SUFFIX
        and not x.requires_grad
        and x.is_contiguous()
        and 0 < x.shape[-1] <= 256
    )


def layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out_channels: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    LayerNorm over the last dim of a contiguous tensor. With `residual`, first
    computes x + residual and returns (that sum, its LayerNorm); otherwise
    (None, LayerNorm(x)). gamma/beta: float32 contiguous. With out_channels
    (>= C), the LayerNorm rows are that wide, zero past C (padding for GEMMs).
    """
    C = x.shape[-1]  # noqa: N806
    tokens = x.numel() // C
    out_ld = C if out_channels is None else out_channels
    out = torch.empty((*x.shape[:-1], out_ld), device=x.device, dtype=x.dtype)
    sum_out = torch.empty_like(x) if residual is not None else None
    if residual is not None:
        residual = residual.contiguous()
    hip = _get_hip()
    hip.launch(
        hip.get(
            x.device.index or 0, "spatial_gate", f"add_ln_{_DTYPE_SUFFIX[x.dtype]}"
        ),
        ((tokens + 7) // 8, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(residual.data_ptr() if residual is not None else 0),
            ctypes.c_void_p(sum_out.data_ptr() if sum_out is not None else 0),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_int(out_ld),
        ],
        _stream(x),
    )
    return sum_out, out


def hab_layer_norm(
    x: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    ca: torch.Tensor,
    scale: float,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    conv_bias: torch.Tensor | None = None,
    out_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HAT's `(x + a) + conv * ca * scale` and its LayerNorm in one pass: returns
    (sum, LayerNorm(sum)). x, a: contiguous (B, L, C); conv: (B, C, H, W) with
    H*W = L, contiguous or channels_last; ca: (B, C) float32; gamma/beta: float32.
    conv_bias: (C,) float32 added to a channels_last conv first, or None. A
    channels_last conv may have more than C channels (zero padding), of which
    the first C are used.
    out_channels: like layer_norm's (channels_last conv only).
    """
    B, L, C = x.shape  # noqa: N806
    if C > 256:
        raise ValueError("too many channels")
    tokens = B * L
    conv_nhwc = is_channels_last(conv)
    a, ca = a.contiguous(), ca.contiguous()
    out_ld = C if out_channels is None else out_channels
    if out_ld != C and not conv_nhwc:
        raise ValueError("out_channels needs a channels_last conv")
    sum_out = torch.empty_like(x)
    out = torch.empty((B, L, out_ld), device=x.device, dtype=x.dtype)
    hip = _get_hip()
    if conv_nhwc:
        hip.launch(
            hip.get(
                x.device.index or 0,
                "spatial_gate",
                f"hab_ln_rows_{_DTYPE_SUFFIX[x.dtype]}",
            ),
            ((tokens + 7) // 8, 1, 1),
            (256, 1, 1),
            [
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(conv.data_ptr()),
                ctypes.c_void_p(conv_bias.data_ptr() if conv_bias is not None else 0),
                ctypes.c_void_p(ca.data_ptr()),
                ctypes.c_float(scale),
                ctypes.c_void_p(sum_out.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(gamma.data_ptr()),
                ctypes.c_void_p(beta.data_ptr()),
                ctypes.c_int(C),
                ctypes.c_int64(L),
                ctypes.c_int64(tokens),
                ctypes.c_float(eps),
                ctypes.c_int(out_ld),
                ctypes.c_int(conv.shape[1]),
            ],
            _stream(x),
        )
        return sum_out, out
    if conv_bias is not None:
        raise ValueError("conv_bias needs a channels_last conv")
    conv = conv.contiguous()
    hip.launch(
        hip.get(
            x.device.index or 0, "spatial_gate", f"hab_ln_{_DTYPE_SUFFIX[x.dtype]}"
        ),
        ((tokens + 31) // 32, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(conv.data_ptr()),
            ctypes.c_void_p(ca.data_ptr()),
            ctypes.c_float(scale),
            ctypes.c_void_p(sum_out.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(L),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_int(1 if conv_nhwc else 0),
        ],
        _stream(x),
    )
    return sum_out, out


# --- layout conversion ---------------------------------------------------------


def _rows_launch(n: int) -> tuple[int, int, int]:
    return (min((n + 255) // 256, 65535), 1, 1)


def bias_act_(y: torch.Tensor, bias: torch.Tensor, gelu: bool) -> torch.Tensor:
    """
    In place on a channels_last conv output: y = act(y + bias[c]), act = exact
    GELU (nn.GELU()) or none. bias: (C,) float32 contiguous.
    """
    C = y.shape[1]  # noqa: N806
    hip = _get_hip()
    hip.launch(
        hip.get(
            y.device.index or 0, "spatial_gate", f"bias_act_{_DTYPE_SUFFIX[y.dtype]}"
        ),
        _rows_launch(y.numel()),
        (256, 1, 1),
        [
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(y.numel()),
            ctypes.c_int(1 if gelu else 0),
        ],
        _stream(y),
    )
    return y


def bias_residual_rows(
    y: torch.Tensor, bias: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """
    (y + bias[c]) + r for a channels_last conv output y (B, Cy, H, W) and
    contiguous token rows r (B, H*W, C), Cy >= C (the rest is padding);
    returns token rows like r.
    """
    C = r.shape[-1]  # noqa: N806
    r = r.contiguous()
    out = torch.empty_like(r)
    hip = _get_hip()
    hip.launch(
        hip.get(
            y.device.index or 0, "spatial_gate", f"bias_res_{_DTYPE_SUFFIX[y.dtype]}"
        ),
        _rows_launch(y.numel()),
        (256, 1, 1),
        [
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(r.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(r.numel()),
            ctypes.c_int(y.shape[1]),
        ],
        _stream(y),
    )
    return out


def pad_add(x: torch.Tensor, a: torch.Tensor | None, width: int) -> torch.Tensor:
    """x + a (contiguous rows of C) as rows of `width` >= C, zero past C."""
    C = x.shape[-1]  # noqa: N806
    x = x.contiguous()
    if a is not None:
        a = a.contiguous()
    out = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    hip = _get_hip()
    hip.launch(
        hip.get(
            x.device.index or 0, "spatial_gate", f"pad_add_{_DTYPE_SUFFIX[x.dtype]}"
        ),
        _rows_launch(out.numel()),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(a.data_ptr() if a is not None else 0),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int(width),
            ctypes.c_int64(out.numel()),
        ],
        _stream(x),
    )
    return out


def is_channels_last(t: torch.Tensor) -> bool:
    """A 4D tensor stored as (B, H, W, C) rows (and not also plain contiguous)."""
    return (
        t.dim() == 4
        and t.is_contiguous(memory_format=torch.channels_last)
        and not t.is_contiguous()
    )


def transpose_last2(x: torch.Tensor) -> torch.Tensor:
    """
    (B, R, S) -> contiguous (B, S, R), e.g. token rows (B, L, C) to the image
    layout (B, C, L) or back. x must be contiguous.
    """
    B, R, S = x.shape  # noqa: N806
    y = torch.empty((B, S, R), device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return y
    if B > _MAX_GRID_Z:
        raise ValueError("batch too large")
    name = {2: "transpose_16", 4: "transpose_32"}[x.element_size()]
    hip = _get_hip()
    hip.launch(
        hip.get(x.device.index or 0, "transpose", name),
        ((S + 31) // 32, (R + 31) // 32, B),
        (32, 8, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int64(R),
            ctypes.c_int64(S),
        ],
        _stream(x),
    )
    return y


def can_transpose(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 3
        and x.element_size() in (2, 4)
        and not x.requires_grad
        and x.is_contiguous()
        and x.shape[0] <= _MAX_GRID_Z
        and x.shape[1] <= 65535 * 32
    )


_nhwc_conv: bool | None = None


def nhwc_conv_available(device: torch.device, dtype: torch.dtype) -> bool:
    """
    Whether PyTorch runs a convolution on a channels_last input in NHWC (and
    returns a channels_last output) instead of first copying it to NCHW. On
    ROCm that depends on PYTORCH_MIOPEN_SUGGEST_NHWC, which PyTorch reads once.
    """
    global _nhwc_conv
    if _nhwc_conv is None:
        with torch.no_grad():
            x = torch.zeros((1, 8, 8, 8), device=device, dtype=dtype)
            w = torch.zeros((8, 8, 3, 3), device=device, dtype=dtype)
            y = torch.nn.functional.conv2d(
                x.to(memory_format=torch.channels_last),
                w.to(memory_format=torch.channels_last),
                padding=1,
            )
        _nhwc_conv = is_channels_last(y)
        logger.debug("NHWC convolutions: %s", _nhwc_conv)
    return _nhwc_conv


# --- adaptive interaction gating ------------------------------------------------


def gate_mix(
    a: torch.Tensor,
    b: torch.Tensor,
    tok: torch.Tensor,
    ch: torch.Tensor,
    a_uses_tok: bool,
) -> torch.Tensor:
    """
    a * sigmoid(tok) + b * sigmoid(ch) (or with tok/ch swapped between a and b)
    for (B, N, C) tensors a and b, a per-token map tok (B, N, 1) and a
    per-channel map ch (B, 1, C), in one pass.
    """
    B, N, C = a.shape  # noqa: N806
    a, b = a.contiguous(), b.contiguous()
    tok = tok.reshape(B, N).to(a.dtype).contiguous()
    ch = ch.reshape(B, C).to(a.dtype).contiguous()
    out = torch.empty_like(a)
    total = a.numel()
    hip = _get_hip()
    hip.launch(
        hip.get(
            a.device.index or 0, "spatial_gate", f"gate_mix_{_DTYPE_SUFFIX[a.dtype]}"
        ),
        (min((total + 255) // 256, 65535), 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(tok.data_ptr()),
            ctypes.c_void_p(ch.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(N),
            ctypes.c_int64(total),
            ctypes.c_int(1 if a_uses_tok else 0),
        ],
        _stream(a),
    )
    return out


# --- ESRGAN dense blocks -------------------------------------------------------


def _image_stride(t: torch.Tensor) -> int | None:
    """
    The per-image element stride of an NCHW tensor whose channels each hold a
    contiguous H*W plane (e.g. a channel prefix of a larger buffer), or None.
    """
    _, c, h, w = t.shape
    if t.stride(3) != 1 or t.stride(2) != w or t.stride(1) != h * w:
        return None
    return t.stride(0) if t.shape[0] > 1 else c * h * w


def can_rrdb(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 4
        and x.dtype in _DTYPE_SUFFIX
        and not x.requires_grad
        and x.shape[0] <= _MAX_GRID_Z
        and _image_stride(x) is not None
    )


def _grid(n: int, batch: int) -> tuple[int, int, int]:
    return (max(1, min((n + 255) // 256, 4096)), batch, 1)


def act_into(
    src: torch.Tensor, dst: torch.Tensor, channel_offset: int, slope: float, relu: bool
) -> None:
    """
    dst[:, channel_offset : channel_offset + C] = LeakyReLU(src) (or ReLU), where
    dst is an NCHW buffer and src a (B, C, H, W) tensor.
    """
    b, c, h, w = src.shape
    src = src.contiguous()
    dst_stride = _image_stride(dst)
    if dst_stride is None:
        raise ValueError("unsupported destination layout")
    n = c * h * w
    hip = _get_hip()
    hip.launch(
        hip.get(dst.device.index or 0, "rrdb", f"act_into_{_DTYPE_SUFFIX[src.dtype]}"),
        _grid(n, b),
        (256, 1, 1),
        [
            ctypes.c_void_p(src.data_ptr()),
            ctypes.c_void_p(dst.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_int64(dst_stride),
            ctypes.c_int64(channel_offset * h * w),
            ctypes.c_float(slope),
            ctypes.c_int(1 if relu else 0),
        ],
        _stream(src),
    )


def residual_add(
    a: torch.Tensor,
    x1: torch.Tensor,
    dst: torch.Tensor,
    s1: float,
    x2: torch.Tensor | None = None,
    s2: float = 0.0,
) -> None:
    """
    dst = a * s1 + x1, then * s2 + x2 if x2 is given, with the rounding of the
    separate PyTorch ops. a is (B, C, H, W); x1, x2 and dst may be channel
    prefixes of larger NCHW buffers.
    """
    b, c, h, w = a.shape
    a = a.contiguous()
    x1_stride = _image_stride(x1)
    dst_stride = _image_stride(dst)
    x2_stride = _image_stride(x2) if x2 is not None else 0
    if x1_stride is None or dst_stride is None or x2_stride is None:
        raise ValueError("unsupported tensor layout")
    n = c * h * w
    hip = _get_hip()
    hip.launch(
        hip.get(a.device.index or 0, "rrdb", f"res_add_{_DTYPE_SUFFIX[a.dtype]}"),
        _grid(n, b),
        (256, 1, 1),
        [
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(x1.data_ptr()),
            ctypes.c_int64(x1_stride),
            ctypes.c_void_p(x2.data_ptr() if x2 is not None else 0),
            ctypes.c_int64(x2_stride),
            ctypes.c_void_p(dst.data_ptr()),
            ctypes.c_int64(dst_stride),
            ctypes.c_int64(n),
            ctypes.c_float(s1),
            ctypes.c_float(s2),
            ctypes.c_int(1 if x2 is not None else 0),
        ],
        _stream(a),
    )
