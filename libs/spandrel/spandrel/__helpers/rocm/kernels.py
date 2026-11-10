from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import math
import os
import sys
import threading
from pathlib import Path

import torch

from . import hipblaslt
from .common import version_of

logger = logging.getLogger("spandrel.rocm")

# Custom HIP kernels for operations that MIOpen handles poorly on consumer
# Radeon cards. They are compiled at runtime with HIPRTC, which ships with
# AMD's ROCm wheels, for whatever GPU architecture is present, and launched on
# PyTorch's current stream. No compiler or SDK has to be installed.

_DWCONV_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long long i, float v) {
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
    const long long plane = (long long)bc * H * W;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
    const int tid = threadIdx.y * TW + threadIdx.x;

    for (int i = tid; i < (TH + 2) * (TW + 2); i += TW * TH) {
        int ty = i / (TW + 2), tx = i % (TW + 2);
        int gy = y0 + ty - 1, gx = x0 + tx - 1;
        tile[ty][tx] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
            ? load_val(x, plane + (long long)gy * W + gx) : 0.0f;
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
    store_val(y, plane + (long long)oy * W + ox, acc);
}

extern "C" __global__ void dwconv3x3_f32(const float* x, const float* w, const float* b, float* y, int C, int H, int W) { dwconv3x3<float>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_f16(const _Float16* x, const float* w, const float* b, _Float16* y, int C, int H, int W) { dwconv3x3<_Float16>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_bf16(const bf16_t* x, const float* w, const float* b, bf16_t* y, int C, int H, int W) { dwconv3x3<bf16_t>(x, w, b, y, C, H, W); }
"""


_ATTN_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long long i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    p[i] = (bf16_t)(u >> 16);
}
__device__ __forceinline__ void copy_val(float* d, long long di, const float* s, long long si) { d[di] = s[si]; }
__device__ __forceinline__ void copy_val(_Float16* d, long long di, const _Float16* s, long long si) { d[di] = s[si]; }
__device__ __forceinline__ void copy_val(bf16_t* d, long long di, const bf16_t* s, long long si) { d[di] = s[si]; }

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
__device__ void win_attn(const T* __restrict__ qkv, long long s0, long long s1, long long s2, long long s3,
                         const float* __restrict__ biasT, const int* __restrict__ labels,
                         T* __restrict__ out, long long out_stride, int out_off, int Hc, int Wc,
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
    const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + (long long)(h * d) * s3;

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
    const float* bcol = biasT ? biasT + (long long)h * N * N + n : nullptr;
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
            if (bcol) dot += bcol[(long long)(k0 + c) * N];
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
    const long long ob = (((long long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
    #pragma unroll
    for (int j = 0; j < DMAX; j++)
        if (j < d) store_val(out, ob + j, acc[j] * inv);
}

#define ATTN_ARGS long long s0, long long s1, long long s2, long long s3, const float* biasT, const int* labels, long long out_stride, int out_off, int Hc, int Wc, int sy, int sx, int H, int W, int H_sp, int W_sp, int heads, int d, float scale
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

// GELU's erf: Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7, far below FP16 and
// BF16 rounding (GELU = 0.5x(1 + erf)); ~10% faster spatial gate than erff
__device__ __forceinline__ float fast_erff(float x) {
    const float ax = fabsf(x);
    const float t = __frcp_rn(1.0f + 0.3275911f * ax);
    const float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t;
    return copysignf(1.0f - p * __expf(-ax * ax), x);
}

__device__ __forceinline__ float load_val(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long long i, float v) {
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
// gelu_in: the statistics of GELU(x2), rounded to T like a separate GELU.
template <typename T>
__device__ void sg_stats(const T* __restrict__ x, int C, long long tokens, float eps,
                         float* __restrict__ stats, int gelu_in) {
    __shared__ float s1[256], s2[256];
    const int lane = threadIdx.x & 31;
    const long long tok = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    float a = 0.0f, q = 0.0f;
    if (tok < tokens) {
        const T* row = x + tok * (2L * C) + C;
        for (int c = lane; c < C; c += 32) {
            float v = load_val(row, c);
            if (gelu_in) {
                T r;
                store_val(&r, 0, 0.5f * v * (1.0f + fast_erff(v * 0.70710678118654752f)));
                v = load_val(&r, 0);
            }
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
    const long long L = (long long)H * W;
    const long long C2 = 2L * C;

    for (int i = threadIdx.x; i < (SY + 2) * (SX + 2) * CC; i += 256) {
        const int c = i % CC, p = i / CC;
        const int gy = y0 + p / (SX + 2) - 1, gx = x0 + p % (SX + 2) - 1;
        float v = 0.0f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W && c0 + c < C) {
            const long long tok = (long long)b * L + (long long)gy * W + gx;
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
        const long long tok = (long long)b * L + (long long)oy * W + ox;
        store_val(out, tok * C + c0 + c, acc * load_val(x, tok * C2 + c0 + c));
    }
}

extern "C" __global__ void sg_stats_f32(const float* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats<float>(x, C, tokens, eps, stats, gelu_in); }
extern "C" __global__ void sg_stats_f16(const _Float16* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats<_Float16>(x, C, tokens, eps, stats, gelu_in); }
extern "C" __global__ void sg_stats_bf16(const bf16_t* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats<bf16_t>(x, C, tokens, eps, stats, gelu_in); }
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
__device__ void dw_tokens(const T* __restrict__ x, long long row_stride, int ch_offset,
                          const float* __restrict__ w, const float* __restrict__ bias,
                          int gelu, T* __restrict__ out, int C, int H, int W) {
    __shared__ float tile[(SY + 2) * (SX + 2) * CC];
    const int chunks = (C + CC - 1) / CC;
    const int b = blockIdx.z / chunks;
    const int c0 = (blockIdx.z % chunks) * CC;
    const int x0 = blockIdx.x * SX, y0 = blockIdx.y * SY;
    const long long L = (long long)H * W;

    for (int i = threadIdx.x; i < (SY + 2) * (SX + 2) * CC; i += 256) {
        const int c = i % CC, p = i / CC;
        const int gy = y0 + p / (SX + 2) - 1, gx = x0 + p % (SX + 2) - 1;
        float v = 0.0f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W && c0 + c < C) {
            const long long tok = (long long)b * L + (long long)gy * W + gx;
            v = load_val(x, tok * row_stride + ch_offset + c0 + c);
            if (gelu & 2) {
                T r;
                store_val(&r, 0, 0.5f * v * (1.0f + fast_erff(v * 0.70710678118654752f)));
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
        if (gelu & 1) acc = 0.5f * acc * (1.0f + fast_erff(acc * 0.70710678118654752f));
        const long long tok = (long long)b * L + (long long)oy * W + ox;
        store_val(out, tok * C + c0 + c, acc);
    }
}

#define DWT_ARGS long long row_stride, int ch_offset, const float* w, const float* bias, int gelu
#define DWT_PASS row_stride, ch_offset, w, bias, gelu
extern "C" __global__ void dw_tokens_f32(const float* x, float* out, DWT_ARGS, int C, int H, int W) { dw_tokens<float>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void dw_tokens_f16(const _Float16* x, _Float16* out, DWT_ARGS, int C, int H, int W) { dw_tokens<_Float16>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void dw_tokens_bf16(const bf16_t* x, bf16_t* out, DWT_ARGS, int C, int H, int W) { dw_tokens<bf16_t>(x, DWT_PASS, out, C, H, W); }

// Same as dw_tokens for C, row_stride and ch_offset divisible by 4: every
// thread owns 4 channels of one column and walks down DWV_TY rows, keeping the
// partial sums of the next two output rows in registers. Each input row is
// read (and GELU'd) once: the left/right neighbours come from a double-buffered
// row in shared memory. A wave reads DWV_CV * 4 consecutive channels of a
// pixel at once.
#define DWV_CV 16
#define DWV_TX 16
#define DWV_TY 8
// at most ~136 VGPRs: 10 waves per SIMD instead of 7 at the 193 the compiler
// picks otherwise; the latency-bound row loop (ATT: ~45% waiting for loads,
// ~25% at the barrier) gains more from the waves than it loses to a
// small spill: spatial gate 0.42 -> 0.355 ms per 256px tile, dwconv same
#define DWV_WAVES __attribute__((amdgpu_waves_per_eu(10)))
__device__ __forceinline__ void load4(const float* p, long long i, float v[4]) {
    const float4 q = *(const float4*)(p + i);
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}
__device__ __forceinline__ void load4(const _Float16* p, long long i, float v[4]) {
    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
    const h4 q = *(const h4*)(p + i);
    v[0] = (float)q.x; v[1] = (float)q.y; v[2] = (float)q.z; v[3] = (float)q.w;
}
__device__ __forceinline__ void load4(const bf16_t* p, long long i, float v[4]) {
    const uint2 q = *(const uint2*)(p + i);
    v[0] = __uint_as_float(q.x << 16); v[1] = __uint_as_float(q.x & 0xFFFF0000u);
    v[2] = __uint_as_float(q.y << 16); v[3] = __uint_as_float(q.y & 0xFFFF0000u);
}
__device__ __forceinline__ void store4(float* p, long long i, const float v[4]) {
    *(float4*)(p + i) = make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void store4(_Float16* p, long long i, const float v[4]) {
    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
    h4 q; q.x = (_Float16)v[0]; q.y = (_Float16)v[1]; q.z = (_Float16)v[2]; q.w = (_Float16)v[3];
    *(h4*)(p + i) = q;
}
__device__ __forceinline__ unsigned int bf16_bits(float v) {
    unsigned int u = __float_as_uint(v);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ void store4(bf16_t* p, long long i, const float v[4]) {
    *(uint2*)(p + i) = make_uint2(bf16_bits(v[0]) | (bf16_bits(v[1]) << 16),
                                  bf16_bits(v[2]) | (bf16_bits(v[3]) << 16));
}

template <typename T>
__device__ __forceinline__ void dwv_load(const T* __restrict__ x, long long row_stride, long long base,
                                         int gy, int gx, int H, int W, float v[4]) {
    if (gy < 0 || gy >= H || gx < 0 || gx >= W) {
        v[0] = v[1] = v[2] = v[3] = 0.0f;
        return;
    }
    load4(x, ((long long)gy * W + gx) * row_stride + base, v);
}

// GELU on the conv input, rounded to T like a separate PyTorch GELU
template <typename T>
__device__ __forceinline__ float4 dwv_in(const float v[4], int gelu_in) {
    if (!gelu_in) return make_float4(v[0], v[1], v[2], v[3]);
    float g[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        T r;
        store_val(&r, 0, 0.5f * v[j] * (1.0f + fast_erff(v[j] * 0.70710678118654752f)));
        g[j] = load_val(&r, 0);
    }
    return make_float4(g[0], g[1], g[2], g[3]);
}

// sg_stats for C % 4 == 0 and 8-byte aligned rows: one wave32 per token, each
// lane loads 4 channels at a time (8 bytes for 16-bit types) and the sums are
// reduced with shuffles instead of shared memory and barriers.
template <typename T>
__device__ void sg_stats_v(const T* __restrict__ x, int C, long long tokens, float eps,
                           float* __restrict__ stats, int gelu_in) {
    const int lane = threadIdx.x & 31;
    const long long tok = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long long row = tok * (2LL * C) + C;
    float a = 0.0f, q = 0.0f;
    for (int c = lane * 4; c < C; c += 128) {
        float v[4];
        load4(x, row + c, v);
        const float4 g = dwv_in<T>(v, gelu_in);
        a += (g.x + g.y) + (g.z + g.w);
        q += (g.x * g.x + g.y * g.y) + (g.z * g.z + g.w * g.w);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_xor(a, o, 32);
        q += __shfl_xor(q, o, 32);
    }
    if (lane == 0) {
        const float mean = a / C;
        const float var = fmaxf(q / C - mean * mean, 0.0f);
        stats[tok * 2] = mean;
        stats[tok * 2 + 1] = rsqrtf(var + eps);
    }
}

// LayerNorm of a loaded value from its token's (mean, rstd); positions outside
// the image stay zero (the conv's zero padding comes after the norm)
__device__ __forceinline__ float4 dwv_norm(float4 v, const float* __restrict__ stats,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta, int c, long long tok,
                                           bool inside) {
    if (!stats || !inside) return v;
    const float mean = stats[tok * 2], rstd = stats[tok * 2 + 1];
    return make_float4((v.x - mean) * rstd * gamma[c] + beta[c],
                       (v.y - mean) * rstd * gamma[c + 1] + beta[c + 1],
                       (v.z - mean) * rstd * gamma[c + 2] + beta[c + 2],
                       (v.w - mean) * rstd * gamma[c + 3] + beta[c + 3]);
}

// With stats (and gamma, beta), the input is layer-normalized while loading;
// with gate_off >= 0, the output is multiplied by channels
// [gate_off, gate_off + C) of the same token (GELU'd too with gelu & 2): that
// is DAT's SpatialGate, x1 * dwconv(LayerNorm(x2)), on (B, L, 2C) rows.
template <typename T>
__device__ void dw_tokens_v(const T* __restrict__ x, long long row_stride, int ch_offset,
                            const float* __restrict__ w, const float* __restrict__ bias,
                            int gelu, T* __restrict__ out, int C, int H, int W,
                            const float* __restrict__ stats = nullptr,
                            const float* __restrict__ gamma = nullptr,
                            const float* __restrict__ beta = nullptr, int gate_off = -1,
                            int out_ld = 0) {
    __shared__ float4 row[2][DWV_TX + 2][DWV_CV];
    const int vecs = C / 4;
    // out rows are ld wide, the columns past C zeroed
    const int ld = out_ld > C ? out_ld : C;
    const int chunks = (ld / 4 + DWV_CV - 1) / DWV_CV;
    const int b = blockIdx.z / chunks;
    const int lv = threadIdx.x % DWV_CV, tx = threadIdx.x / DWV_CV;
    const int cv = (blockIdx.z % chunks) * DWV_CV + lv;
    const int x0 = blockIdx.x * DWV_TX;
    const int ox = x0 + tx;
    const int y0 = blockIdx.y * DWV_TY;
    // no early return: every thread takes part in the barriers
    const bool active = cv < vecs && ox < W;
    const int c = min(cv, vecs - 1) * 4;
    const long long L = (long long)H * W;
    const T* xb = x + (long long)b * L * row_stride;
    T* ob = out + (long long)b * L * ld;
    const long long base = ch_offset + c;
    const bool pad = cv >= vecs && cv * 4 < ld && ox < W;
    const int gin = gelu & 2;
    // the first and last column of the block also load the halo column
    const int hx = tx == 0 ? x0 - 1 : (tx == DWV_TX - 1 ? x0 + DWV_TX : -2);

    float k[4][9], bc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 9; i++) k[j][i] = w[(c + j) * 9 + i];
        bc[j] = bias ? bias[c + j] : 0.0f;
    }

    // p0: output row y - 1 (has rows y - 2, y - 1), p1: output row y (has y - 1)
    float p0[4], p1[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) p0[j] = p1[j] = bc[j];
    const int y_end = min(y0 + DWV_TY, H);
    const int mx = cv < vecs ? ox : -1;
    const int hxx = cv < vecs ? hx : -1;
    // the next row is loaded one iteration ahead
    float m[4], h[4];
    dwv_load(xb, row_stride, base, y0 - 1, mx, H, W, m);
    if (hx != -2) dwv_load(xb, row_stride, base, y0 - 1, hxx, H, W, h);
    const long long tok_row = (long long)b * L;
    for (int y = y0 - 1; y <= y_end; y++) {
        float4 (*buf)[DWV_CV] = row[y & 1];
        const bool y_in = y >= 0 && y < H;
        buf[tx + 1][lv] = dwv_norm(dwv_in<T>(m, gin), stats, gamma, beta, c,
                                   tok_row + (long long)y * W + mx, y_in && mx >= 0 && mx < W);
        if (hx != -2)
            buf[tx == 0 ? 0 : DWV_TX + 1][lv] =
                dwv_norm(dwv_in<T>(h, gin), stats, gamma, beta, c,
                         tok_row + (long long)y * W + hxx, y_in && hxx >= 0 && hxx < W);
        if (y < y_end) {
            dwv_load(xb, row_stride, base, y + 1, mx, H, W, m);
            if (hx != -2) dwv_load(xb, row_stride, base, y + 1, hxx, H, W, h);
        }
        __syncthreads();
        const float4 lq = buf[tx][lv], cq = buf[tx + 1][lv], rq = buf[tx + 2][lv];
        const float l[4] = {lq.x, lq.y, lq.z, lq.w}, ce[4] = {cq.x, cq.y, cq.z, cq.w},
                    r[4] = {rq.x, rq.y, rq.z, rq.w};
        float o[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float h0 = k[j][0] * l[j] + k[j][1] * ce[j] + k[j][2] * r[j];
            const float h1 = k[j][3] * l[j] + k[j][4] * ce[j] + k[j][5] * r[j];
            const float h2 = k[j][6] * l[j] + k[j][7] * ce[j] + k[j][8] * r[j];
            o[j] = p0[j] + h2;
            p0[j] = p1[j] + h1;
            p1[j] = bc[j] + h0;
        }
        if (active && y - 1 >= y0) {
            if (gelu & 1) {
                #pragma unroll
                for (int j = 0; j < 4; j++) o[j] = 0.5f * o[j] * (1.0f + fast_erff(o[j] * 0.70710678118654752f));
            }
            if (gate_off >= 0) {
                float g[4];
                load4(xb, ((long long)(y - 1) * W + ox) * row_stride + gate_off + c, g);
                const float4 gq = dwv_in<T>(g, gin);
                o[0] *= gq.x; o[1] *= gq.y; o[2] *= gq.z; o[3] *= gq.w;
            }
            store4(ob, ((long long)(y - 1) * W + ox) * ld + c, o);
        } else if (pad && y - 1 >= y0) {
            const float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            store4(ob, ((long long)(y - 1) * W + ox) * ld + cv * 4, z);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) DWV_WAVES dw_tokens_v_f32(const float* x, float* out, DWT_ARGS, int C, int H, int W) { dw_tokens_v<float>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void __launch_bounds__(256) DWV_WAVES dw_tokens_v_f16(const _Float16* x, _Float16* out, DWT_ARGS, int C, int H, int W) { dw_tokens_v<_Float16>(x, DWT_PASS, out, C, H, W); }
extern "C" __global__ void __launch_bounds__(256) DWV_WAVES dw_tokens_v_bf16(const bf16_t* x, bf16_t* out, DWT_ARGS, int C, int H, int W) { dw_tokens_v<bf16_t>(x, DWT_PASS, out, C, H, W); }
#define SGV_ARGS const float* stats, const float* gamma, const float* beta, int gate_off, int out_ld
#define SGV_PASS stats, gamma, beta, gate_off, out_ld
// The end of DAT's spatial interaction: out[tok] = w2 . GELU(h[tok]) + b2 for the
// (tokens, J) output h of its first 1x1 conv (a GEMM with N = J). PyTorch's
// GEMM with N = 1 for this took 0.12 ms per 256px tile, this ~0.01 ms.
template <typename T>
__device__ void spatial_map(const T* __restrict__ h, long long ld, int J,
                            const float* __restrict__ w2, float b2, T* __restrict__ out,
                            long long tokens) {
    const long long tok = (long long)blockIdx.x * 256 + threadIdx.x;
    if (tok >= tokens) return;
    const T* row = h + tok * ld;
    float r = b2;
    for (int j = 0; j < J; j++) {
        const float a = load_val(row, j);
        r += w2[j] * (0.5f * a * (1.0f + fast_erff(a * 0.70710678118654752f)));
    }
    store_val(out, tok, r);
}
extern "C" __global__ void spatial_map_f32(const float* h, long long ld, int J, const float* w2, float b2, float* out, long long tokens) { spatial_map<float>(h, ld, J, w2, b2, out, tokens); }
extern "C" __global__ void spatial_map_f16(const _Float16* h, long long ld, int J, const float* w2, float b2, _Float16* out, long long tokens) { spatial_map<_Float16>(h, ld, J, w2, b2, out, tokens); }
extern "C" __global__ void spatial_map_bf16(const bf16_t* h, long long ld, int J, const float* w2, float b2, bf16_t* out, long long tokens) { spatial_map<bf16_t>(h, ld, J, w2, b2, out, tokens); }
extern "C" __global__ void sg_stats_v_f32(const float* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats_v<float>(x, C, tokens, eps, stats, gelu_in); }
extern "C" __global__ void sg_stats_v_f16(const _Float16* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats_v<_Float16>(x, C, tokens, eps, stats, gelu_in); }
extern "C" __global__ void sg_stats_v_bf16(const bf16_t* x, int C, long long tokens, float eps, float* stats, int gelu_in) { sg_stats_v<bf16_t>(x, C, tokens, eps, stats, gelu_in); }
extern "C" __global__ void __launch_bounds__(256) DWV_WAVES sg_gate_v_f32(const float* x, float* out, DWT_ARGS, int C, int H, int W, SGV_ARGS) { dw_tokens_v<float>(x, DWT_PASS, out, C, H, W, SGV_PASS); }
extern "C" __global__ void __launch_bounds__(256) DWV_WAVES sg_gate_v_f16(const _Float16* x, _Float16* out, DWT_ARGS, int C, int H, int W, SGV_ARGS) { dw_tokens_v<_Float16>(x, DWT_PASS, out, C, H, W, SGV_PASS); }
extern "C" __global__ void __launch_bounds__(256) DWV_WAVES sg_gate_v_bf16(const bf16_t* x, bf16_t* out, DWT_ARGS, int C, int H, int W, SGV_ARGS) { dw_tokens_v<bf16_t>(x, DWT_PASS, out, C, H, W, SGV_PASS); }

// Statistics of a transposed ("channel") attention over the tokens, from a
// (B, N, 3C) qkv buffer with heads of d <= 32 channels: per head the Gram
// matrix q^T k (d x d) and the squared norms of every q and k channel, all in
// float32. One block per (token chunk, head, batch) writes the partial sums of
// its chunk; they are summed afterwards, so the result does not depend on the
// scheduling. Thread (i, 4 * j4 .. 4 * j4 + 3) accumulates gram[i][j].
#define GR_TT 32
template <typename T>
__device__ void chan_gram(const T* __restrict__ qkv, int C, int nh, long long N, long long chunk_len,
                          float* __restrict__ gram, float* __restrict__ sq) {
    __shared__ float qs[GR_TT][33], ks[GR_TT][33];
    const int d = C / nh;
    const int chunk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int i = threadIdx.x >> 3, j0 = (threadIdx.x & 7) * 4;
    const long long n_begin = chunk * chunk_len;
    const long long n_end = min(N, n_begin + chunk_len);
    const T* base = qkv + (long long)b * N * 3 * C + h * d;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sq_q = 0.0f, sq_k[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (long long n0 = n_begin; n0 < n_end; n0 += GR_TT) {
        for (int e = threadIdx.x; e < GR_TT * 32; e += 256) {
            const int t = e >> 5, ch = e & 31;
            float q = 0.0f, k = 0.0f;
            if (ch < d && n0 + t < n_end) {
                const long long r = (n0 + t) * 3 * C + ch;
                q = load_val(base, r);
                k = load_val(base, r + C);
            }
            qs[t][ch] = q;
            ks[t][ch] = k;
        }
        __syncthreads();
        #pragma unroll 8
        for (int t = 0; t < GR_TT; t++) {
            const float q = qs[t][i];
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) acc[jj] += q * ks[t][j0 + jj];
            if (j0 == 0) sq_q += q * q;
            if (i == 0) {
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) sq_k[jj] += ks[t][j0 + jj] * ks[t][j0 + jj];
            }
        }
        __syncthreads();
    }

    const long long part = (long long)chunk * gridDim.z + b;
    float* g = gram + (part * nh + h) * d * d;
    #pragma unroll
    for (int jj = 0; jj < 4; jj++)
        if (i < d && j0 + jj < d) g[i * d + j0 + jj] = acc[jj];
    float* s = sq + part * 2 * C + h * d;
    if (j0 == 0 && i < d) s[i] = sq_q;
    if (i == 0) {
        #pragma unroll
        for (int jj = 0; jj < 4; jj++)
            if (j0 + jj < d) s[C + j0 + jj] = sq_k[jj];
    }
}

#define GR_ARGS int C, int nh, long long N, long long chunk_len, float* gram, float* sq
#define GR_PASS C, nh, N, chunk_len, gram, sq
extern "C" __global__ void __launch_bounds__(256) chan_gram_f32(const float* qkv, GR_ARGS) { chan_gram<float>(qkv, GR_PASS); }
extern "C" __global__ void __launch_bounds__(256) chan_gram_f16(const _Float16* qkv, GR_ARGS) { chan_gram<_Float16>(qkv, GR_PASS); }
extern "C" __global__ void __launch_bounds__(256) chan_gram_bf16(const bf16_t* qkv, GR_ARGS) { chan_gram<bf16_t>(qkv, GR_PASS); }

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

// T: x and the sum; TA: the residual a; TO: the LayerNorm output. Mixed for an FP16
// model's FP32 trunk (fp32_last_group): x, sum in float, a and out in _Float16.
template <typename T, typename TA = T, typename TO = T>
__device__ void add_ln(const T* __restrict__ x, const TA* __restrict__ a, T* __restrict__ sum_out,
                       TO* __restrict__ out, const float* __restrict__ gamma,
                       const float* __restrict__ beta, int C, long long tokens, float eps,
                       int out_ld, int a_ld) {
    const int lane = threadIdx.x & 31;
    const long long tok = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long long row = tok * C;
    // the residual's rows may be wider (a [..., :C] view of a padded GEMM output)
    const long long arow = tok * (a_ld > 0 ? a_ld : C);
    float v[8];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        float t = 0.0f;
        if (c < C) {
            t = load_val(x, row + c);
            if (a) t = round_t(t + load_val(a, arow + c), x);
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
    // out rows are out_ld wide, the columns past C zeroed (input of a GEMM
    // with K padded to out_ld)
    const long long orow = tok * out_ld;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        if (c < C) {
            if (sum_out) store_val(sum_out, row + c, v[k]);
            store_val(out, orow + c, (v[k] - mean) * rstd * gamma[c] + beta[c]);
        } else if (c < out_ld) {
            store_val(out, orow + c, 0.0f);
        }
    }
}

// add_ln for C % 4 == 0 (and out_ld % 4 == 0): each lane holds 4 consecutive
// channels (8-byte loads), up to 2 * 128 = 256 channels per token.
template <typename T>
__device__ void add_ln_v(const T* __restrict__ x, const T* __restrict__ a, T* __restrict__ sum_out,
                         T* __restrict__ out, const float* __restrict__ gamma,
                         const float* __restrict__ beta, int C, long long tokens, float eps,
                         int out_ld, int a_ld) {
    const int lane = threadIdx.x & 31;
    const long long tok = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long long row = tok * C;
    // the residual's rows may be wider (a [..., :C] view of a padded GEMM output)
    const long long arow = tok * (a_ld > 0 ? a_ld : C);
    float v[2][4];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int c = 4 * lane + 128 * k;
        if (c < C) {
            load4(x, row + c, v[k]);
            if (a) {
                float r[4];
                load4(a, arow + c, r);
                #pragma unroll
                for (int j = 0; j < 4; j++) v[k][j] = round_t(v[k][j] + r[j], x);
            }
        } else {
            v[k][0] = v[k][1] = v[k][2] = v[k][3] = 0.0f;
        }
        s += (v[k][0] + v[k][1]) + (v[k][2] + v[k][3]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    const float mean = s / C;
    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        if (4 * lane + 128 * k < C) {
            #pragma unroll
            for (int j = 0; j < 4; j++) { const float dv = v[k][j] - mean; q += dv * dv; }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor(q, o, 32);
    const float rstd = rsqrtf(q / C + eps);
    const long long orow = tok * out_ld;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int c = 4 * lane + 128 * k;
        if (c < C) {
            if (sum_out) store4(sum_out, row + c, v[k]);
            float o[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) o[j] = (v[k][j] - mean) * rstd * gamma[c + j] + beta[c + j];
            store4(out, orow + c, o);
        } else if (c < out_ld) {
            const float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            store4(out, orow + c, z);
        }
    }
}

// add_ln after a gated mix of two branches a and c (the interaction of DAT-like
// blocks, e.g. FDAT's SimplifiedAIM), so the mix is never stored:
//   sw:    g = sigmoid(a . sw) per token,  mix = a + c * g   (spatial gate)
//   cgate: g = cgate[b][ch],               mix = c + a * g   (channel gate)
//   sum = x + mix, out = LayerNorm(sum)
// The gate and the mix are rounded to T like PyTorch's linear, sigmoid and
// addcmul. L: tokens per batch item.
template <typename T>
__device__ void aim_ln(const T* __restrict__ x, const T* __restrict__ a, const T* __restrict__ cb,
                       const float* __restrict__ sw, const float* __restrict__ cgate,
                       T* __restrict__ sum_out, T* __restrict__ out,
                       const float* __restrict__ gamma, const float* __restrict__ beta,
                       int C, long long tokens, long long L, float eps) {
    const int lane = threadIdx.x & 31;
    const long long tok = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= tokens) return;
    const long long row = tok * C;
    float av[8], cv[8];
    float dot = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        av[k] = cv[k] = 0.0f;
        if (c < C) {
            av[k] = load_val(a, row + c);
            cv[k] = load_val(cb, row + c);
            if (sw) dot += av[k] * sw[c];
        }
    }
    float g = 0.0f;
    if (sw) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o, 32);
        g = round_t(1.0f / (1.0f + expf(-round_t(dot, x))), x);
    }
    const float* cg = cgate ? cgate + (tok / L) * C : nullptr;
    float v[8];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int c = lane + 32 * k;
        float t = 0.0f;
        if (c < C) {
            const float mix = sw ? round_t(av[k] + cv[k] * g, x)
                                 : round_t(cv[k] + av[k] * cg[c], x);
            t = round_t(load_val(x, row + c) + mix, x);
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

#define LN_ARGS const float* gamma, const float* beta, int C, long long tokens, float eps
#define LN_PASS gamma, beta, C, tokens, eps
#define AIM_ARGS const float* sw, const float* cgate
#define AIM_PASS sw, cgate
extern "C" __global__ void aim_ln_f32(const float* x, const float* a, const float* c, AIM_ARGS, float* sum_out, float* out, LN_ARGS, long long L) { aim_ln<float>(x, a, c, AIM_PASS, sum_out, out, gamma, beta, C, tokens, L, eps); }
extern "C" __global__ void aim_ln_f16(const _Float16* x, const _Float16* a, const _Float16* c, AIM_ARGS, _Float16* sum_out, _Float16* out, LN_ARGS, long long L) { aim_ln<_Float16>(x, a, c, AIM_PASS, sum_out, out, gamma, beta, C, tokens, L, eps); }
extern "C" __global__ void aim_ln_bf16(const bf16_t* x, const bf16_t* a, const bf16_t* c, AIM_ARGS, bf16_t* sum_out, bf16_t* out, LN_ARGS, long long L) { aim_ln<bf16_t>(x, a, c, AIM_PASS, sum_out, out, gamma, beta, C, tokens, L, eps); }
extern "C" __global__ void add_ln_v_f32(const float* x, const float* a, float* sum_out, float* out, LN_ARGS, int out_ld, int a_ld) { add_ln_v<float>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_v_f16(const _Float16* x, const _Float16* a, _Float16* sum_out, _Float16* out, LN_ARGS, int out_ld, int a_ld) { add_ln_v<_Float16>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_v_bf16(const bf16_t* x, const bf16_t* a, bf16_t* sum_out, bf16_t* out, LN_ARGS, int out_ld, int a_ld) { add_ln_v<bf16_t>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_f32(const float* x, const float* a, float* sum_out, float* out, LN_ARGS, int out_ld, int a_ld) { add_ln<float>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_f16(const _Float16* x, const _Float16* a, _Float16* sum_out, _Float16* out, LN_ARGS, int out_ld, int a_ld) { add_ln<_Float16>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_bf16(const bf16_t* x, const bf16_t* a, bf16_t* sum_out, bf16_t* out, LN_ARGS, int out_ld, int a_ld) { add_ln<bf16_t>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }
extern "C" __global__ void add_ln_f32_f16(const float* x, const _Float16* a, float* sum_out, _Float16* out, LN_ARGS, int out_ld, int a_ld) { add_ln<float, _Float16, _Float16>(x, a, sum_out, out, LN_PASS, out_ld, a_ld); }

// HAT's HAB residual and second LayerNorm in one pass:
//   sum = (x + a) + (conv * ca[b, c]) * scale     (each op rounded to T, as in
//         `shortcut + attn_x + conv_x * conv_scale` with conv_x = conv * ca)
//   out = LayerNorm(sum)
// x and a are (B, L, C) token rows; conv is the CAB conv output in the image
// layout (B, C, L), so a block first stages 32 tokens of all channels in shared
// memory (coalesced along the tokens) and then works on token rows like add_ln.
// conv may be fp16 for a bf16 model (the convs run in fp16, see HAT/_rocm.py).
template <typename T, typename TC = T>
__device__ void hab_ln(const T* __restrict__ x, const T* __restrict__ a,
                       const TC* __restrict__ conv, const float* __restrict__ ca,
                       float scale, T* __restrict__ sum_out, T* __restrict__ out,
                       const float* __restrict__ gamma, const float* __restrict__ beta,
                       int C, long long L, long long tokens, float eps) {
    __shared__ float tile[256 * 33];
    const long long t0 = (long long)blockIdx.x * 32;
    for (int i = threadIdx.x; i < C * 32; i += 256) {
        const int c = i / 32, j = i % 32;
        const long long tok = t0 + j;
        float v = 0.0f;
        if (tok < tokens) {
            const long long b = tok / L, p = tok - b * L;
            v = load_val(conv, (b * C + c) * L + p);
        }
        tile[c * 33 + j] = v;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int j = wave; j < 32; j += 8) {
        const long long tok = t0 + j;
        if (tok >= tokens) break;
        const long long b = tok / L;
        const long long row = tok * C;
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

#define HAB_SIG(T, S) extern "C" __global__ void hab_ln_##S(const T* x, const T* a, const T* conv, const float* ca, float scale, T* sum_out, T* out, const float* gamma, const float* beta, int C, long long L, long long tokens, float eps) { hab_ln<T>(x, a, conv, ca, scale, sum_out, out, gamma, beta, C, L, tokens, eps); }
HAB_SIG(float, f32)
HAB_SIG(_Float16, f16)
HAB_SIG(bf16_t, bf16)
extern "C" __global__ void hab_ln_bf16_c16(const bf16_t* x, const bf16_t* a, const _Float16* conv, const float* ca, float scale, bf16_t* sum_out, bf16_t* out, const float* gamma, const float* beta, int C, long long L, long long tokens, float eps) { hab_ln<bf16_t, _Float16>(x, a, conv, ca, scale, sum_out, out, gamma, beta, C, L, tokens, eps); }

// LayerNorm over token rows like add_ln (no residual), also writing the result
// in the image layout (B, C, L) for a following convolution, which otherwise
// needs a slow transposing copy. A block normalizes 32 tokens, keeps them in
// shared memory and writes the image layout coalesced along the tokens. The
// image may be fp16 for a bf16 model: the rounded bf16 values convert exactly.
template <typename T, typename TI = T>
__device__ void ln_nchw(const T* __restrict__ x, T* __restrict__ out, TI* __restrict__ img,
                        const float* __restrict__ gamma, const float* __restrict__ beta,
                        int C, long long L, long long tokens, float eps) {
    __shared__ float tile[256 * 33];
    const long long t0 = (long long)blockIdx.x * 32;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int j = wave; j < 32; j += 8) {
        const long long tok = t0 + j;
        if (tok >= tokens) break;
        const long long row = tok * C;
        float v[8];
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = lane + 32 * k;
            const float t = c < C ? load_val(x, row + c) : 0.0f;
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
                // rounded here, so both stores below convert an exact value
                const float r = round_t((v[k] - mean) * rstd * gamma[c] + beta[c], x);
                store_val(out, row + c, r);
                tile[c * 33 + j] = r;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C * 32; i += 256) {
        const int c = i / 32, j = i % 32;
        const long long tok = t0 + j;
        if (tok >= tokens) continue;
        const long long b = tok / L, p = tok - b * L;
        store_val(img, (b * C + c) * L + p, tile[c * 33 + j]);
    }
}

#define LNI_SIG(T, S) extern "C" __global__ void ln_nchw_##S(const T* x, T* out, T* img, const float* gamma, const float* beta, int C, long long L, long long tokens, float eps) { ln_nchw<T>(x, out, img, gamma, beta, C, L, tokens, eps); }
LNI_SIG(float, f32)
LNI_SIG(_Float16, f16)
LNI_SIG(bf16_t, bf16)
extern "C" __global__ void ln_nchw_bf16_i16(const bf16_t* x, bf16_t* out, _Float16* img, const float* gamma, const float* beta, int C, long long L, long long tokens, float eps) { ln_nchw<bf16_t, _Float16>(x, out, img, gamma, beta, C, L, tokens, eps); }

// DAT's Adaptive Interaction gating, in one pass over (B, N, C):
//   out = a * sigmoid(tok[b, n]) + b * sigmoid(ch[b, c])   (a_uses_tok = 1)
//   out = a * sigmoid(ch[b, c]) + b * sigmoid(tok[b, n])   (a_uses_tok = 0)
// PyTorch runs this as four broadcast kernels with temporaries.
template <typename T>
__device__ void gate_mix(const T* __restrict__ a, const T* __restrict__ bt,
                         const T* __restrict__ tok, const T* __restrict__ ch,
                         T* __restrict__ out, int C, long long N, long long total, int a_uses_tok) {
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long long)gridDim.x * 256) {
        const long long t = i / C;
        const int c = (int)(i - t * C);
        const long long bi = t / N;
        const float ts = 1.0f / (1.0f + __expf(-load_val(tok, t)));
        const float cs = 1.0f / (1.0f + __expf(-load_val(ch, bi * C + c)));
        const float va = load_val(a, i), vb = load_val(bt, i);
        store_val(out, i, a_uses_tok ? va * ts + vb * cs : va * cs + vb * ts);
    }
}

// gate_mix for C % 4 == 0: 4 channels per thread with 8-byte loads.
// out rows are out_ld wide, the columns past C zeroed.
template <typename T>
__device__ void gate_mix_v(const T* __restrict__ a, const T* __restrict__ bt,
                           const T* __restrict__ tok, const T* __restrict__ ch,
                           T* __restrict__ out, int C, long long N, long long total, int a_uses_tok,
                           int out_ld) {
    const int vecs = out_ld / 4;
    for (long long q = (long long)blockIdx.x * 256 + threadIdx.x; q < total / C * vecs; q += (long long)gridDim.x * 256) {
        const long long t = q / vecs;
        const int c = (int)(q - t * vecs) * 4;
        if (c >= C) {
            const float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            store4(out, t * out_ld + c, z);
            continue;
        }
        const long long i = t * C + c;
        const float ts = 1.0f / (1.0f + __expf(-load_val(tok, t)));
        const T* chb = ch + (t / N) * C + c;
        float va[4], vb[4], o[4];
        load4(a, i, va);
        load4(bt, i, vb);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float cs = 1.0f / (1.0f + __expf(-load_val(chb, j)));
            o[j] = a_uses_tok ? va[j] * ts + vb[j] * cs : va[j] * cs + vb[j] * ts;
        }
        store4(out, t * out_ld + c, o);
    }
}

#define GM_ARGS int C, long long N, long long total, int a_uses_tok, int out_ld
#define GM_PASS C, N, total, a_uses_tok, out_ld
extern "C" __global__ void gate_mix_v_f32(const float* a, const float* b, const float* tok, const float* ch, float* out, GM_ARGS) { gate_mix_v<float>(a, b, tok, ch, out, GM_PASS); }
extern "C" __global__ void gate_mix_v_f16(const _Float16* a, const _Float16* b, const _Float16* tok, const _Float16* ch, _Float16* out, GM_ARGS) { gate_mix_v<_Float16>(a, b, tok, ch, out, GM_PASS); }
extern "C" __global__ void gate_mix_v_bf16(const bf16_t* a, const bf16_t* b, const bf16_t* tok, const bf16_t* ch, bf16_t* out, GM_ARGS) { gate_mix_v<bf16_t>(a, b, tok, ch, out, GM_PASS); }
extern "C" __global__ void gate_mix_f32(const float* a, const float* b, const float* tok, const float* ch, float* out, int C, long long N, long long total, int a_uses_tok) { gate_mix<float>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
extern "C" __global__ void gate_mix_f16(const _Float16* a, const _Float16* b, const _Float16* tok, const _Float16* ch, _Float16* out, int C, long long N, long long total, int a_uses_tok) { gate_mix<_Float16>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
extern "C" __global__ void gate_mix_bf16(const bf16_t* a, const bf16_t* b, const bf16_t* tok, const bf16_t* ch, bf16_t* out, int C, long long N, long long total, int a_uses_tok) { gate_mix<bf16_t>(a, b, tok, ch, out, C, N, total, a_uses_tok); }
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
__device__ void win_attn_wmma(const u16* __restrict__ qkv, long long s0, long long s1, long long s2, long long s3,
                              const float* __restrict__ biasT, const int* __restrict__ labels,
                              u16* __restrict__ out, long long out_stride, int out_off, int Hc, int Wc,
                              int sy, int sx, int H, int W, int H_sp, int W_sp,
                              int heads, int d, float scale) {
    constexpr int KS = KD / 16;  // k steps over the head dim, and O column tiles
    extern __shared__ unsigned char lds_raw[];
    const int N = H_sp * W_sp;
    // V^T rows are padded by 8 elements: the staging threads of a wave write
    // one token's channels, i.e. one column of V^T, and an unpadded row stride
    // of N * 2 bytes would put all of them in the same LDS bank
    const int NV = N + 8;
    u16* Ks = (u16*)lds_raw;           // [N][KD]
    u16* Vt = Ks + N * KD;             // [KD][NV]
    u16* Pb = Vt + KD * NV;            // [8 waves][2 tiles][16][16]
    int* labs = (int*)(Pb + 8 * 512);  // [N]

    const int nw = W / W_sp, nh = H / H_sp, nW = nh * nw;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int ih = wimg / nw, iw = wimg % nw;

    // stage K and V^T (zero padded to KD) and the region labels; consecutive
    // threads read consecutive channels of a token, so the loads coalesce
    for (int idx = threadIdx.x; idx < N * KD; idx += 256) {
        const int t = idx / KD, j = idx % KD;
        const int y = ih * H_sp + t / W_sp, x = iw * W_sp + t % W_sp;
        int ys = y + sy, xs = x + sx;
        if (ys >= H) ys -= H;
        if (xs >= W) xs -= W;
        const bool in = j < d && ys < Hc && xs < Wc;  // window padding is zero
        const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + (long long)(h * d + j) * s3;
        Ks[t * KD + j] = in ? qkv[s0 + base] : (u16)0;
        Vt[j * NV + t] = in ? qkv[2 * s0 + base] : (u16)0;
    }
    if (labels)
        for (int t = threadIdx.x; t < N; t += 256) labs[t] = labels[wimg * N + t];
    __syncthreads();

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    u16* P = Pb + wave * 512;

    for (int qt = wave; qt < N / 16; qt += 8) {
        // A operands for Q: row = query qt*16 + col, k = s*16 + 8*half + i
        v8s qa[KS];
        {
            const int n = qt * 16 + col;
            const int y = ih * H_sp + n / W_sp, x = iw * W_sp + n % W_sp;
            const int ys = (y + sy) % H, xs = (x + sx) % W;
            const bool valid = ys < Hc && xs < Wc;
            const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + (long long)(h * d) * s3;
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

        // two 16-key tiles per step: one row max reduction and one pair of
        // wave barriers for both; with an odd tile count the second tile of
        // the last step is masked out
        const int NT = N / 16;
        for (int kt = 0; kt < NT; kt += 2) {
            const bool two = kt + 1 < NT;
            float sv[2][8];
            #pragma unroll
            for (int u = 0; u < 2; u++) {
                if (u == 1 && !two) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) sv[u][i] = -__builtin_inff();
                    continue;
                }
                const int key = (kt + u) * 16 + col;
                v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int s = 0; s < KS; s++) {
                    const v8s kb = *(const v8s*)(Ks + key * KD + s * 16 + 8 * half);
                    sacc = mma<BF16>(qa[s], kb, sacc);
                }
                const int klab = labels ? labs[key] : 0;
                const float* brow = biasT ? biasT + ((long long)h * N + key) * N + q0 : nullptr;
                v4f b0 = {0, 0, 0, 0}, b1 = {0, 0, 0, 0};
                if (brow) { b0 = *(const v4f*)brow; b1 = *(const v4f*)(brow + 4); }
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                    if (labels && klab != qlab[i]) v -= 100.0f;
                    sv[u][i] = v;
                }
            }
            // row max across the 32 keys (16 lanes x 2 tiles) of this half wave
            float corr[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float mx = fmaxf(sv[0][i], sv[1][i]);
                mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
                mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
                const float mn = fmaxf(m[i], mx);
                corr[i] = __expf(m[i] - mn);
                const float p0 = __expf(sv[0][i] - mn), p1 = __expf(sv[1][i] - mn);
                // per-lane partial row sum, reduced across the 16 lanes once
                // at the end (the correction factors are the same for all)
                l[i] = l[i] * corr[i] + p0 + p1;
                m[i] = mn;
                // P[tile][row][key] for the A-layout re-read below
                P[(8 * half + i) * 16 + col] = BF16 ? f2bf(p0) : f2h(p0);
                P[256 + (8 * half + i) * 16 + col] = BF16 ? f2bf(p1) : f2h(p1);
            }
            #pragma unroll
            for (int t = 0; t < KS; t++)
                #pragma unroll
                for (int i = 0; i < 8; i++) o[t][i] *= corr[i];
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            #pragma unroll
            for (int u = 0; u < 2; u++) {
                if (u == 1 && !two) break;
                // A operand: row = query col, k = key 8*half + i
                const v8s pa = *(const v8s*)(P + u * 256 + col * 16 + 8 * half);
                #pragma unroll
                for (int t = 0; t < KS; t++) {
                    // B operand: col = head dim t*16 + col, k = key (kt+u)*16 + 8*half + i
                    const v8s vb =
                        *(const v8s*)(Vt + (t * 16 + col) * NV + (kt + u) * 16 + 8 * half);
                    o[t] = mma<BF16>(pa, vb, o[t]);
                }
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
            // reduce before the bounds check, with all 16 lanes active
            float ls = l[i];
            ls += __shfl_xor(ls, 1, 16);
            ls += __shfl_xor(ls, 2, 16);
            ls += __shfl_xor(ls, 4, 16);
            ls += __shfl_xor(ls, 8, 16);
            if (ys >= Hc || xs >= Wc) continue;
            const float inv = 1.0f / ls;
            const long long ob = (((long long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
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
__device__ void win_attn_wmmat(const u16* __restrict__ qkv, long long s0, long long s1, long long s2, long long s3,
                               const float* __restrict__ biasT, const int* __restrict__ labels,
                               u16* __restrict__ out, long long out_stride, int out_off, int Hc, int Wc,
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
                const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + h * d + j;
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
            const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + (long long)(h * d) * s3;
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
                    qkv + (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + h * d);
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
                const long long base = (long long)b * s1 + ((long long)ys * Wc + xs) * s2 + (long long)(h * d) * s3;
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
            bq[r] = biasT ? biasT + ((long long)h * N + qn[r]) * N : nullptr;
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
                const long long ob = (((long long)b * Hc + ys) * Wc + xs) * out_stride + out_off + h * d;
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

#define WMMA_ARGS long long s0, long long s1, long long s2, long long s3, const float* biasT, const int* labels, long long out_stride, int out_off, int Hc, int Wc, int sy, int sx, int H, int W, int H_sp, int W_sp, int heads, int d, float scale
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
__device__ void ocab_wmma(const u16* __restrict__ qkv, long long s0, long long s1, long long s2, long long s3,
                          const float* __restrict__ biasT, u16* __restrict__ out,
                          int H, int W, int ws, int ows, int heads, int d, float scale,
                          int chunk) {
    constexpr int KS = KD / 16;
    constexpr int MAXT = 2;  // query tiles per wave: ws*ws <= 8 * 16 * MAXT
    extern __shared__ unsigned char lds_raw[];
    const int NQ = ws * ws, NK = ows * ows, NKp = (NK + 15) / 16 * 16;
    const int pad = (ows - ws) / 2;
    const int CV = chunk + 8;     // padded V^T rows, see win_attn_wmma
    u16* Ks = (u16*)lds_raw;      // [chunk][KD]
    u16* Vt = Ks + chunk * KD;    // [KD][CV]
    u16* Pb = Vt + KD * CV;       // [8 waves][2 tiles][16][16]

    const int nwx = W / ws, nW = (H / ws) * nwx;
    const int h = blockIdx.x % heads;
    const int win = blockIdx.x / heads;
    const int b = win / nW, wimg = win % nW;
    const int iy = wimg / nwx, ix = wimg % nwx;
    const int ky0 = iy * ws - pad, kx0 = ix * ws - pad;

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    u16* P = Pb + wave * 512;
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
            const long long base = (long long)b * s1 + ((long long)y * W + x) * s2 + (long long)(h * d) * s3;
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
        // consecutive threads read consecutive channels of a key (coalesced)
        for (int idx = threadIdx.x; idx < nc * KD; idx += 256) {
            const int t = idx / KD, j = idx % KD;
            const int key = c0 + t;
            const int y = ky0 + key / ows, x = kx0 + key % ows;
            const bool in = j < d && key < NK && y >= 0 && y < H && x >= 0 && x < W;
            const long long base = in ? (long long)b * s1 + ((long long)y * W + x) * s2 + (long long)(h * d + j) * s3 : 0;
            Ks[t * KD + j] = in ? qkv[s0 + base] : (u16)0;
            Vt[j * CV + t] = in ? qkv[2 * s0 + base] : (u16)0;
        }
        __syncthreads();

        const int NT = nc / 16;
        #pragma unroll
        for (int tt = 0; tt < MAXT; tt++) {
            const int qt = wave + 8 * tt;
            if (qt >= ntiles) break;
            const int q0 = qt * 16 + 8 * half;
            // two 16-key tiles per step, as in win_attn_wmma
            for (int kt = 0; kt < NT; kt += 2) {
                const bool two = kt + 1 < NT;
                float sv[2][8];
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int kl = (kt + u) * 16 + col, key = c0 + kl;
                    if ((u == 1 && !two) || key >= NK) {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) sv[u][i] = -__builtin_inff();
                        if (u == 1 && !two) continue;
                    }
                    v8f sacc = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (int s = 0; s < KS; s++) {
                        const v8s kb = *(const v8s*)(Ks + kl * KD + s * 16 + 8 * half);
                        sacc = mma<BF16>(qa[tt][s], kb, sacc);
                    }
                    if (key < NK) {
                        const float* brow = biasT + ((long long)h * NK + key) * NQ + q0;
                        const v4f b0 = *(const v4f*)brow, b1 = *(const v4f*)(brow + 4);
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            sv[u][i] = sacc[i] * scale + (i < 4 ? b0[i] : b1[i - 4]);
                    }
                }
                float corr[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float mx = fmaxf(sv[0][i], sv[1][i]);
                    mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
                    mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
                    const float mn = fmaxf(m[tt][i], mx);
                    corr[i] = __expf(m[tt][i] - mn);
                    const float p0 = __expf(sv[0][i] - mn), p1 = __expf(sv[1][i] - mn);
                    l[tt][i] = l[tt][i] * corr[i] + p0 + p1;  // per-lane partial sum
                    m[tt][i] = mn;
                    P[(8 * half + i) * 16 + col] = BF16 ? f2bf(p0) : f2h(p0);
                    P[256 + (8 * half + i) * 16 + col] = BF16 ? f2bf(p1) : f2h(p1);
                }
                #pragma unroll
                for (int t = 0; t < KS; t++)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) o[tt][t][i] *= corr[i];
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    if (u == 1 && !two) break;
                    const v8s pa = *(const v8s*)(P + u * 256 + col * 16 + 8 * half);
                    #pragma unroll
                    for (int t = 0; t < KS; t++) {
                        const v8s vb =
                            *(const v8s*)(Vt + (t * 16 + col) * CV + (kt + u) * 16 + 8 * half);
                        o[tt][t] = mma<BF16>(pa, vb, o[tt][t]);
                    }
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
            float ls = l[tt][i];
            ls += __shfl_xor(ls, 1, 16);
            ls += __shfl_xor(ls, 2, 16);
            ls += __shfl_xor(ls, 4, 16);
            ls += __shfl_xor(ls, 8, 16);
            const float inv = 1.0f / ls;
            const long long ob = (((long long)b * H + y) * W + x) * (long long)(heads * d) + h * d;
            #pragma unroll
            for (int t = 0; t < KS; t++) {
                const int j = t * 16 + col;
                if (j < d) out[ob + j] = BF16 ? f2bf(o[tt][t][i] * inv) : f2h(o[tt][t][i] * inv);
            }
        }
    }
}

#define OCAB_ARGS long long s0, long long s1, long long s2, long long s3, const float* biasT, int H, int W, int ws, int ows, int heads, int d, float scale, int chunk
#define OCAB_PASS s0, s1, s2, s3, biasT, out, H, W, ws, ows, heads, d, scale, chunk
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_16(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 16>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_32(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 32>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_bf16_64(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<true, 64>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_16(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 16>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_32(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 32>(qkv, OCAB_PASS); }
extern "C" __global__ void __launch_bounds__(256) ocab_wmma_f16_64(const u16* qkv, u16* out, OCAB_ARGS) { ocab_wmma<false, 64>(qkv, OCAB_PASS); }
"""
)

_RRDB_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ unsigned short bf16_bits(float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ void store_val(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long long i, float v) { p[i] = bf16_bits(v); }

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
__device__ void act_into(const T* __restrict__ src, T* __restrict__ dst, long long n,
                         long long dst_stride, long long dst_off, float slope, int relu) {
    const long long b = blockIdx.y;
    const T* s = src + b * n;
    T* d = dst + b * dst_stride + dst_off;
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long long)gridDim.x * 256) {
        float v = load_val(s, i);
        store_val(d, i, v > 0.0f ? v : (relu ? 0.0f : v * slope));
    }
}

// dst = a * s1 + x1, then optionally * s2 + x2, rounding to T after every
// multiply and add like the separate PyTorch ops do.
template <typename T>
__device__ void res_add(const T* __restrict__ a, const T* __restrict__ x1, long long x1_stride,
                        const T* __restrict__ x2, long long x2_stride, T* __restrict__ dst,
                        long long dst_stride, long long n, float s1, float s2, int two) {
    const long long b = blockIdx.y;
    const T* pa = a + b * n;
    const T* p1 = x1 + b * x1_stride;
    const T* p2 = two ? x2 + b * x2_stride : nullptr;
    T* d = dst + b * dst_stride;
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long long)gridDim.x * 256) {
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
    extern "C" __global__ void act_into_##S(const T* src, T* dst, long long n, long long dst_stride,   \
                                            long long dst_off, float slope, int relu) {           \
        act_into<T>(src, dst, n, dst_stride, dst_off, slope, relu);                          \
    }                                                                                        \
    extern "C" __global__ void res_add_##S(const T* a, const T* x1, long long x1_stride,          \
                                           const T* x2, long long x2_stride, T* dst,              \
                                           long long dst_stride, long long n, float s1, float s2,      \
                                           int two) {                                        \
        res_add<T>(a, x1, x1_stride, x2, x2_stride, dst, dst_stride, n, s1, s2, two);        \
    }
RRDB_KERNELS(float, f32)
RRDB_KERNELS(_Float16, f16)
RRDB_KERNELS(bf16_t, bf16)
"""

_TW, _TH = 32, 16
_MAX_GRID_Z = 65535
_DTYPE_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
# Layout changes between DAT's token rows (B, L, C) and the (B, C, H, W) images of
# its residual-group convolutions, through 32x32 tiles in shared memory. PyTorch
# runs `.contiguous()` of such a transposed view with its strided copy kernel at
# ~50 GB/s (0.9 ms per 256px tile at C = 180, 1.3 ms with an FP16 -> FP32 cast,
# measured with rocprofv3 on gfx1201); the way back fuses the residual add.
_LAYOUT_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float load_val(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, long long i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, long long i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, long long i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, long long i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    p[i] = v != v ? (bf16_t)0x7FC0 : (bf16_t)(u >> 16);
}

// (B, L, C) tokens -> (B, C, L) image, cast to TO. Block 32x8, grid (C/32, L/32, B).
template <typename TI, typename TO>
__device__ void tok2img(const TI* __restrict__ x, TO* __restrict__ y, int C, long long L) {
    __shared__ float tile[32][33];
    const int c0 = blockIdx.x * 32, tx = threadIdx.x, ty = threadIdx.y;
    const long long l0 = (long long)blockIdx.y * 32;
    const long long b = blockIdx.z;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const long long l = l0 + ty + k;
        const int c = c0 + tx;
        tile[ty + k][tx] = (l < L && c < C) ? load_val(x, (b * L + l) * C + c) : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const int c = c0 + ty + k;
        const long long l = l0 + tx;
        if (c < C && l < L) store_val(y, (b * C + c) * L + l, tile[tx][ty + k]);
    }
}

// out (B, L, C) = res (B, L, C) + img (B, C, L), the sum in float rounded to TO.
template <typename TR, typename TI, typename TO>
__device__ void img2tok_add(const TR* __restrict__ res, const TI* __restrict__ img,
                            TO* __restrict__ out, int C, long long L) {
    __shared__ float tile[32][33];
    const int c0 = blockIdx.x * 32, tx = threadIdx.x, ty = threadIdx.y;
    const long long l0 = (long long)blockIdx.y * 32;
    const long long b = blockIdx.z;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const int c = c0 + ty + k;
        const long long l = l0 + tx;
        tile[ty + k][tx] = (c < C && l < L) ? load_val(img, (b * C + c) * L + l) : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        const long long l = l0 + ty + k;
        const int c = c0 + tx;
        if (l < L && c < C) {
            const long long i = (b * L + l) * C + c;
            store_val(out, i, load_val(res, i) + tile[tx][ty + k]);
        }
    }
}

#define T2I(NI, TI, NO, TO) extern "C" __global__ void __launch_bounds__(256) tok2img_##NI##_##NO(const TI* x, TO* y, int C, long long L) { tok2img<TI, TO>(x, y, C, L); }
#define I2T(NR, TR, NI, TI, NO, TO) extern "C" __global__ void __launch_bounds__(256) img2tok_add_##NR##_##NI##_##NO(const TR* r, const TI* x, TO* y, int C, long long L) { img2tok_add<TR, TI, TO>(r, x, y, C, L); }
T2I(f32, float, f32, float) T2I(f16, _Float16, f16, _Float16) T2I(bf16, bf16_t, bf16, bf16_t) T2I(f16, _Float16, f32, float)
I2T(f32, float, f32, float, f32, float) I2T(f16, _Float16, f16, _Float16, f16, _Float16)
I2T(bf16, bf16_t, bf16, bf16_t, bf16, bf16_t) I2T(f16, _Float16, f32, float, f32, float)
"""

_LAYOUT_T2I = [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("f16", "f32")]
_LAYOUT_I2T = [
    ("f32", "f32", "f32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f16", "f32", "f32"),
]

# name -> (source, kernel names, extra compiler options)
_PROGRAMS: dict[str, tuple[str, list[str], list[str]]] = {
    "dwconv": (_DWCONV_SRC, [f"dwconv3x3_{s}" for s in _DTYPE_SUFFIX.values()], []),
    "layout": (
        _LAYOUT_SRC,
        [f"tok2img_{i}_{o}" for i, o in _LAYOUT_T2I]
        + [f"img2tok_add_{r}_{i}_{o}" for r, i, o in _LAYOUT_I2T],
        [],
    ),
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
        [f"ocab_wmma_{s}_{kd}" for s in ("bf16", "f16") for kd in (16, 32, 64)],
        ["-std=c++17"],
    ),
    "spatial_gate": (
        _SG_SRC,
        [f"sg_{k}_{s}" for k in ("stats", "gate") for s in _DTYPE_SUFFIX.values()]
        + [f"dw_tokens_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"dw_tokens_v_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"sg_gate_v_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"sg_stats_v_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"spatial_map_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"chan_gram_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"add_ln_{s}" for s in _DTYPE_SUFFIX.values()]
        + ["add_ln_f32_f16"]
        + [f"add_ln_v_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"aim_ln_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"gate_mix_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"gate_mix_v_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"hab_ln_{s}" for s in _DTYPE_SUFFIX.values()]
        + [f"ln_nchw_{s}" for s in _DTYPE_SUFFIX.values()]
        + ["hab_ln_bf16_c16", "ln_nchw_bf16_i16"],
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
        bin_dir = self._find_bin_dir()
        os.add_dll_directory(str(bin_dir))
        self.hip = ctypes.WinDLL(str(next(bin_dir.glob("amdhip64*.dll"))))
        self.rtc = ctypes.WinDLL(str(next(bin_dir.glob("hiprtc0*.dll"))))
        self.funcs: dict[tuple[int, str], ctypes.c_void_p] = {}
        # function handle -> the device whose context it was loaded in
        self.fn_device: dict[int | None, int] = {}
        self.compiled: set[tuple[int, str]] = set()
        self.lock = threading.Lock()

    @staticmethod
    def _find_bin_dir() -> Path:
        import _rocm_sdk_core  # type: ignore

        return Path(_rocm_sdk_core.__file__).parent / "bin"

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
                self.fn_device[fn.value] = device_index
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
        # The launch goes to the current device: the default stream's handle is 0,
        # the null stream of whichever device is current, and fn belongs to the
        # device it was loaded on. Make that device current if it is not
        # (Rift73/spandrel 3e11fdd).
        device = self.fn_device.get(fn.value)
        if device is not None and device != torch.cuda.current_device():
            with torch.cuda.device(device):
                err = self.hip.hipModuleLaunchKernel(
                    fn, *grid, *block, shared_mem, stream, params, None
                )
        else:
            err = self.hip.hipModuleLaunchKernel(
                fn, *grid, *block, shared_mem, stream, params, None
            )
        self._check(err, "hipModuleLaunchKernel")


_hip: _Hip | None = None
_hip_lock = threading.Lock()


def _cache_dir() -> Path | None:
    """
    Where compiled kernels are kept between runs: SPANDREL_HIP_CACHE_DIR if set
    (an empty value disables the cache), otherwise %LOCALAPPDATA%/spandrel/hip_cache.
    """
    custom = os.environ.get("SPANDREL_HIP_CACHE_DIR")
    if custom is not None:
        return Path(custom) if custom.strip() else None
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "spandrel" / "hip_cache" if local else None


def hip_kernels_available() -> bool:
    """Only for AMD's ROCm builds of PyTorch on Windows, where HIPRTC ships with torch."""
    if os.environ.get("SPANDREL_HIP_KERNELS", "").strip() == "0":
        return False
    if not _platform_ok():
        return False
    # The kernels read raw pointers that tracing and dispatch modes cannot see:
    # ONNX export, torch.jit, torch.compile and any TorchDispatchMode (e.g.
    # FakeTensor) keep the original code (Rift73/spandrel 3e11fdd).
    return not (
        torch.jit.is_tracing()
        or torch.onnx.is_in_onnx_export()
        or torch.compiler.is_compiling()
        or _dispatch_mode_count() > 0
    )


@functools.cache
def _platform_ok() -> bool:
    return (
        sys.platform == "win32"
        and bool(torch.version.hip)
        and torch.cuda.is_available()
    )


# The number of active TorchDispatchModes; PyTorch has no public API for it.
_dispatch_mode_count = getattr(torch._C, "_len_torch_dispatch_stack", lambda: 0)


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
    # must match win_attn_wmma: K [N][KD], V^T [KD][N + 8], 8 x 2 P tiles, labels
    return n * kd * 2 + kd * (n + 8) * 2 + 8 * 512 * 2 + n * 4


def _wmmat_lds_bytes(n: int, kd: int) -> int:
    # must match win_attn_wmmat: K [N][KD], V^T [KD][N], labels
    return 2 * n * kd * 2 + n * 4


_bias_q_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _bias_query_major(bias_t: torch.Tensor) -> torch.Tensor:
    """(heads, key, query) -> contiguous (heads, query, key) for win_attn_wmmat,
    cached per tensor and version. From the linux-hip-hat branch."""
    # version_of: inference tensors (DAT's cached bias) have no version counter
    key = (bias_t.data_ptr(), version_of(bias_t), bias_t.shape[-1])
    hit = _bias_q_cache.get(key)
    if hit is None or hit[0] is not bias_t:
        if len(_bias_q_cache) >= 64:
            _bias_q_cache.pop(next(iter(_bias_q_cache)))
        hit = (bias_t, bias_t.transpose(1, 2).contiguous())
        _bias_q_cache[key] = hit
    return hit[1]


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


def _ocab_lds_bytes(chunk: int, kd: int) -> int:
    # must match ocab_wmma: K [chunk][KD], V^T [KD][chunk + 8], 8 x 2 P tiles
    return chunk * kd * 2 + kd * (chunk + 8) * 2 + 8 * 512 * 2


def _ocab_chunk(n_keys: int, kd: int) -> int:
    """Keys staged in shared memory at a time."""
    fit = (_LDS_LIMIT - 8 * 512 * 2 - kd * 8 * 2) // (2 * kd * 2) // 16 * 16
    return min(fit, (n_keys + 15) // 16 * 16)


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
    chunk = _ocab_chunk(ows * ows, kd)
    out = torch.empty((B, H, W, C), device=qkv.device, dtype=qkv.dtype)
    hip = _get_hip()
    try:
        fn = hip.get(
            qkv.device.index or 0,
            "ocab_wmma",
            f"ocab_wmma_{_DTYPE_SUFFIX[qkv.dtype]}_{kd}",
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
        shared_mem=_ocab_lds_bytes(chunk, kd),
    )
    return out


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
            # (DAT2 fp16 tile 256: 16.7 -> 12.6 ms per pass, measured on Linux)
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
            if bias_t is not None:
                bias_t = _bias_query_major(bias_t)
    else:
        name = f"win_attn_{_DTYPE_SUFFIX[qkv.dtype]}_{32 if d <= 32 else 64}"
        fn = hip.get(device, "win_attn", name)
        block, shared = n, _attn_lds_bytes(n, d, qkv.dtype)
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
_DWV_CV, _DWV_TX, _DWV_TY = 16, 16, 8
_dwv = os.environ.get("SPANDREL_DW_VEC", "").strip() != "0"


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
    gelu_in: bool = False,
    out_ld: int | None = None,
) -> torch.Tensor:
    """
    DAT's SpatialGate: x1 * dwconv3x3(LayerNorm(x2)), in the (B, H*W, 2C) layout.
    With gelu_in, of GELU(x) (the FFN's activation before it), without storing
    GELU(x).

    gamma/beta: (C,) float32. weight: (C, 9) float32. bias: (C,) float32 or None.
    Returns (B, H*W, C); with `out_ld` (> C, a multiple of 4), a [..., :C] view
    of a buffer with out_ld wide rows whose other columns are zero (see
    padded_linear).
    """
    B, L, C2 = x.shape  # noqa: N806
    C = C2 // 2  # noqa: N806
    hip = _get_hip()
    device = x.device.index or 0
    sfx = _DTYPE_SUFFIX[x.dtype]
    stream = _stream(x)
    vec = _dwv and C % 4 == 0 and x.data_ptr() % 16 == 0
    if gelu_in and not vec:
        x = torch.nn.functional.gelu(x)
        gelu_in = False
    tokens = B * L
    stats = torch.empty((tokens, 2), device=x.device, dtype=torch.float32)
    hip.launch(
        hip.get(device, "spatial_gate", f"sg_stats{'_v' if vec else ''}_{sfx}"),
        ((tokens + 7) // 8, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_void_p(stats.data_ptr()),
            ctypes.c_int(1 if gelu_in else 0),
        ],
        stream,
    )
    if vec:
        ld = out_ld if out_ld is not None and out_ld > C and out_ld % 4 == 0 else C
        full = torch.empty((B, L, ld), device=x.device, dtype=x.dtype)
        chunks = (ld // 4 + _DWV_CV - 1) // _DWV_CV
        hip.launch(
            hip.get(device, "spatial_gate", f"sg_gate_v_{sfx}"),
            (-(-W // _DWV_TX), -(-H // _DWV_TY), B * chunks),
            (256, 1, 1),
            [
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(full.data_ptr()),
                ctypes.c_int64(C2),
                ctypes.c_int(C),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
                ctypes.c_int(2 if gelu_in else 0),
                ctypes.c_int(C),
                ctypes.c_int(H),
                ctypes.c_int(W),
                ctypes.c_void_p(stats.data_ptr()),
                ctypes.c_void_p(gamma.data_ptr()),
                ctypes.c_void_p(beta.data_ptr()),
                ctypes.c_int(0),
                ctypes.c_int(ld),
            ],
            stream,
        )
        return full if ld == C else full[..., :C]
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
    suffix = _DTYPE_SUFFIX[x.dtype]
    if (
        _dwv
        and channels % 4 == 0
        and x.stride(1) % 4 == 0
        and ch_offset % 4 == 0
        and x.data_ptr() % 16 == 0
    ):
        chunks = (channels // 4 + _DWV_CV - 1) // _DWV_CV
        name = f"dw_tokens_v_{suffix}"
        grid = (-(-W // _DWV_TX), -(-H // _DWV_TY), B * chunks)
    else:
        chunks = (channels + _SG_CC - 1) // _SG_CC
        name = f"dw_tokens_{suffix}"
        grid = ((W + _SG_SX - 1) // _SG_SX, (H + _SG_SY - 1) // _SG_SY, B * chunks)
    hip.launch(
        hip.get(x.device.index or 0, "spatial_gate", name),
        grid,
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


def can_channel_gram(qkv: torch.Tensor, heads: int) -> bool:
    if qkv.dim() != 3 or not qkv.is_cuda or qkv.dtype not in _DTYPE_SUFFIX:
        return False
    C = qkv.shape[-1] // 3  # noqa: N806
    return (
        qkv.shape[-1] == 3 * C
        and C % heads == 0
        and 0 < C // heads <= 32
        and qkv.is_contiguous()
    )


def channel_gram(
    qkv: torch.Tensor, heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For a (B, N, 3C) qkv tensor of a transposed attention with `heads` heads:
    the per-head Gram matrices q^T k over the tokens, (B, heads, d, d), and the
    L2 norms over the tokens of every q and k channel, (B, C) each. float32.
    """
    B, N, C3 = qkv.shape  # noqa: N806
    C = C3 // 3  # noqa: N806
    d = C // heads
    # about 512 blocks, at least GR_TT tokens each
    chunks = max(1, min(-(-512 // (heads * B)), -(-N // 32)))
    chunk_len = math.ceil(math.ceil(N / chunks) / 32) * 32
    chunks = -(-N // chunk_len)
    gram = torch.empty((chunks, B, heads, d, d), device=qkv.device, dtype=torch.float32)
    sq = torch.empty((chunks, B, 2, C), device=qkv.device, dtype=torch.float32)
    hip = _get_hip()
    hip.launch(
        hip.get(
            qkv.device.index or 0,
            "spatial_gate",
            f"chan_gram_{_DTYPE_SUFFIX[qkv.dtype]}",
        ),
        (chunks, heads, B),
        (256, 1, 1),
        [
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int(heads),
            ctypes.c_int64(N),
            ctypes.c_int64(chunk_len),
            ctypes.c_void_p(gram.data_ptr()),
            ctypes.c_void_p(sq.data_ptr()),
        ],
        _stream(qkv),
    )
    norms = sq.sum(0).sqrt_()
    return gram.sum(0), norms[:, 0], norms[:, 1]


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
    out_ld: int | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    LayerNorm over the last dim of a contiguous tensor. With `residual`, first
    computes x + residual and returns (that sum, its LayerNorm); otherwise
    (None, LayerNorm(x)). gamma/beta: float32 contiguous.

    With `out_ld` (> C, <= 256), the LayerNorm is a [..., :C] view of a buffer
    with out_ld wide rows whose other columns are zero (see padded_linear).

    `out_dtype` float16 with a float32 x (the FP32 trunk of an FP16 model, see
    fp32_last_group): the sum stays float32, the residual and the LayerNorm
    output are float16.
    """
    C = x.shape[-1]  # noqa: N806
    tokens = x.numel() // C
    out_dtype = out_dtype or x.dtype
    mixed = out_dtype != x.dtype
    if mixed and not (
        x.dtype == torch.float32
        and out_dtype == torch.float16
        and (residual is None or residual.dtype == torch.float16)
    ):
        raise ValueError(f"no LayerNorm kernel from {x.dtype} to {out_dtype}")
    ld = out_ld if out_ld is not None and C < out_ld <= 256 else C
    full = torch.empty((*x.shape[:-1], ld), device=x.device, dtype=out_dtype)
    out = full if ld == C else full[..., :C]
    sum_out = torch.empty_like(x) if residual is not None else None
    a_ld = C
    if residual is not None:
        # a [..., :C] view of rows a_ld wide (padded_linear with pad_out) is
        # read in place; anything else is made contiguous
        if _row_view_ld(residual, C) is not None:
            a_ld = _row_view_ld(residual, C)
        else:
            residual = residual.contiguous()
    vec = (
        _dwv
        and not mixed
        and C % 4 == 0
        and ld % 4 == 0
        and a_ld % 4 == 0
        and all(
            t is None or t.data_ptr() % 16 == 0 for t in (x, residual, sum_out, full)
        )
    )
    name = (
        "add_ln_f32_f16"
        if mixed
        else f"add_ln{'_v' if vec else ''}_{_DTYPE_SUFFIX[x.dtype]}"
    )
    hip = _get_hip()
    hip.launch(
        hip.get(x.device.index or 0, "spatial_gate", name),
        ((tokens + 7) // 8, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(residual.data_ptr() if residual is not None else 0),
            ctypes.c_void_p(sum_out.data_ptr() if sum_out is not None else 0),
            ctypes.c_void_p(full.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_int(ld),
            ctypes.c_int(a_ld),
        ],
        _stream(x),
    )
    return sum_out, out


def _row_view_ld(t: torch.Tensor, c: int) -> int | None:
    """The row width of t if t is t_full[..., :c] of a contiguous t_full with
    rows wider than c, else None (t contiguous or another layout)."""
    if t.shape[-1] != c or t.stride(-1) != 1 or t.dim() < 2:
        return None
    ld = t.stride(-2)
    if ld <= c:
        return None
    expect = ld
    for size, stride in zip(reversed(t.shape[:-1]), reversed(t.stride()[:-1])):
        if size != 1 and stride != expect:
            return None
        expect *= size
    return ld


def padded_width(k: int) -> int:
    """
    The width to pad a GEMM's K to: a multiple of 32. hipBLASLt's kernels for
    DAT's K = 180 are up to 1.7x slower than for K = 192 on gfx1201.
    """
    return -(-k // 32) * 32


def padded_linear(
    lin: torch.nn.Linear, x: torch.Tensor, pad_out: bool = False
) -> torch.Tensor:
    """
    lin(x) for x = buf[..., :K] of a buffer with wider rows whose extra columns
    are zero (from layer_norm, gate_mix or spatial_gate with out_ld): the GEMM
    runs with K = the row width and the weight zero padded to it (cached).

    With pad_out, the output width N is padded to padded_width(N) too (zero
    weight rows and bias) and a [..., :N] view is returned: hipBLASLt runs DAT's
    N = 180 GEMMs ~25% slower than N = 192 on gfx1201. layer_norm reads such a
    view as its residual in place. Same values as lin(x), for any x.
    """
    K = x.shape[-1]  # noqa: N806
    ld = x.stride(-2) if x.dim() >= 2 else K
    k_pad = not (ld == K or x.stride(-1) != 1)
    w, bias = lin.weight, lin.bias
    N = w.shape[0]  # noqa: N806
    n_full = padded_width(N) if pad_out else N
    if not k_pad and n_full == N:
        return lin(x)
    inp = x.as_strided((*x.shape[:-1], ld), x.stride()) if k_pad else x
    kw = ld if k_pad else K
    # version_of: weights cast inside inference mode have no version counter
    key = (
        w.data_ptr(),
        version_of(w),
        w.dtype,
        kw,
        n_full,
        None if bias is None else (bias.data_ptr(), version_of(bias)),
    )
    cached = lin.__dict__.get("_spandrel_padded_w")
    if cached is None or cached[0] != key:
        wp = torch.zeros((n_full, kw), device=w.device, dtype=w.dtype)
        wp[:N, :K] = w.detach()
        bp = bias
        if bias is not None and n_full != N:
            bp = torch.zeros((n_full,), device=bias.device, dtype=bias.dtype)
            bp[:N] = bias.detach()
        cached = (key, wp, bp)
        lin.__dict__["_spandrel_padded_w"] = cached
    # hipBLASLt with a timed algorithm (hipblaslt.py), else PyTorch's choice
    out = hipblaslt.linear(inp, cached[1], cached[2])
    if out is None:
        out = torch.nn.functional.linear(inp, cached[1], cached[2])
    return out if n_full == N else out[..., :N]


def gated_mix_layer_norm(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    spatial_weight: torch.Tensor | None = None,
    channel_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x + mix and LayerNorm(x + mix) of contiguous (B, L, C) tensors, where
    mix = a + c * sigmoid(a @ spatial_weight)  (spatial_weight: (C,) float32)
    or  mix = c + a * channel_gate             (channel_gate: (B, C) float32).
    """
    if (spatial_weight is None) == (channel_gate is None):
        raise ValueError("pass exactly one of spatial_weight and channel_gate")
    B, L, C = x.shape  # noqa: N806
    tokens = B * L
    a = a.contiguous()
    c = c.contiguous()
    out = torch.empty_like(x)
    sum_out = torch.empty_like(x)
    hip = _get_hip()
    hip.launch(
        hip.get(
            x.device.index or 0, "spatial_gate", f"aim_ln_{_DTYPE_SUFFIX[x.dtype]}"
        ),
        ((tokens + 7) // 8, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_void_p(
                spatial_weight.data_ptr() if spatial_weight is not None else 0
            ),
            ctypes.c_void_p(channel_gate.data_ptr() if channel_gate is not None else 0),
            ctypes.c_void_p(sum_out.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
            ctypes.c_int64(L),
        ],
        _stream(x),
    )
    return sum_out, out


def layer_norm_with_image(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    image_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LayerNorm of contiguous (B, H*W, C) tokens, returned both as tokens and as
    a contiguous (B, C, H, W) image, bit-identical to permuting the tokens.
    image_fp16 (bf16 x only): the image is fp16, holding the same values.
    """
    B, L, C = x.shape  # noqa: N806
    if C > 256 or L != H * W:
        raise ValueError("unsupported shape")
    tokens = B * L
    out = torch.empty_like(x)
    name = f"ln_nchw_{_DTYPE_SUFFIX[x.dtype]}"
    img_dtype = x.dtype
    if image_fp16:
        if x.dtype != torch.bfloat16:
            raise ValueError("image_fp16 needs a bf16 input")
        name, img_dtype = "ln_nchw_bf16_i16", torch.float16
    img = torch.empty((B, C, H, W), device=x.device, dtype=img_dtype)
    hip = _get_hip()
    hip.launch(
        hip.get(x.device.index or 0, "spatial_gate", name),
        ((tokens + 31) // 32, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(img.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(beta.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(L),
            ctypes.c_int64(tokens),
            ctypes.c_float(eps),
        ],
        _stream(x),
    )
    return out, img


def hab_layer_norm(
    x: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    ca: torch.Tensor,
    scale: float,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HAT's `(x + a) + conv * ca * scale` and its LayerNorm in one pass: returns
    (sum, LayerNorm(sum)). x, a: contiguous (B, L, C); conv: contiguous
    (B, C, H, W) with H*W = L, of x's dtype or fp16 for a bf16 x; ca: (B, C)
    float32; gamma/beta: float32.
    """
    B, L, C = x.shape  # noqa: N806
    if C > 256:
        raise ValueError("too many channels")
    tokens = B * L
    a, conv, ca = a.contiguous(), conv.contiguous(), ca.contiguous()
    name = f"hab_ln_{_DTYPE_SUFFIX[x.dtype]}"
    if conv.dtype != x.dtype:
        if (x.dtype, conv.dtype) != (torch.bfloat16, torch.float16):
            raise ValueError("unsupported conv dtype")
        name = "hab_ln_bf16_c16"
    sum_out, out = torch.empty_like(x), torch.empty_like(x)
    hip = _get_hip()
    hip.launch(
        hip.get(x.device.index or 0, "spatial_gate", name),
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
        ],
        _stream(x),
    )
    return sum_out, out


# --- adaptive interaction gating ------------------------------------------------


def gate_mix(
    a: torch.Tensor,
    b: torch.Tensor,
    tok: torch.Tensor,
    ch: torch.Tensor,
    a_uses_tok: bool,
    out_ld: int | None = None,
) -> torch.Tensor:
    """
    a * sigmoid(tok) + b * sigmoid(ch) (or with tok/ch swapped between a and b)
    for (B, N, C) tensors a and b, a per-token map tok (B, N, 1) and a
    per-channel map ch (B, 1, C), in one pass. With `out_ld` (> C, a multiple
    of 4), a [..., :C] view of a buffer with out_ld wide rows whose other
    columns are zero (see padded_linear).
    """
    B, N, C = a.shape  # noqa: N806
    a, b = a.contiguous(), b.contiguous()
    tok = tok.reshape(B, N).to(a.dtype).contiguous()
    ch = ch.reshape(B, C).to(a.dtype).contiguous()
    total = a.numel()
    vec = _dwv and C % 4 == 0 and all(t.data_ptr() % 16 == 0 for t in (a, b))
    ld = out_ld if vec and out_ld is not None and out_ld > C and out_ld % 4 == 0 else C
    full = torch.empty((B, N, ld), device=a.device, dtype=a.dtype)
    out = full if ld == C else full[..., :C]
    name = f"gate_mix{'_v' if vec else ''}_{_DTYPE_SUFFIX[a.dtype]}"
    threads = total // C * (ld // 4) if vec else total
    args = [
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(tok.data_ptr()),
        ctypes.c_void_p(ch.data_ptr()),
        ctypes.c_void_p(full.data_ptr()),
        ctypes.c_int(C),
        ctypes.c_int64(N),
        ctypes.c_int64(total),
        ctypes.c_int(1 if a_uses_tok else 0),
    ]
    if vec:
        args.append(ctypes.c_int(ld))
    hip = _get_hip()
    hip.launch(
        hip.get(a.device.index or 0, "spatial_gate", name),
        (min((threads + 255) // 256, 65535), 1, 1),
        (256, 1, 1),
        args,
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


# --- token rows <-> images ----------------------------------------------------


def tokens_to_image(
    x: torch.Tensor,
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """
    rearrange(x, "b (h w) c -> b c h w").contiguous().to(dtype) for contiguous
    token rows x (B, H*W, C), or None if no kernel covers the dtypes.
    """
    dtype = dtype or x.dtype
    if not x.is_cuda or not x.is_contiguous() or x.dim() != 3 or x.shape[1] != H * W:
        return None
    key = (_DTYPE_SUFFIX.get(x.dtype), _DTYPE_SUFFIX.get(dtype))
    if key not in _LAYOUT_T2I:
        return None
    B, L, C = x.shape  # noqa: N806
    y = torch.empty((B, C, H, W), device=x.device, dtype=dtype)
    hip = _get_hip()
    hip.launch(
        hip.get(x.device.index or 0, "layout", f"tok2img_{key[0]}_{key[1]}"),
        (-(-C // 32), -(-L // 32), B),
        (32, 8, 1),
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(L),
        ],
        _stream(x),
    )
    return y


def image_add_tokens(res: torch.Tensor, img: torch.Tensor) -> torch.Tensor | None:
    """
    res + rearrange(img, "b c h w -> b (h w) c") as contiguous (B, H*W, C) rows,
    in PyTorch's promoted dtype, for contiguous res (B, L, C) and img
    (B, C, H, W); None if no kernel covers the dtypes.
    """
    if not (res.is_cuda and res.is_contiguous() and img.is_contiguous()):
        return None
    if res.dim() != 3 or img.dim() != 4:
        return None
    B, L, C = res.shape  # noqa: N806
    if img.shape[0] != B or img.shape[1] != C or img.shape[2] * img.shape[3] != L:
        return None
    out_dtype = torch.promote_types(res.dtype, img.dtype)
    key = tuple(_DTYPE_SUFFIX.get(t) for t in (res.dtype, img.dtype, out_dtype))
    if key not in _LAYOUT_I2T:
        return None
    out = torch.empty((B, L, C), device=res.device, dtype=out_dtype)
    hip = _get_hip()
    hip.launch(
        hip.get(res.device.index or 0, "layout", "img2tok_add_{}_{}_{}".format(*key)),
        (-(-C // 32), -(-L // 32), B),
        (32, 8, 1),
        [
            ctypes.c_void_p(res.data_ptr()),
            ctypes.c_void_p(img.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(C),
            ctypes.c_int64(L),
        ],
        _stream(res),
    )
    return out



def can_spatial_map(t: torch.Tensor, hidden: int) -> bool:
    return t.is_cuda and t.dtype in _DTYPE_SUFFIX and t.dim() == 3 and hidden <= 64


def spatial_map(
    t: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: float,
) -> torch.Tensor:
    """
    w2 . GELU(t @ w1^T + b1) + b2 per token of t (B, L, C), as (B, L, 1) in t's
    dtype. The first product is a GEMM (w1: (J, C), b1: (J,), in t's dtype);
    GELU and the product with w2 ((J,) float32) are one kernel, summed in
    float32.
    """
    h = torch.nn.functional.linear(t, w1, b1).contiguous()  # B, L, J
    B, L, J = h.shape  # noqa: N806
    tokens = B * L
    out = torch.empty((B, L, 1), device=t.device, dtype=t.dtype)
    hip = _get_hip()
    name = f"spatial_map_{_DTYPE_SUFFIX[t.dtype]}"
    hip.launch(
        hip.get(t.device.index or 0, "spatial_gate", name),
        ((tokens + 255) // 256, 1, 1),
        (256, 1, 1),
        [
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_int64(J),  # row stride of h
            ctypes.c_int(J),
            ctypes.c_void_p(w2.data_ptr()),
            ctypes.c_float(b2),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(tokens),
        ],
        _stream(t),
    )
    return out
