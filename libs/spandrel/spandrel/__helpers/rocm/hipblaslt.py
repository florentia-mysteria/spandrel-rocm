"""
hipBLASLt called directly for the GEMMs of padded_linear, with the algorithm
picked by timing.

PyTorch takes hipBLASLt's first heuristic candidate, which for DAT's block
GEMMs (65536 tokens, K = 192, N = 540 / 360 / 192, FP16 with bias) is 10-25%
slower than the best of its candidates on gfx1201 (qkv 0.29 vs 0.22 ms), and
TunableOp is not an option (its tuning crashes on some bf16 batched
candidates). Here each new GEMM shape times up to _CANDIDATES of hipBLASLt's
own candidates once per process (~50 ms), checks the winner against
F.linear, and later calls go straight to hipblasLtMatmul with it. Any failure
switches this off for the process and padded_linear falls back to F.linear.

SPANDREL_HIPBLASLT=0 turns it off.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path

import torch

logger = logging.getLogger("spandrel.rocm")

_CANDIDATES = 24
_WORKSPACE = 32 << 20
# hipblaslt.h / hipblas-common.h / library_types.h
_R16F, _R32F, _R16BF = 2, 0, 14
_COMPUTE_32F = 2
_OP_N, _OP_T = 111, 112
_DESC_TRANSA, _DESC_TRANSB, _DESC_EPILOGUE, _DESC_BIAS_PTR, _DESC_BIAS_TYPE = range(5)
_EPI_DEFAULT, _EPI_BIAS = 1, 4
_PREF_MAX_WORKSPACE = 1
_DTYPES = {torch.float16: _R16F, torch.bfloat16: _R16BF}

_vp = ctypes.c_void_p


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]  # noqa: RUF012


class _Heuristic(ctypes.Structure):
    _fields_ = [  # noqa: RUF012
        ("algo", _Algo),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


class _Plan:
    """Descriptors and the chosen algorithm of one GEMM shape."""

    def __init__(self, desc: _vp, a: _vp, b: _vp, d: _vp, algo: _Algo) -> None:
        self.desc, self.a, self.b, self.d, self.algo = desc, a, b, d, algo


class _Lt:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL(str(_library_path()))
        self.handle = _vp()
        self._chk(self.lib.hipblasLtCreate(ctypes.byref(self.handle)), "hipblasLtCreate")
        self.plans: dict[tuple[object, ...], _Plan] = {}
        self.workspace: dict[int, torch.Tensor] = {}
        self.alpha, self.beta = ctypes.c_float(1.0), ctypes.c_float(0.0)

    @staticmethod
    def _chk(status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed: hipblasStatus {status}")

    def _set(self, desc: _vp, attr: int, value: ctypes._SimpleCData) -> None:
        self._chk(
            self.lib.hipblasLtMatmulDescSetAttribute(
                desc, attr, ctypes.byref(value), ctypes.sizeof(value)
            ),
            "hipblasLtMatmulDescSetAttribute",
        )

    def _layout(self, dtype: int, rows: int, cols: int, ld: int) -> _vp:
        layout = _vp()
        self._chk(
            self.lib.hipblasLtMatrixLayoutCreate(
                ctypes.byref(layout),
                dtype,
                ctypes.c_uint64(rows),
                ctypes.c_uint64(cols),
                ctypes.c_int64(ld),
            ),
            "hipblasLtMatrixLayoutCreate",
        )
        return layout

    def _ws(self, device: int) -> torch.Tensor:
        ws = self.workspace.get(device)
        if ws is None:
            ws = torch.empty(_WORKSPACE, dtype=torch.uint8, device=f"cuda:{device}")
            self.workspace[device] = ws
        return ws

    def matmul(
        self, plan: _Plan, w: torch.Tensor, x: torch.Tensor, bias, out: torch.Tensor
    ) -> None:
        if bias is not None:
            self._set(plan.desc, _DESC_BIAS_PTR, _vp(bias.data_ptr()))
        ws = self._ws(x.device.index or 0)
        self._chk(
            self.lib.hipblasLtMatmul(
                self.handle,
                plan.desc,
                ctypes.byref(self.alpha),
                _vp(w.data_ptr()),
                plan.a,
                _vp(x.data_ptr()),
                plan.b,
                ctypes.byref(self.beta),
                _vp(out.data_ptr()),
                plan.d,
                _vp(out.data_ptr()),
                plan.d,
                ctypes.byref(plan.algo),
                _vp(ws.data_ptr()),
                ctypes.c_size_t(_WORKSPACE),
                _vp(torch.cuda.current_stream(x.device).cuda_stream),
            ),
            "hipblasLtMatmul",
        )

    def plan(self, x2: torch.Tensor, w: torch.Tensor, bias) -> _Plan:
        """The plan for out (M, N) = x2 (M, K, row stride ldb) @ w^T + bias."""
        m, k = x2.shape
        n = w.shape[0]
        ldb = x2.stride(0)
        key = (m, n, k, ldb, x2.dtype, bias is not None, x2.device)
        p = self.plans.get(key)
        if p is not None:
            return p
        dt = _DTYPES[x2.dtype]
        desc = _vp()
        self._chk(
            self.lib.hipblasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _R32F),
            "hipblasLtMatmulDescCreate",
        )
        # column-major: D (N x M) = W^T-op(A: K x N) * B (K x M) [+ bias per row]
        self._set(desc, _DESC_TRANSA, ctypes.c_int32(_OP_T))
        self._set(desc, _DESC_TRANSB, ctypes.c_int32(_OP_N))
        self._set(desc, _DESC_EPILOGUE, ctypes.c_uint32(_EPI_BIAS if bias is not None else _EPI_DEFAULT))
        if bias is not None:
            self._set(desc, _DESC_BIAS_PTR, _vp(bias.data_ptr()))
            self._set(desc, _DESC_BIAS_TYPE, ctypes.c_int32(dt))
        a = self._layout(dt, k, n, w.stride(0))
        b = self._layout(dt, k, m, ldb)
        d = self._layout(dt, n, m, n)
        pref = _vp()
        self._chk(self.lib.hipblasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        wsb = ctypes.c_uint64(_WORKSPACE)
        self._chk(
            self.lib.hipblasLtMatmulPreferenceSetAttribute(
                pref, _PREF_MAX_WORKSPACE, ctypes.byref(wsb), 8
            ),
            "pref workspace",
        )
        heur = (_Heuristic * _CANDIDATES)()
        count = ctypes.c_int()
        self._chk(
            self.lib.hipblasLtMatmulAlgoGetHeuristic(
                self.handle, desc, a, b, d, d, pref, _CANDIDATES, heur, ctypes.byref(count)
            ),
            "hipblasLtMatmulAlgoGetHeuristic",
        )
        self.lib.hipblasLtMatmulPreferenceDestroy(pref)
        cands = [heur[i].algo for i in range(count.value) if heur[i].state == 0]
        if not cands:
            raise RuntimeError("hipBLASLt returned no algorithm")

        # time the candidates (wall clock: HIP events are unreliable on Windows)
        out = torch.empty((m, n), device=x2.device, dtype=x2.dtype)
        ref = torch.nn.functional.linear(x2, w, bias)
        tol = 1e-2 * max(1.0, ref.abs().max().item())
        best: tuple[float, _Algo] | None = None
        for algo in cands:
            trial = _Plan(desc, a, b, d, algo)
            try:
                self.matmul(trial, w, x2, bias, out)
                torch.cuda.synchronize(x2.device)
                if not (out - ref).abs().max().item() <= tol:
                    continue
                for _ in range(2):
                    self.matmul(trial, w, x2, bias, out)
                torch.cuda.synchronize(x2.device)
                t = time.perf_counter()
                for _ in range(8):
                    self.matmul(trial, w, x2, bias, out)
                torch.cuda.synchronize(x2.device)
                ms = (time.perf_counter() - t) * 1000 / 8
            except RuntimeError:
                continue
            if best is None or ms < best[0]:
                best = (ms, algo)
        if best is None:
            raise RuntimeError("no hipBLASLt candidate matched F.linear")
        p = _Plan(desc, a, b, d, best[1])
        self.plans[key] = p
        logger.debug("hipBLASLt plan %s: %.3f ms of %d candidates", key[:5], best[0], len(cands))
        return p


def _library_path() -> Path:
    base = Path(torch.__file__).resolve().parent.parent
    names = ["libhipblaslt.dll"] if sys.platform == "win32" else ["libhipblaslt.so", "libhipblaslt.so.1"]
    for d in (base / "_rocm_sdk_libraries" / "bin", base / "_rocm_sdk_libraries" / "lib", base / "torch" / "lib"):
        for name in names:
            if (d / name).exists():
                return d / name
    raise FileNotFoundError("libhipblaslt not found next to torch")


_lt: _Lt | None = None
_failed = False
_lock = threading.Lock()


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
    """
    F.linear(x, w, bias) through hipBLASLt with a timed algorithm, or None if
    this path does not apply. x: (..., K) rows with unit inner stride and a
    uniform row stride; w: contiguous (N, K); bias: contiguous (N,) or None;
    all FP16 or all BF16 on one GPU.
    """
    global _lt, _failed
    if _failed or os.environ.get("SPANDREL_HIPBLASLT", "").strip() == "0":
        return None
    if (
        not x.is_cuda
        or x.dtype not in _DTYPES
        or w.dtype != x.dtype
        or (bias is not None and (bias.dtype != x.dtype or not bias.is_contiguous()))
        or not w.is_contiguous()
        or x.stride(-1) != 1
    ):
        return None
    lead = x.shape[:-1]
    m = 1
    for s in lead:
        m *= s
    k = x.shape[-1]
    if m == 0 or x.dim() < 2:
        return None
    try:
        x2 = x.reshape(m, k) if x.dim() > 2 else x
    except RuntimeError:
        return None
    if x2.stride(-1) != 1 or x2.data_ptr() != x.data_ptr():
        return None
    try:
        with _lock:
            if _lt is None:
                _lt = _Lt()
            plan = _lt.plan(x2, w, bias)
        out = torch.empty((m, w.shape[0]), device=x.device, dtype=x.dtype)
        _lt.matmul(plan, w, x2, bias, out)
        return out.view(*lead, w.shape[0])
    except Exception as e:  # noqa: BLE001
        _failed = True
        logger.warning("hipBLASLt direct path failed, using F.linear: %s", e)
        return None
