# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Shared. Jacobi/inverse-series source construction for the uniform/CK comparison routes.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from .convolve import ConvolutionBackend, next_pow2


def build_series(sym, cnt, leading, backend=None, stats=None):
    """Return all L=n+w coefficients as ``C[d][s][q]``.

    Each leaf solves n columns, including degree zero's identity forcing.
    Every completed-source/unfinished-target pair belongs to one CDQ merge.
    ``stats``, when supplied, receives counts for this construction only.
    Storage is O(L n^2), including the pending right-hand sides.
    """
    F, n, L = sym.F, sym.n, sym.n + sym.w
    if leading.n != n:
        raise ValueError("leading inverse order must equal symbol order")
    if leading.F is not sym.F or leading.cnt is not cnt:
        raise ValueError("leading inverse must use sym.F and the caller's counter")
    if tuple(leading.lag) != tuple(sym.leading_lag()):
        raise ValueError("leading inverse must be built from sym.leading_lag()")
    if backend is None:
        backend = ConvolutionBackend(
            F, cnt, "auto", (next_pow2(2 * L), next_pow2(3 * n - 2)))

    rhs = [[[F.zero()] * n for _ in range(n)] for _ in range(L)]
    columns = [[None] * n for _ in range(L)]
    for q in range(n):
        rhs[0][q][q] = F.one()
    kernel = [None] + [sym.lag(d - sym.w) for d in range(1, L)]
    st = {"merges": 0, "leaf_applies": 0, "convolution_cells": 0,
          "scalar_2d_convolutions": 0}
    seen_shapes = set()

    def cdq(lo, hi):
        if hi - lo == 1:
            with cnt.stage("apply_inv"):
                for q in range(n):
                    columns[lo][q] = leading.apply(rhs[lo][q])
            st["leaf_applies"] += n
            return
        mid = (lo + hi) // 2
        cdq(lo, mid)
        length, left = hi - lo, mid - lo
        nt, ns = next_pow2(left + length - 1), next_pow2(3 * n - 2)
        shape = (length, nt, ns)
        with cnt.stage("merge"):
            fresh = False
            for q in range(n):
                sources = [columns[lo + a][q] for a in range(left)]
                out, column_fresh = backend.merge_window(
                    sources, kernel[:length], n, left, length, left)
                fresh = fresh or column_fresh
                for tau, vec in out.items():
                    target = rhs[lo + tau][q]
                    for s in range(n):
                        target[s] = F.sub(target[s], vec[s])
            model_fresh = fresh if backend.kind == "ntt" else shape not in seen_shapes
            cnt.charge_conv2d(
                n, nt, ns, kernel_cells=nt * ns if model_fresh else 0,
                raw=(left + length - 1, 3 * n - 2),
                engine=backend.kind, kernel_fresh=fresh)
        seen_shapes.add(shape)
        st["merges"] += 1
        st["scalar_2d_convolutions"] += n
        st["convolution_cells"] += n * (left + length - 1) * (3 * n - 2)
        cdq(mid, hi)

    cdq(0, L)
    if stats is not None:
        stats.update(st)
        stats["kernel_shapes"] = len(seen_shapes)
    return [[[columns[d][q][s] for q in range(n)] for s in range(n)]
            for d in range(L)]


def apply_window(F, C, b, in_blocks, out_blocks, offset, backend=None):
    """Apply blocks ``C[offset+i-j]`` to flat temporal-major ``b``.

    Negative or unavailable coefficients are zero. Returns a flat
    ``out_blocks*n`` vector, using n^2 scalar linear convolutions. Caller
    owns the arithmetic stage; the shared backend owns convolution charges.
    """
    if not C or not C[0] or len(C[0]) != len(C[0][0]):
        raise ValueError("nonempty square series coefficients required")
    if in_blocks < 0 or out_blocks < 0:
        raise ValueError("block counts must be nonnegative")
    n = len(C[0])
    if len(b) != in_blocks * n:
        raise ValueError("right-hand side length must equal in_blocks*n")
    out = [F.zero()] * (out_blocks * n)
    if in_blocks == 0 or out_blocks == 0:
        return out
    if backend is None:
        backend = ConvolutionBackend(
            F, None, "auto", (next_pow2(len(C) + in_blocks - 1),))
    for s in range(n):
        for q in range(n):
            coefficients = [block[s][q] for block in C]
            source = [b[j * n + q] for j in range(in_blocks)]
            product = backend.linear(coefficients, source)
            for i in range(out_blocks):
                degree = offset + i
                if 0 <= degree < len(product):
                    out[i * n + s] = F.add(out[i * n + s], product[degree])
    return out
