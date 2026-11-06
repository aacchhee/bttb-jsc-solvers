# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .dense import LU


class BlockChartBreakdown(ArithmeticError):
    def __init__(self, section):
        self.section = section
        super().__init__("singular leading block section of order %d" % section)


def zeros(F, n, m=None):
    return [[F.zero()] * (n if m is None else m) for _ in range(n)]


def add(F, A, B, subtract=False):
    op = F.sub if subtract else F.add
    return [[op(a, b) for a, b in zip(ar, br)] for ar, br in zip(A, B)]


def matmul(F, A, B):
    nr, inner, nc = len(A), len(B), len(B[0])
    out = zeros(F, nr, nc)
    for i in range(nr):
        for k in range(inner):
            for j in range(nc):
                out[i][j] = F.add(out[i][j], F.mul(A[i][k], B[k][j]))
    return out


def solve_matrix(lu, B):
    columns = [lu.solve(list(c)) for c in zip(*B)]
    return [list(r) for r in zip(*columns)]


def row_product(F, row, col):
    out = zeros(F, len(row[0]), len(col[0][0]))
    for a, b in zip(row, col):
        out = add(F, out, matmul(F, a, b))
    return out


def materialize(lags, width):
    n = len(lags[0])
    return [[lags[i // n - j // n][i % n][j % n]
             for j in range(width * n)] for i in range(width * n)]


class BlockLevinson:
    """Factor a ``lags[i-j]`` block Toeplitz matrix using two predictors.

    Reflection pairs reconstruct predictors during each solve so retained
    storage is linear in width. This adds O(w**2 n**3) work per query; it is
    deliberately reported rather than hidden behind a factorization label.
    """
    def __init__(self, F, lags, width, fallback="raise"):
        if fallback not in ("raise", "dense"):
            raise ValueError("fallback must be 'raise' or 'dense'")
        if width < 1 or set(lags) != set(range(1-width, width)):
            raise ValueError("all 2w-1 lag blocks are required")
        self.F, self.lags, self.width = F, lags, width
        self.block_size = len(lags[0])
        n = self.block_size
        if n < 1 or any(len(a) != n or any(len(r) != n for r in a)
                        for a in lags.values()):
            raise ValueError("lag blocks must be nonempty square matrices")
        self.n = n * width
        self.factors, self.reflections = [], []
        self.dense = None
        self.breakdown_section = None
        try:
            self._factor()
        except BlockChartBreakdown as exc:
            self.breakdown_section = exc.section
            if fallback == "raise":
                raise
            self.dense = LU(F, materialize(lags, width))
            self.factors, self.reflections = [], []
            self.det, self.singular, self.rank = (
                self.dense.det, self.dense.singular, self.dense.rank)

    def _checked(self, A, section):
        factor = LU(self.F, A)
        if factor.singular:
            raise BlockChartBreakdown(section)
        return factor

    def _factor(self):
        F, a, w = self.F, self.lags, self.width
        first = self._checked(a[0], 1)
        self.factors.append(first)
        self.det = first.det
        if w > 1:
            U, V = [solve_matrix(first, a[1])], [solve_matrix(first, a[-1])]
        for k in range(1, w):
            d, r = [a[k-i] for i in range(k)], [a[-1-i] for i in range(k)]
            S = self._checked(add(F, a[0], row_product(F, d, V), True), k+1)
            self.factors.append(S)
            self.det = F.mul(self.det, S.det)
            if k + 1 < w:
                Sbar = self._checked(add(F, a[0], row_product(F, r, U), True), k+1)
                A = solve_matrix(S, add(F, a[k+1], row_product(F, d, U), True))
                B = solve_matrix(Sbar, add(F, a[-k-1], row_product(F, r, V), True))
                self.reflections.append((A, B))
                U, V = ([add(F, u, matmul(F, v, A), True) for u, v in zip(U, V)] + [A],
                        [B] + [add(F, v, matmul(F, u, B), True) for u, v in zip(U, V)])
        self.singular, self.rank = False, self.n

    def solve(self, b):
        if len(b) != self.n:
            raise ValueError("wrong terminal right-hand-side length")
        if self.dense is not None:
            return self.dense.solve(b)
        F, a, w, n = self.F, self.lags, self.width, self.block_size
        x = [[[v] for v in self.factors[0].solve(b[:n])]]
        if w > 1:
            U = [solve_matrix(self.factors[0], a[1])]
            V = [solve_matrix(self.factors[0], a[-1])]
        for k in range(1, w):
            d = [a[k-i] for i in range(k)]
            t = solve_matrix(self.factors[k], add(
                F, [[v] for v in b[k*n:(k+1)*n]], row_product(F, d, x), True))
            x = [add(F, xi, matmul(F, vi, t), True) for xi, vi in zip(x, V)] + [t]
            if k + 1 < w:
                A, B = self.reflections[k-1]
                U, V = ([add(F, u, matmul(F, v, A), True) for u, v in zip(U, V)] + [A],
                        [B] + [add(F, v, matmul(F, u, B), True) for u, v in zip(U, V)])
        return [row[0] for block in x for row in block]

    def nullity(self):
        return self.n-self.rank

    def summary(self):
        return {"kind": "block_levinson_bridge" if self.dense is None else "dense_fallback",
                "terminal_rate": "O(w^2 n^3)" if self.dense is None else "O(w^3 n^3)",
                "published_rate_achieved": False,
                "breakdown_section": self.breakdown_section,
                "full_terminal_materialized": self.dense is not None,
                "stored_scalar_cells": ((2*self.width-1 + len(self.factors) +
                                          2*len(self.reflections))*self.block_size**2
                                         if self.dense is None else self.n**2),
                "chart": "all leading block sections nonsingular"}
