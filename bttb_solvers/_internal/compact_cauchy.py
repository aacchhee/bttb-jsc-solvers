# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial/shared. Compact Cauchy transform operations used by the grouped source/chart constructions.

Entry points: CompactCauchy.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.fast_linear import inverse_and_determinant


class CompactCauchy:
    def __init__(self, catalog, G, H, diagonal, precision, *, width=None,
                 _exact=True):
        if type(precision) is not int or precision < 1:
            raise ValueError("positive integer precision required")
        self.catalog, self.F, self.precision = catalog, catalog.F, precision
        s = catalog.s
        if width is None:
            width = len(G[0]) if s and len(G) == s else 0
        if type(width) is not int or width < 0:
            raise ValueError("nonnegative integer width required")
        if len(G) != s or len(H) != s or len(diagonal) != s:
            raise ValueError("catalog row count mismatch")
        if any(len(row) != width for factor in (G, H) for row in factor):
            raise ValueError("generator width mismatch")
        if any(len(poly) != precision for factor in (G, H)
               for row in factor for poly in row):
            raise ValueError("generator coefficient count mismatch")
        if any(len(row) != precision for row in diagonal):
            raise ValueError("diagonal coefficient count mismatch")
        self.width = width
        self.G = [[list(poly) for poly in row] for row in G]
        self.H = [[list(poly) for poly in row] for row in H]
        self.diagonal = [list(poly) for poly in diagonal]
        self._exact = _exact
        if _exact:
            for rowg, rowh in zip(self.G, self.H):
                defect = self._zero()
                for g, h in zip(rowg, rowh):
                    defect = self._add(defect, self._mul(g, h))
                if any(not self.F.is_zero(value) for value in defect):
                    raise ValueError("nonzero diagonal displacement product")

    def _zero(self):
        return [self.F.zero() for _ in range(self.precision)]

    def _add(self, a, b):
        return [self.F.add(x, y) for x, y in zip(a, b)]

    def _sub(self, a, b):
        return [self.F.sub(x, y) for x, y in zip(a, b)]

    def _neg(self, a):
        return [self.F.neg(x) for x in a]

    def _mul(self, a, b):
        return self.catalog.mul(a, b)[:self.precision]

    def at(self, precision, *, _exact=None):
        """Truncate, or coefficient-zero-extend a represented offdiagonal lift.

        Extending exact factors need not preserve zero diagonal product. Such a
        lift is marked inexact and may be acted with, but not multiplied as an
        exact displacement or used as an inverse input.
        """
        if type(precision) is not int or precision < 1:
            raise ValueError("positive integer precision required")
        def resize(poly):
            return poly[:precision] + [self.F.zero()] * max(0, precision-len(poly))
        exact = self._exact and precision <= self.precision if _exact is None else _exact
        return CompactCauchy(self.catalog,
            [[resize(p) for p in row] for row in self.G],
            [[resize(p) for p in row] for row in self.H],
            [resize(p) for p in self.diagonal], precision,
            width=self.width, _exact=exact)

    def action(self, vector, *, transpose=False):
        """Equation (5), also valid for a temporary offdiagonal-only lift."""
        s, P = self.catalog.s, self.precision
        if len(vector) != s or any(len(row) != P for row in vector):
            raise ValueError("vector must have shape (catalog size, precision)")
        out = [self._mul(d, v) for d, v in zip(self.diagonal, vector)]
        left, right = (self.H, self.G) if transpose else (self.G, self.H)
        for u in range(self.width):
            weights = [self._mul(right[i][u], vector[i]) for i in range(s)]
            transformed = self.catalog.apply(weights, P)[0]
            for i in range(s):
                term = self._mul(left[i][u], transformed[i])
                out[i] = self._sub(out[i], term) if transpose else self._add(out[i], term)
        return out

    def _compatible(self, other):
        if self.catalog is not other.catalog or self.precision != other.precision:
            raise ValueError("same catalog object and precision required")

    def product_diagonal(self, other):
        """Literal equation (7), valid even for offdiagonal-only lifts."""
        self._compatible(other)
        s, P = self.catalog.s, self.precision
        out = [self._mul(a, b) for a, b in zip(self.diagonal, other.diagonal)]
        for u in range(self.width):
            for v in range(other.width):
                weights = [self._mul(self.H[i][u], other.G[i][v]) for i in range(s)]
                squared = self.catalog.apply(weights, P)[1]
                for i in range(s):
                    out[i] = self._sub(out[i], self._mul(
                        self._mul(self.G[i][u], other.H[i][v]), squared[i]))
        return out

    def product(self, other):
        """Exact factor product (6), and its independently acquired diagonal."""
        self._compatible(other)
        if not self._exact or not other._exact:
            raise ValueError("factor product requires exact displacements")
        s, P = self.catalog.s, self.precision
        newg = [[list(p) for p in row] for row in self.G]
        newh = [[] for _ in range(s)]
        for u in range(self.width):
            col = other.action([self.H[i][u] for i in range(s)], transpose=True)
            for i in range(s):
                newh[i].append(col[i])
        for v in range(other.width):
            col = self.action([other.G[i][v] for i in range(s)])
            for i in range(s):
                newg[i].append(col[i])
                newh[i].append(list(other.H[i][v]))
        return CompactCauchy(self.catalog, newg, newh,
            self.product_diagonal(other), P, width=self.width+other.width)

    def inverse(self, *, return_determinant=False, lift_hook=None):
        """Actual inverse factors/diagonal, equations (9)–(16).

        lift_hook(state) observes each correct retained precision, including the
        constant seed. It is a validation seam, never used in production work.
        Optional determinant is det(A(0)) only, not a series determinant.
        """
        if not self._exact:
            raise ValueError("inverse input requires an exact displacement")
        F, s, r, P = self.F, self.catalog.s, self.width, self.precision
        constant = [[F.zero() for _ in range(s)] for _ in range(s)]
        for i in range(s):
            for j in range(s):
                if i == j:
                    constant[i][j] = self.diagonal[i][0]
                else:
                    value = F.zero()
                    for u in range(r):
                        value = F.add(value, F.mul(self.G[i][u][0], self.H[j][u][0]))
                    constant[i][j] = F.mul(value, F.inv(F.sub(
                        self.catalog.labels[i], self.catalog.labels[j])))
        seed, determinant = inverse_and_determinant(F, constant)
        X = [[[F.zero()] for _ in range(r)] for _ in range(s)]
        Y = [[[F.zero()] for _ in range(r)] for _ in range(s)]
        for i in range(s):
            for u in range(r):
                for j in range(s):
                    X[i][u][0] = F.add(X[i][u][0], F.mul(seed[i][j], self.G[j][u][0]))
                    Y[i][u][0] = F.add(Y[i][u][0], F.mul(seed[j][i], self.H[j][u][0]))
        state = CompactCauchy(self.catalog,
            [[[F.neg(p[0])] for p in row] for row in X], Y,
            [[seed[i][i]] for i in range(s)], 1, width=r)
        del seed, constant
        if lift_hook is not None:
            lift_hook(state)
        m = 1
        while m < P:
            t = min(2*m, P)
            A, B = self.at(t), state.at(t)

            
            U, V = B.G, B.H
            e = [B._zero() for _ in range(s)]
            for i in range(s):
                for u in range(r):
                    e[i] = B._add(e[i], B._mul(U[i][u], V[i][u]))
            Z = [[None]*r for _ in range(s)]
            T = [[None]*r for _ in range(s)]
            for u in range(r):
                zcol = A.action([U[i][u] for i in range(s)])
                tcol = B.action([A.H[i][u] for i in range(s)], transpose=True)
                for i in range(s):
                    Z[i][u], T[i][u] = zcol[i], tcol[i]
            c = A.product_diagonal(B)
            q = [B._mul(b, ci) for b, ci in zip(B.diagonal, c)]
            for u in range(r):
                for a in range(r):
                    weights = [B._mul(V[i][u], A.G[i][a]) for i in range(s)]
                    _, S2, S3 = self.catalog.apply(weights, t)
                    for i in range(s):
                        q[i] = B._sub(q[i], B._mul(B._mul(U[i][u], T[i][a]), S2[i]))

                        q[i] = B._sub(q[i], B._mul(e[i], B._mul(
                            B._mul(U[i][u], A.H[i][a]), S3[i])))
                for v in range(r):
                    weights = [B._mul(V[i][u], Z[i][v]) for i in range(s)]
                    S2 = self.catalog.apply(weights, t)[1]
                    for i in range(s):
                        q[i] = B._sub(q[i], B._mul(B._mul(U[i][u], V[i][v]), S2[i]))
            diagonal = [B._sub(B._add(b, b), qi) for b, qi in zip(B.diagonal, q)]

            
            length = t-m
            lower = state.at(length)
            newx = [[None]*r for _ in range(s)]
            newy = [[None]*r for _ in range(s)]
            for u in range(r):
                xm = [B._neg(U[i][u]) for i in range(s)]
                ym = [V[i][u] for i in range(s)]
                ax = A.action(xm)
                aty = A.action(ym, transpose=True)
                rx = [A._sub(A.G[i][u], ax[i])[m:t] for i in range(s)]
                ry = [A._sub(A.H[i][u], aty[i])[m:t] for i in range(s)]
                dx, dy = lower.action(rx), lower.action(ry, transpose=True)
                for i in range(s):
                    newx[i][u] = xm[i][:m] + dx[i]
                    newy[i][u] = ym[i][:m] + dy[i]
            state = CompactCauchy(self.catalog,
                [[[F.neg(v) for v in p] for p in row] for row in newx],
                newy, diagonal, t, width=r)
            m = t
            if lift_hook is not None:
                lift_hook(state)
        return (state, determinant) if return_determinant else state
