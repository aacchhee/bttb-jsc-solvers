# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.banded_bttb.toeplitz import gauss_rank
from bttb_solvers._internal.halfstrip_core.flint_backend import lu_update, enabled_prime, dot


class LU(object):
    """PA = LU with unit-diagonal L, stored in one array.  `singular` is a flag,
    never an exception: the caller decides what a singular terminal matrix means."""

    __slots__ = ("F", "n", "M", "perm", "sign", "det", "singular", "rank")

    def __init__(self, F, A):
        self.F = F
        n = self.n = len(A)
        M = [list(r) for r in A]
        perm = list(range(n))
        sign = 1
        singular = False
        for c in range(n):
            piv = None
            for r in range(c, n):
                if not F.is_zero(M[r][c]):
                    piv = r
                    break
            if piv is None:
                singular = True
                continue
            if piv != c:
                M[c], M[piv] = M[piv], M[c]
                perm[c], perm[piv] = perm[piv], perm[c]
                sign = -sign
            iv = F.inv(M[c][c])
            if lu_update(F,M,c,iv):
                continue
            for r in range(c + 1, n):
                if not F.is_zero(M[r][c]):
                    f = F.mul(M[r][c], iv)
                    M[r][c] = f
                    for j in range(c + 1, n):
                        M[r][j] = F.sub(M[r][j], F.mul(f, M[c][j]))
        self.M, self.perm, self.sign, self.singular = M, perm, sign, singular
        if singular:
            self.det = F.zero()
            self.rank = gauss_rank(F, A)
        else:
            d = F.one()
            for c in range(n):
                d = F.mul(d, M[c][c])
            self.det = d if sign > 0 else F.neg(d)
            self.rank = n

    def nullity(self):
        return self.n - self.rank

    def solve(self, b):
        """x with A x = b.  Returns None if A is singular."""
        if self.singular:
            return None
        F, n, M = self.F, self.n, self.M
        if enabled_prime(F) is not None:
            y=[]
            for i in range(n):
                y.append(F.sub(b[self.perm[i]],dot(F,M[i][:i],y)))
            x=[F.zero()]*n
            for i in range(n-1,-1,-1):
                x[i]=F.div(F.sub(y[i],dot(F,M[i][i+1:],x[i+1:])),M[i][i])
            return x
        y = [b[self.perm[i]] for i in range(n)]
        for i in range(n):
            for j in range(i):
                if not F.is_zero(M[i][j]) and not F.is_zero(y[j]):
                    y[i] = F.sub(y[i], F.mul(M[i][j], y[j]))
        x = [F.zero()] * n
        for i in range(n - 1, -1, -1):
            acc = y[i]
            for j in range(i + 1, n):
                if not F.is_zero(M[i][j]) and not F.is_zero(x[j]):
                    acc = F.sub(acc, F.mul(M[i][j], x[j]))
            x[i] = F.div(acc, M[i][i])
        return x

    def solve_transpose(self, b):
        """x with A^T x = b, from the SAME factorisation."""
        if self.singular:
            return None
        F, n, M = self.F, self.n, self.M
        if enabled_prime(F) is not None:
            z=[]
            for i in range(n):
                z.append(F.div(F.sub(b[i],dot(F,[M[j][i] for j in range(i)],z)),M[i][i]))
            y=[F.zero()]*n
            for i in range(n-1,-1,-1):
                y[i]=F.sub(z[i],dot(F,[M[j][i] for j in range(i+1,n)],y[i+1:]))
            x=[F.zero()]*n
            for i in range(n): x[self.perm[i]]=y[i]
            return x

        z = [F.zero()] * n
        for i in range(n):
            acc = b[i]
            for j in range(i):
                if not F.is_zero(M[j][i]) and not F.is_zero(z[j]):
                    acc = F.sub(acc, F.mul(M[j][i], z[j]))
            z[i] = F.div(acc, M[i][i])
        y = [F.zero()] * n
        for i in range(n - 1, -1, -1):
            acc = z[i]
            for j in range(i + 1, n):
                if not F.is_zero(M[j][i]) and not F.is_zero(y[j]):
                    acc = F.sub(acc, F.mul(M[j][i], y[j]))
            y[i] = acc
        x = [F.zero()] * n
        for i in range(n):
            x[self.perm[i]] = y[i]
        return x
