# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Shared. Compact one-level Toeplitz inverse of the leading block and its convolution applications.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.banded_bttb.toeplitz import toeplitz_solve, toeplitz_dense, gauss_det
from bttb_solvers._internal.banded_bttb.errors import SingularToeplitz

from .convolve import (ConvolutionBackend, linear_conv, cyclic_conv,
                       cyclic_corr, linear_corr, next_pow2)
from .errors import SingularLeadingBlock


def _cyc_shift(v, n):
    """Z_1 v, the CYCLIC down shift: (Z_1 v)[i] = v[(i-1) mod n]."""
    return [v[(i - 1) % n] for i in range(n)]


class LeadingInverse(object):
    """The compact representation of `T^{-1}` for one scalar Toeplitz `T`.

    Attributes: `g1, h1, g2, h2` (lists of n field elements), `n`, `det`
    (det T, exactly)."""

    __slots__ = ("F", "n", "g1", "h1", "g2", "h2", "det", "cnt", "lag",
                 "backend")

    def __init__(self, F, lag, n, cnt, want_det=True, backend=None):
        if len(lag) != 2 * n - 1:
            raise ValueError("lag vector of length %d expected" % (2 * n - 1))
        self.F, self.n, self.cnt, self.lag = F, n, cnt, tuple(lag)
        self.backend = backend or ConvolutionBackend(
            F, cnt, "auto", (next_pow2(2 * n - 1),))

        def a(l):
            return lag[l + (n - 1)]


        r = [None] * n
        c = [None] * n
        for j in range(n):
            i = n - 1
            v = F.zero()
            r[j] = F.sub(a(i - j), v)
        for i in range(n):
            j = n - 1
            if i == n - 1:
                v = F.zero()
            else:
                v = a((i + 1) - 0)          
            c[i] = F.sub(a(i - j), v)
        c[n - 1] = F.zero()

        e0 = [F.zero()] * n
        e0[0] = F.one()
        en = [F.zero()] * n
        en[n - 1] = F.one()
        try:
            al = toeplitz_solve(F, lag, n, en)                                  
            ga = toeplitz_solve(F, lag, n, c)                                   
            be = toeplitz_solve(F, lag, n, _cyc_shift(r, n), transpose=True)    
            de = toeplitz_solve(F, lag, n, e0, transpose=True)                  
        except SingularToeplitz:
            raise SingularLeadingBlock(n=n, lag=tuple(lag))

        self.g1, self.h1 = _cyc_shift(al, n), be
        self.g2, self.h2 = _cyc_shift(ga, n), de

        

        

        self.det = gauss_det(F, toeplitz_dense(F, lag, n)) if want_det else None


    def apply(self, v):
        """T^{-1} v.  EXACTLY FOUR 1-D convolutions, on every branch."""
        F, n, cnt = self.F, self.n, self.cnt
        if len(v) != n:
            raise ValueError("vector of length %d expected" % n)
        before = cnt.convolutions
        out = [F.zero()] * n
        for g, h in ((self.g1, self.h1), (self.g2, self.h2)):
            s = linear_corr(F, h, v, cnt, self.backend)       
            part = cyclic_conv(F, g, s, cnt, self.backend)    
            for i in range(n):
                out[i] = F.add(out[i], part[i])
        cnt.charge_apply_inv()
        cnt.convolutions_on_apply += cnt.convolutions - before
        return out

    def apply_transpose(self, v):
        """T^{-T} v, from the SAME generators.  No second backend, no new solve,
        again exactly four 1-D convolutions."""
        F, n, cnt = self.F, self.n, self.cnt
        if len(v) != n:
            raise ValueError("vector of length %d expected" % n)
        before = cnt.convolutions
        out = [F.zero()] * n
        for g, h in ((self.g1, self.h1), (self.g2, self.h2)):
            t = cyclic_corr(F, g, v, cnt, self.backend)       
            part = linear_conv(F, h, t, cnt, self.backend)    
            for j in range(n):
                out[j] = F.add(out[j], part[j])
        cnt.charge_apply_inv()
        cnt.convolutions_on_apply += cnt.convolutions - before
        return out


    def dense(self):
        """The dense T^{-1}, by the division-free expansion.  TEST ORACLE ONLY;
        never called on the solve path, because materialising it and then doing
        dense matvecs is the QUARTIC row of `oc1_reledger.md` section 1."""
        F, n = self.F, self.n
        D = [[F.add(F.mul(self.g1[i], self.h1[j]), F.mul(self.g2[i], self.h2[j]))
              for j in range(n)] for i in range(n)]
        X = [[F.zero()] * n for _ in range(n)]
        for i in range(n):
            X[i][0] = D[i][0]
        for j in range(1, n):
            X[0][j] = F.add(X[n - 1][j - 1], D[0][j])
            for i in range(1, n):
                X[i][j] = F.add(X[i - 1][j - 1], D[i][j])
        return X
