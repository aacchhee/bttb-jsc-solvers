# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .errors import WidthOutOfRange, SupportViolation

CONVENTION = (
    "A[delta_o, delta_i]; delta_o = s-q in [-(n-1), n-1] free (spatial), "
    "delta_i = i-j in [-w, n-1] constrained (temporal), zero for delta_i < -w. "
    "(TX)[i,s] = sum A[s-q, i-j] X[j,q]. vec index i*n+s (temporal major), so "
    "T[i*n+s][j*n+q] = A[s-q, i-j]. B_l[q,s] = A[s-q,l] = M_l^T with "
    "M_l[s][q] = A[s-q,l]."
)


class HalfStripSymbol(object):
    """n, w and the lag array.  Validates the class membership and nothing more."""

    __slots__ = ("F", "n", "w", "lags")

    def __init__(self, F, n, w, lags):
        """`lags` is a mapping l -> sequence of 2n-1 field elements, for
        l = -w .. n-1 (missing l is read as the zero lag vector).  Any l outside
        that range with a nonzero entry raises SupportViolation."""
        if not (1 <= w < n):
            raise WidthOutOfRange(w, n)
        self.F, self.n, self.w = F, n, w
        z = tuple([F.zero()] * (2 * n - 1))
        store = {}
        for l, vec in lags.items():
            v = tuple(vec)
            if len(v) != 2 * n - 1:
                raise ValueError("lag vector for l = %d has length %d, expected %d"
                                 % (l, len(v), 2 * n - 1))
            if l < -w or l > n - 1:
                for sigma, x in enumerate(v):
                    if not F.is_zero(x):
                        raise SupportViolation(l, sigma - (n - 1), w)
                continue
            store[l] = v
        for l in range(-w, n):
            store.setdefault(l, z)
        self.lags = store


    def lag(self, l):
        """The lag vector A[:, l], indexed by sigma = delta_o + (n-1)."""
        if l < -self.w or l > self.n - 1:
            return tuple([self.F.zero()] * (2 * self.n - 1))
        return self.lags[l]

    def leading_lag(self):
        """A[:, -w], the lag vector of B_{-w}."""
        return self.lags[-self.w]

    def block_M(self, l):
        """The dense M_l with M_l[s][q] = A[s-q, l].  ORACLE / DIAGNOSTIC ONLY."""
        n = self.n
        v = self.lag(l)
        return [[v[(s - q) + (n - 1)] for q in range(n)] for s in range(n)]

    def free_parameters(self):
        """(n + w)(2n - 1), the Statement's count."""
        return (self.n + self.w) * (2 * self.n - 1)

    def describe(self):
        return {"n": self.n, "w": self.w, "field": self.F.name(), "q": self.F.q,
                "order_N": self.n * self.n, "free_parameters": self.free_parameters(),
                "generic_bttb_parameters": (2 * self.n - 1) ** 2,
                "convention": CONVENTION}


def from_banded_lags(F, n, k, alag):
    """The containment map of `ext105-containment.md` section 5(i), made concrete.

    `alag[l]`, l = -k .. k, is the banded class's inner lag vector
    (`a_l[delta_o]`, index delta_o + (n-1)), with m = n block rows.  Returns the
    half-strip symbol at w = k with A[delta_o, delta_i] := a_{delta_i}[delta_o].
    Under this package's flattening the two dense matrices are EQUAL, not merely
    permutation-similar (see the module docstring)."""
    if not (1 <= k < n):
        raise WidthOutOfRange(k, n)
    return HalfStripSymbol(F, n, k, {l: tuple(alag[l]) for l in range(-k, k + 1)})
