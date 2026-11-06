# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial. Construct wide-band physical displacement generators directly from original Toeplitz lags and their reversed orientation.

Entry points: WideGenerator; acquire.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from dataclasses import dataclass


@dataclass
class WideGenerator:
    n: int
    k: int
    G: list
    H: list
    first_row: list

    @property
    def physical_order(self):
        return self.n * self.n

    @property
    def order(self):
        return len(self.G)

    @property
    def width(self):
        return len(self.G[0])

    def rhs(self, F, physical):
        """Pi b, followed by the mandatory zero padding when n=2,3."""
        if len(physical) != self.physical_order:
            raise ValueError("physical RHS must have n^2 coordinates")
        n = self.n
        return [physical[i*n+a] for a in range(n) for i in range(n)] + [
            F.zero() for _ in range(self.order-self.physical_order)
        ]

    def recover(self, represented):
        """Discard padding and apply Pi^t; input follows terminal reversal undo."""
        if len(represented) != self.order:
            raise ValueError("represented vector has incorrect order")
        n = self.n
        return [represented[a*n+i] for i in range(n) for a in range(n)]

    def stein(self, F):
        """algorithm I: R Ahat R - J R Ahat R J^t = G0 H0^t.

        This is c=1 only: G0=[R P^-1 G | e0], H0=[R H | J R v].
        For a weighted cyclic pair the appended e0 needs c^-1; this API
        deliberately does not accept such a pair. The terminal must reverse
        rhs(self.rhs(...)) before its solve and reverse its result afterwards.
        """
        D = self.order
        G0 = [list(self.G[(D-i) % D]) + [F.one() if i == 0 else F.zero()]
              for i in range(D)]
        H0 = [list(self.H[D-1-i]) + [F.zero() if i == 0 else self.first_row[D-i]]
              for i in range(D)]
        return G0, H0


def reverse_raw(n, k, raw):
    """Independent literal-transpose source t'[j,s]=t[-j,-s]."""
    _dimensions(n, k)
    return {(j, s): raw[-j, -s]
            for j in range(-k, k+1) for s in range(1-n, n)}


def _dimensions(n, k):
    if type(n) is not int or type(k) is not int:
        raise ValueError("n and k must be integers")
    if not 1 <= k < n or 2*k < n:
        raise ValueError("wide branch requires 1<=k<n and 2k>=n")


def acquire(F, n, k, raw):
    """Construct the literal algorithm factors in O(n^3) work/storage.

    Does not assume nonsingularity or inspect element truth/int conversions.
    A future nonsingular terminal supplies the solve, with its own cost.
    """
    _dimensions(n, k)

    for j in range(-k, k+1):
        for s in range(1-n, n):
            if (j, s) not in raw:
                raise ValueError("missing raw lag coefficient")

    def lag(j, s):
        return raw[j, s] if -k <= j <= k else F.zero()


    def entry(q, p):
        a, i = divmod(q, n)
        b, j = divmod(p, n)
        return lag(i-j, a-b)

    d = n*n
    D = d+4*n if n in (2, 3) else d
    alpha = 4*n+1 if n in (2, 3) else 4*n-1
    G = [[F.zero() for _ in range(alpha)] for _ in range(D)]
    H = [[F.zero() for _ in range(alpha)] for _ in range(D)]

    for a in range(n):
        G[a*n][a] = F.one()
        for b in range(n):
            for j in range(n):
                H[b*n+j][a] = (lag(0, a-b) if j == n-1 else
                    F.sub(lag(n-1-j, a-b), lag(-j-1, a-b)))

    for b in range(n):
        H[b*n+n-1][n+b] = F.one()
        for a in range(n):
            for i in range(1, n):
                G[a*n+i][n+b] = lag(i-n, a-b)

    for a in range(n):
        c = 2*n+a
        G[((a+1) % n)*n][c] = F.one()
        G[a*n][c] = F.neg(F.one())
        for p in range(d):
            H[p][c] = entry(a*n+n-1, p)

    for a in range(n-1):
        c = 3*n+a
        H[a*n+n-1][c] = F.one()
        for q in range(d):
            G[q][c] = F.neg(entry(q, (a+1)*n))
    if D != d:
        c = 4*n-1
        G[0][c] = F.one()
        G[d][c+1] = F.one()
        for p in range(d):
            endpoint = entry(d-1, p)
            H[p][c] = F.neg(endpoint)
            H[p][c+1] = F.sub(endpoint, F.one() if p == d-1 else F.zero())
        H[D-1][c] = F.one()

    first = [entry(0, p) for p in range(d)] + [F.zero() for _ in range(D-d)]
    return WideGenerator(n, k, G, H, first)
