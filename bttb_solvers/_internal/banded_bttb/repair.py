# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""CC Toeplitz, symbol and correction primitives.

Role: Shared. Normal-rank tests and low-rank Toeplitz symbol repair for CC.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from .errors import RepairFailed, FieldTooSmall
from .field import ordered_elements
from .symbol import symbol_lag_at, eval_points
from .toeplitz import (toeplitz_rank, toeplitz_dense, gauss_det, gauss_rank,
                       zeros)


def normal_rank(F, sym, points=None, return_point=False):
    """rank of a(z) over K(z).

    Deterministic: evaluate a at 2kn+1 distinct points of K* (the first such
    elements in the field's element order) and take the MAXIMUM rank.  Correct
    because every (n-s)-minor of a has degree <= 2k(n-s) < 2kn+1, so the rank
    drops at fewer than 2kn+1 points.

    raises FieldTooSmall if |K*| < 2kn+1.  Under (H3) that cannot happen; if it
    fires, (H3) was not checked and the instance is a bug worth a witness."""
    n, k = sym.n, sym.k
    need = 2 * k * n + 1
    if points is None:
        if F.q - 1 < need:
            d, qd = 1, F.q
            while qd - 1 < need:
                d += 1
                qd *= F.q
            raise FieldTooSmall(required_degree=d, q=F.q, k=k, n=n,
                                needed_nonzero=need)
        points = eval_points(F, need)
    else:
        points = list(points)
        if len(points) < need:
            raise ValueError("normal_rank needs at least %d points; got %d"
                             % (need, len(points)))
        def field_key(x):
            if F.e == 1:
                return int(x) % F.p
            return tuple(c % F.p for c in x)
        keys = [field_key(x) for x in points]
        zero_key = field_key(F.zero())
        if zero_key in keys:
            raise ValueError("normal_rank points must be nonzero in the field")
        if len(set(keys)) != len(points):
            raise ValueError("normal_rank points must be distinct in the field")
    best = 0
    best_point = None
    for x in points:
        r = toeplitz_rank(F, symbol_lag_at(F, sym, x), n)
        if r > best:
            best = r
            best_point = x
            if best == n:
                break
    return (best, best_point) if return_point else best


def projective_points(F, n):
    """P^1(K) in a FIXED order: [1:0], [0:1], then [1:c] for c in F* in element
    order.  len == q + 1.

    The two corner points matter: [1:0] gives e_n e_1^T and [0:1] gives
    e_1 e_n^T, and dropping them is exactly the error `-03`'s adversarial pass
    corrected when it moved the threshold from 2n to 2n-2."""
    one, zero = F.one(), F.zero()
    pts = [(one, zero), (zero, one)]
    for c in ordered_elements(F):
        if F.is_zero(c):
            continue
        pts.append((one, c))
    return pts


def _projective_point_iter(F):
    """The same fixed order as :func:`projective_points`, but lazy."""
    one, zero = F.one(), F.zero()
    yield one, zero
    yield zero, one
    for c in ordered_elements(F):
        if not F.is_zero(c):
            yield one, c


def _rank_one_factors(F, n, point):
    alpha, beta = point
    f = [F.mul(F.pow(alpha, i), F.pow(beta, n - 1 - i)) for i in range(n)]
    g = [F.mul(F.pow(beta, j), F.pow(alpha, n - 1 - j)) for j in range(n)]
    return f, g


def rank_one_toeplitz(F, n, point):
    """V_P for P = [alpha:beta]:  f(P)_i = alpha^{i-1} beta^{n-i},
    g(P)_j = beta^{j-1} alpha^{n-j},  V_P = f(P) g(P)^T, so
    (V_P)[i][j] = alpha^{n+i-j-1} beta^{n+j-i-1}: rank 1 and Toeplitz.

    Returns (V_P, f, g) with f, g as length-n column vectors."""
    f, g = _rank_one_factors(F, n, point)
    V = zeros(F, n, n)
    for i in range(n):
        for j in range(n):
            V[i][j] = F.mul(f[i], g[j])
    return V, f, g


def _add_into(F, A, B):
    return [[F.add(A[i][j], B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]


def _kernel_bases(F, A):
    """Return bases of ``ker(A^T)`` and ``ker(A)`` from one dense RREF.

    Row-reducing ``[A | I]`` gives ``E A = R``.  Free columns of ``R`` give
    the right kernel, while rows of ``E`` paired with zero rows of ``R`` give
    the left kernel.  Thus line L1.5 pays for one elimination, not two.
    """
    if not A:
        return [], []
    rows, cols = len(A), len(A[0])
    if rows != cols:
        raise ValueError("kernel-pairing matrix must be square")
    M = [list(A[i]) + [F.one() if i == j else F.zero() for j in range(rows)]
         for i in range(rows)]
    pivots = []
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if not F.is_zero(M[i][c])), None)
        if pivot is None:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        inv = F.inv(M[r][c])
        M[r] = [F.mul(x, inv) for x in M[r]]
        for i in range(rows):
            if i == r or F.is_zero(M[i][c]):
                continue
            a = M[i][c]
            M[i] = [F.sub(M[i][j], F.mul(a, M[r][j]))
                    for j in range(cols + rows)]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    right = []
    for fcol in free:
        v = [F.zero()] * cols
        v[fcol] = F.one()
        for row, pcol in reversed(list(enumerate(pivots))):
            v[pcol] = F.neg(M[row][fcol])
        right.append(v)
    left = [row[cols:] for row in M
            if all(F.is_zero(x) for x in row[:cols])]
    return left, right


def _dot(F, u, v):
    out = F.zero()
    for a, b in zip(u, v):
        if not F.is_zero(a) and not F.is_zero(b):
            out = F.add(out, F.mul(a, b))
    return out


class _IncrementalBasis(object):
    """RREF basis of a growing subspace of K^s; each test costs O(s^2)."""

    def __init__(self, F, dimension):
        self.F = F
        self.dimension = dimension
        self.rows = {}

    def candidate(self, vector):
        F = self.F
        v = list(vector)
        for pivot in sorted(self.rows):
            if F.is_zero(v[pivot]):
                continue
            a = v[pivot]
            row = self.rows[pivot]
            v = [F.sub(v[j], F.mul(a, row[j])) for j in range(self.dimension)]
        pivot = next((j for j, a in enumerate(v) if not F.is_zero(a)), None)
        if pivot is None:
            return None
        inv = F.inv(v[pivot])
        return pivot, [F.mul(a, inv) for a in v]

    def commit(self, item):
        pivot, vector = item
        F = self.F
        for oldpivot, row in list(self.rows.items()):
            if F.is_zero(row[pivot]):
                continue
            a = row[pivot]
            self.rows[oldpivot] = [
                F.sub(row[j], F.mul(a, vector[j])) for j in range(self.dimension)]
        self.rows[pivot] = vector


def find_E0(F, sym, s, stats=None, _normal_rank_witness=None):
    """(E0, X, Y) with E0 = X Y^T constant Toeplitz, rank(E0) == s EXACTLY, and
    det(a(z) + E0) not identically zero.  X, Y are n x s.

    pre: s == n - normal_rank(F, sym) >= 1 and |K| >= 2n-2 (implied by (H3)).

    DETERMINISTIC GREEDY over the fixed projective order.  The candidate test is
    exactly the kernel-pairing test of algorithm: ``M f(P)`` and ``N^T g(P)`` must
    each extend their maintained echelon basis.  No determinant, symbol rank or
    field restart occurs inside the scan.

    raises RepairFailed(witness) if the grid is exhausted.  That CONTRADICTS
    `-03` (d) whenever |K| >= 2n-2, so it is a REFUTATION and the headline
    finding, with the smallest witness - not something to work around.  Below
    that threshold it is the lemma being honest (GF(2), n = 4)."""
    n = sym.n
    if _normal_rank_witness is None:
        base, x0 = normal_rank(F, sym, return_point=True)
    else:
        base, x0 = _normal_rank_witness
    if s < 1 or s != n - base:
        raise ValueError("s must equal n - normal_rank(a) >= 1 (got s = %r, n - rank = %r)"
                         % (s, n - base))
    if x0 is None:

        
        x0 = eval_points(F, 1)[0]
    A = toeplitz_dense(F, symbol_lag_at(F, sym, x0), n)
    left_kernel, right_kernel = _kernel_bases(F, A)
    if len(left_kernel) != s or len(right_kernel) != s:
        raise RepairFailed({"reason": "kernel dimensions disagree with normal nullity",
                            "field": F.name(), "n": n, "s": s})

    left_span = _IncrementalBasis(F, s)
    right_span = _IncrementalBasis(F, s)
    E0 = zeros(F, n, n)
    chosen, X, Y = [], [], []
    scanned = 0
    for P in _projective_point_iter(F):
        scanned += 1
        f, g = _rank_one_factors(F, n, P)
        mf = [_dot(F, row, f) for row in left_kernel]
        ntg = [_dot(F, col, g) for col in right_kernel]
        lcand = left_span.candidate(mf)
        rcand = right_span.candidate(ntg)
        if lcand is None or rcand is None:
            continue
        left_span.commit(lcand)
        right_span.commit(rcand)
        chosen.append(P)
        X.append(f)
        Y.append(g)
        if len(chosen) == s:
            break
    if len(chosen) != s:
        raise RepairFailed({
            "reason": "projective grid exhausted at step %d of %d" % (len(chosen) + 1, s),
            "field": F.name(), "q": F.q, "n": n, "k": sym.k, "s": s,
            "normal_rank_so_far": base,
            "lags": [[F.to_str(x) for x in v] for v in sym.lags],
            "chosen_points": [(F.to_str(P[0]), F.to_str(P[1])) for P in chosen],
            "note": "contradicts algorithm(d) whenever |K| >= 2n-2; below the "
                    "threshold the refusal is intentional",
        })


    for f, g in zip(X, Y):
        for i in range(n):
            for j in range(n):
                E0[i][j] = F.add(E0[i][j], F.mul(f[i], g[j]))
    Xm = [[X[r][i] for r in range(s)] for i in range(n)]      
    Ym = [[Y[r][i] for r in range(s)] for i in range(n)]      

    
    if gauss_rank(F, E0) != s:
        raise RepairFailed({"reason": "rank(E0) != s, contradicting -03 (c)/(d)",
                            "field": F.name(), "n": n, "s": s})
    if F.is_zero(gauss_det(F, _add_into(F, A, E0))):
        raise RepairFailed({"reason": "det(a + E0) still identically zero",
                            "field": F.name(), "n": n, "s": s})
    if stats is not None:
        stats.update(argmax=F.to_str(x0), candidates_scanned=scanned,
                     candidates_accepted=s,
                     normal_rank_evaluations=2 * sym.k * n + 1)
    return E0, Xm, Ym
