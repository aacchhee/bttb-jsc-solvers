# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""BFM local-front implementation.

Role: Main/shared. Guarded row-profile selection and inverse maps for normalized boundary equations.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.guarded_boundary.boundary import RankFailure

from .kernel import matmul, add_matrix, sub_matrix, neg_matrix


class SelectionFailure(RankFailure):
    """A scalar base case found no nonzero entry: A lacks full column rank."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = dict(context or {})


def _bump(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _bump_max(stats, key, value):
    if stats is not None and value > stats.get(key, 0):
        stats[key] = value


def select_rows(F, A, *, stats=None, context=None):
    """A: r x q list of rows, r >= q >= 1, full column rank promised.

    Returns (P, Binv): P is the ordered list of q selected row indices and
    Binv = (A[P,:])^-1 (q x q).  Raises SelectionFailure on rank deficiency.
    """
    r = len(A)
    q = len(A[0]) if A else 0
    if q < 1 or r < q:
        raise ValueError("select_rows needs r >= q >= 1")
    ctx = dict(context or {})
    return _select(F, [list(row) for row in A], stats, ctx, 0, 0)


def _select(F, A, stats, ctx, depth, col0):
    r = len(A)
    q = len(A[0])
    _bump_max(stats, "select_max_depth", depth)
    if q == 1:
        for i in range(r):
            if not F.is_zero(A[i][0]):
                _bump(stats, "select_inv_calls")
                _bump(stats, "select_base_cases")
                if i > 0:
                    _bump(stats, "base_scan_offset")
                return [i], [[F.inv(A[i][0])]]
        failure = dict(ctx)
        failure.update(column=col0, depth=depth, rows=r)
        raise SelectionFailure("no nonzero entry in a base-case column "
                               "(full column rank fails)", failure)
    qL = q // 2
    AL = [row[:qL] for row in A]
    AR = [row[qL:] for row in A]
    P1, B1i = _select(F, AL, stats, ctx, depth + 1, col0)
    inP1 = set(P1)
    J = [i for i in range(r) if i not in inP1]
    U = matmul(F, B1i, [AR[i] for i in P1], stats=stats)
    ALJ = [AL[i] for i in J]
    Cs = sub_matrix(F, [AR[i] for i in J], matmul(F, ALJ, U, stats=stats))
    P2, Si = _select(F, Cs, stats, ctx, depth + 1, col0 + qL)
    Fs = [ALJ[i] for i in P2]
    X = matmul(F, Fs, B1i, stats=stats)          
    Y = matmul(F, U, Si, stats=stats)            
    top_left = add_matrix(F, B1i, matmul(F, Y, X, stats=stats))
    top_right = neg_matrix(F, Y)
    bottom_left = neg_matrix(F, matmul(F, Si, X, stats=stats))
    Binv = [a + b for a, b in zip(top_left, top_right)] + \
           [a + b for a, b in zip(bottom_left, Si)]
    return P1 + [J[i] for i in P2], Binv


def build_q(F, A, P, Binv):
    """Dense Q (r x r) with Q A = [I_q; 0]; returns (Q, rest).

    rest = rows not in P in increasing original index.  Q = [[Binv,0],[-D Binv,I]] Pi.
    """
    r = len(A)
    q = len(P)
    inP = set(P)
    rest = [i for i in range(r) if i not in inP]
    D = [A[i] for i in rest]
    DB = neg_matrix(F, matmul(F, D, Binv, stats=None)) if rest else []
    Q = [[F.zero() for _ in range(r)] for _ in range(r)]
    for c in range(q):
        col = P[c]
        for i in range(q):
            Q[i][col] = Binv[i][c]
        for i in range(len(rest)):
            Q[q + i][col] = DB[i][c]
    for i, orig in enumerate(rest):
        Q[q + i][orig] = F.one()
    return Q, rest


def invert_square(F, S, *, stats=None, context=None):
    """Inverse of a square nonsingular S through the selection primitive.

    (P,B) = select_rows(S); S^-1[:, P[c]] = B[:, c]  (the explicitly paid row permutation).
    """
    d = len(S)
    P, B = select_rows(F, S, stats=stats, context=context)
    inverse = [[None] * d for _ in range(d)]
    for c in range(d):
        for i in range(d):
            inverse[i][P[c]] = B[i][c]
    return inverse
