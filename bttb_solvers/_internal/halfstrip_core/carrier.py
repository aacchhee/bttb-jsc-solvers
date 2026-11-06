# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from . import poly as hp
from .exact import (kernel, inverse, inverse_and_determinant, coefficient, transpose,
                      clean)


def _add(F, a, b, P):
    return hp.sa(F, a, b, 0, P)


def _neg(F, a):
    return {k: F.neg(v) for k, v in a.items()}


def _mul(F, a, b, P):
    return hp.sm(F, a, b, 0, P)


def _matvec(F, A, b, P):
    out = []
    for row in A:
        value = {}
        for x, y in zip(row, b):
            if not x or not y:
                continue
            value = _add(F, value, _mul(F, x, y, P), P)
        out.append(value)
    return out


def _lift_solve(F, T, R, b, P):
    """R=inv(T) mod z^m implies this result is inv(T)b mod z^(2m)."""
    y = _matvec(F, R, b, P)
    Ty = _matvec(F, T, y, P)
    residual = [hp.ss(F, u, v, 0, P) for u, v in zip(b, Ty)]
    correction = _matvec(F, R, residual, P)
    return [_add(F, u, v, P) for u, v in zip(y, correction)]


def toeplitz_inverse(F, T, P, *, constant_inverse=None):
    """Full inverse coefficients, O(n^3+n^2 M(P) log(P)) field work.

    For Z the downshift, ZT-TZ=e0*h^t+g*en^t. Thus
    ZR-RZ=-x*v^t-y*t^t, with x=R e0, y=R g,
    v=R^t h, t=R^t en. First row and column plus this commutator
    reconstruct every other entry. No corner of R is required to be a unit.
    """
    if P < 1:
        raise ValueError('positive precision required')
    n = len(T)

    
    R0 = inverse(F, coefficient(F, T, 0)) if constant_inverse is None else constant_inverse
    R = [[clean(F, {0: x}) for x in row] for row in R0]
    e0 = [{0: F.one()}] + [{} for _ in range(n-1)]
    en = [{} for _ in range(n-1)] + [{0: F.one()}]
    h = [_neg(F, T[0][j+1]) for j in range(n-1)] + [{}]
    g = [{}] + [dict(T[i-1][n-1]) for i in range(1, n)]
    TT = transpose(T)
    m = 1
    while m < P:
        target = min(2*m, P)
        RT = transpose(R)
        x = _lift_solve(F, T, R, e0, target)
        y = _lift_solve(F, T, R, g, target)
        row = _lift_solve(F, TT, RT, e0, target)
        v = _lift_solve(F, TT, RT, h, target)
        t = _lift_solve(F, TT, RT, en, target)
        out = [[{} for _ in range(n)] for _ in range(n)]
        out[0] = row
        for i in range(1, n):
            out[i][0] = x[i]
            for j in range(1, n):
                delta = _add(F, _mul(F, x[i], v[j-1], target),
                             _mul(F, y[i], t[j-1], target), target)
                out[i][j] = _add(F, out[i-1][j-1], delta, target)
        R, m = out, target
    return R


def _degree(F, a):
    return max((k for k, v in a.items() if not F.is_zero(v)), default=-1)


def _remainder(F, a, b, cache=None, *, divisor_degree=None):
    """Fast monic remainder; optional node-local reversed-inverse cache.

    A cache is [precision, inverse] for this fixed divisor over this exact
    working field. Geometric growth amortizes repeated input-degree changes.
    It must never be shared across divisors or field/restart attempts.
    """
    da = _degree(F, a)
    db = _degree(F, b) if divisor_degree is None else divisor_degree
    if db < 0:
        raise ZeroDivisionError('zero product-tree divisor')
    if da < db:
        return clean(F, a)
    if db == 0:
        return {}
    length = da-db+1
    ar = {da-k: v for k, v in a.items() if 0 <= da-k < length}
    br = {db-k: v for k, v in b.items() if 0 <= db-k < length}
    if cache is None:
        reciprocal = hp.sinv(F, br, length)
    else:
        if cache[0] < length:
            target = max(length, 2*cache[0])

            
            reverse = {db-k: v for k, v in b.items() if 0 <= k <= db}
            cache[:] = [target, hp.sinv(F, reverse, target)]
        reciprocal = cache[1]
    qr = _mul(F, ar, reciprocal, length)
    q = {length-1-k: v for k, v in qr.items()}
    product = hp.sm(F, b, q, 0, db)
    return hp.ss(F, a, product, 0, db)


class MomentEvaluator:
    """Product-tree multipoint evaluation over the guarded coefficient field."""
    def __init__(self, F, labels):
        self.F = F
        self.count = len(labels)
        def build(points):
            if len(points) == 1:
                return ({0: F.neg(points[0]), 1: F.one()}, None, None, [0, {}], 1)
            split = len(points)//2
            left, right = build(points[:split]), build(points[split:])
            return (hp.sm(F, left[0], right[0]), left, right, [0, {}], len(points))
        self.tree = build(list(labels)) if labels else None

    def evaluate(self, a):
        F = self.F
        if self.tree is None:
            return []
        def descend(node, value):
            value = _remainder(F, value, node[0], node[3], divisor_degree=node[4])
            if node[1] is None:
                return [value.get(0, F.zero())]
            return descend(node[1], value) + descend(node[2], value)
        return descend(self.tree, a)

    def series(self, entries, P):
        """Evaluate sum_i entries[i](z) X^i at each selected label."""
        out = [{} for _ in range(self.count)]
        for d in range(P):
            polynomial = {i: x[d] for i, x in enumerate(entries) if d in x}
            if not polynomial:
                continue
            values = self.evaluate(polynomial)
            for j, x in enumerate(values):
                if not self.F.is_zero(x):
                    out[j][d] = x
        return out


def _select_moments(F, ker, labels, n, q):
    """Incremental rank selection, O(n^2 q+n q^2), preserving label order."""
    selected, vectors, basis = [], [], {}
    for label in labels:
        v = [F.one()]
        for _ in range(1, n):
            v.append(F.mul(v[-1], label))
        c = []
        for j in range(q):
            value = F.zero()
            for i in range(n):
                value = F.add(value, F.mul(ker[i][j], v[i]))
            c.append(value)
        for pivot in sorted(basis):
            scale = c[pivot]
            if not F.is_zero(scale):
                c = [F.sub(x, F.mul(scale, y)) for x, y in zip(c, basis[pivot])]
        pivot = next((j for j, x in enumerate(c) if not F.is_zero(x)), None)
        if pivot is not None:
            unit = F.inv(c[pivot])
            basis[pivot] = [F.mul(x, unit) for x in c]
            selected.append(label)
            vectors.append(v)
        if len(selected) == q:
            break
    if len(selected) != q:
        raise ArithmeticError('moment rank selection failed')
    return selected, transpose(vectors) if q else [[] for _ in range(n)]


def carrier(F, blocks, labels, P):
    """Reference-compatible acquisition without cubic polynomial contractions."""
    if not blocks or not blocks[0] or P < 1:
        raise ValueError('nonempty blocks and positive precision required')
    n = len(blocks[0])
    for block in blocks:
        if len(block) != n or any(len(row) != n for row in block):
            raise ValueError('square blocks of one fixed order required')
        for i in range(1, n):
            for j in range(1, n):
                if not F.eq(block[i][j], block[i-1][j-1]):
                    raise ValueError('Toeplitz blocks required')
    ker = kernel(F, list(reversed(blocks[0])))
    q = len(ker[0]) if ker else 0
    selected, V = _select_moments(F, ker, labels, n, q)
    moments = [F.zero() for _ in range(2*n-1)]
    for label in selected:
        value = F.one()
        for d in range(2*n-1):
            moments[d] = F.add(moments[d], value)
            value = F.mul(value, label)
    H = [[{} for _ in range(n)] for _ in range(n)]
    for d, block in enumerate(blocks[:P]):
        for i in range(n):
            for j in range(n):
                value = F.add(block[i][j], moments[n-1-i+j]) if d == 0 else block[i][j]
                if not F.is_zero(value):
                    H[i][j][d] = value
    R0, H0_det = inverse_and_determinant(F, coefficient(F, H, 0))
    if F.is_zero(H0_det):
        raise ArithmeticError('constructed carrier has singular constant term')
    R = toeplitz_inverse(F, H, P, constant_inverse=R0)
    BA = [list(reversed(row)) for row in R]
    if q:
        evaluate = MomentEvaluator(F, selected)
        BV = [evaluate.series(row, P) for row in BA]
        contracted = transpose([evaluate.series(column, P) for column in transpose(BV)])
        S = [[hp.ss(F, {0: F.one()} if i == j else {}, contracted[i][j], 0, P)
              for j in range(q)] for i in range(q)]
    else:
        BV, S = [[] for _ in range(n)], []
    return {'n': n, 'q': q, 'V': V, 'selected_labels': selected, 'kernel': ker,
            'H': H, 'H0_det': H0_det, 'R': R, 'BA': BA, 'BV': BV, 'S': S,
            'acquisition': 'rank-two Toeplitz inverse Newton lift; fast moment product/remainder trees'}
