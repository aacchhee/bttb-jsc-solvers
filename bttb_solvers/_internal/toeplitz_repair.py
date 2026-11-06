# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.banded_bttb.polynomial import ProductTree, trim
from bttb_solvers._internal.constant_profile import rank_profile_kernel
from bttb_solvers._internal.fast_scalar import FastScalar


class _ImageBasis:
    """Monotone image independence, same elimination as legacy repair.py.

    At most s pivots; O(s^2) per candidate/commit, never n-order rank probes.
    No element is hashed, compared by Python, coerced, or stored as a key.
    """
    def __init__(self, F, dimension):
        self.F, self.dimension, self.rows = F, dimension, {}

    def candidate(self, vector):
        F, v = self.F, list(vector)
        for pivot in sorted(self.rows):
            coefficient = v[pivot]
            if not F.is_zero(coefficient):
                v = [F.sub(x, F.mul(coefficient, y))
                     for x, y in zip(v, self.rows[pivot])]
        pivot = next((i for i, x in enumerate(v) if not F.is_zero(x)), None)
        if pivot is None:
            return None
        inverse = F.inv(v[pivot])
        return pivot, [F.mul(inverse, x) for x in v]

    def commit(self, candidate):
        F = self.F
        pivot, vector = candidate
        for old, row in list(self.rows.items()):
            coefficient = row[pivot]
            if not F.is_zero(coefficient):
                self.rows[old] = [F.sub(x, F.mul(coefficient, y))
                                  for x, y in zip(row, vector)]
        self.rows[pivot] = vector


def _dot(F, a, b):
    total = F.zero()
    for x, y in zip(a, b):
        total = F.add(total, F.mul(x, y))
    return total


def _projective_factors(F, n, index, labels):
    if index < 2:
        f, g = [F.zero() for _ in range(n)], [F.zero() for _ in range(n)]
        f[n-1 if index == 0 else 0] = F.one()
        g[0 if index == 0 else n-1] = F.one()
        return f, g
    powers = [F.one()]
    for _ in range(n-1):
        powers.append(F.mul(powers[-1], labels[index-2]))
    return list(reversed(powers)), powers


def repair_toeplitz_symbol(context, symbol, certified_minor, delta, *, stats=None):
    """Return ordered X,Y,E0 and H=z^k(a+E0), or finite-window refusal.

    context is the internally constructed n-aware catalog owned by the global
    attempt; symbol is F=z^k a in that EXACT working field. No embedding or
    auxiliary algebra construction happens here. Restart propagates unchanged.
    certified_minor is the accepted two-kernel record; delta must be the actual
    determinant of its selected minor, including normalization, not an oracle.
    """
    F, metadata = context.field, context.metadata
    n, k = metadata['n'], metadata['k']
    if type(n) is not int or type(k) is not int or not 1 <= k < n or 2*k >= n:
        raise ValueError('narrow integer dimensions 1<=k<n, 2k<n required')
    if metadata['M'] != 2*k*n+1 or len(context.labels) != metadata['M']:
        raise ValueError('owned n-aware label catalog required')
    if len(symbol) != n or any(len(row) != n for row in symbol):
        raise ValueError('square symbol of catalog dimension required')
    A = [[trim(F, list(a)) for a in row] for row in symbol]
    if any(len(a) > 2*k+1 for row in A for a in row):
        raise ValueError('symbol degree exceeds 2k')

    for i in range(1, n):
        for j in range(1, n):
            a, b = A[i][j], A[i-1][j-1]
            if len(a) != len(b) or any(not F.is_zero(F.sub(x, y)) for x, y in zip(a, b)):
                raise ValueError('symbol is not coefficientwise Toeplitz')
    rho = certified_minor['normal_rank']
    if type(rho) is not int or not 0 <= rho <= n or certified_minor.get('minor_certified_nonsingular') is not True:
        raise ValueError('certified maximum-rank minor record required')
    I, J = certified_minor['rows'], certified_minor['columns']
    if (len(I) != rho or len(J) != rho or len(set(I)) != rho or len(set(J)) != rho
            or any(type(i) is not int or not 0 <= i < n for i in I+J)):
        raise ValueError('invalid minor coordinate indices')
    actual = certified_minor['minor']
    if len(actual) != rho or any(len(row) != rho for row in actual):
        raise ValueError('minor shape mismatch')
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            a, b = trim(F, list(actual[ii][jj])), A[i][j]
            if len(a) != len(b) or any(not F.is_zero(F.sub(x, y)) for x, y in zip(a, b)):
                raise ValueError('minor record is not the selected actual source')
    s = n-rho
    counters = {} if stats is None else stats
    counters.update(constant_kernel_calls=0, projective_candidates=0,
                    accepted_nodes=0, scalar_multipoint_calls=0)

    if n*s > 2*k*rho:
        return {'status': 'PHYSICAL_SINGULAR_CERTIFICATE', 'n': n, 'k': k,
                'normal_rank': rho, 'nullity': s,
                'certificate': {'lhs_n_times_s': n*s, 'rhs_2k_times_rho': 2*k*rho,
                                'source': 'algorithm(f); certified normal rank premise'},
                'stats': counters}
    f = trim(F, list(delta))
    if not f or len(f)-1 > 2*k*rho:
        raise ValueError('actual nonzero minor determinant of degree<=2k*rho required')
    F.inv(f[-1])
    arithmetic = FastScalar(F)
    values = ProductTree(F, context.labels, arithmetic=arithmetic).evaluate(f)
    counters['scalar_multipoint_calls'] = 1
    point_id = next((i for i, value in enumerate(values) if not F.is_zero(value)), None)
    if point_id is None:
        raise ArithmeticError('nonzero determinant has no unit catalog evaluation')
    x0 = context.labels[point_id]
    F.inv(x0)
    F.inv(values[point_id])
    def evaluate(a):
        value = F.zero()
        for coefficient in reversed(a):
            value = F.add(F.mul(value, x0), coefficient)
        return value
    scale = F.one()
    inverse_x0 = F.inv(x0)
    for _ in range(k):
        scale = F.mul(scale, inverse_x0)
    at_point = [[F.mul(scale, evaluate(a)) for a in row] for row in A]
    X, Y = [[] for _ in range(n)], [[] for _ in range(n)]
    nodes = []
    if s:
        right = rank_profile_kernel(F, at_point, kernel=True, stats=counters)
        left = rank_profile_kernel(F, list(map(list, zip(*at_point))), kernel=True, stats=counters)
        counters['constant_kernel_calls'] = 2
        if right['rank'] != rho or left['rank'] != rho:
            raise ArithmeticError('certified normal rank/minor determinant premise inconsistent at x0')
        left_rows = list(map(list, zip(*left['kernel'])))
        right_rows = list(map(list, zip(*right['kernel'])))
        lb, rb = _ImageBasis(F, s), _ImageBasis(F, s)
        for index in range(2*n-1):
            fcol, gcol = _projective_factors(F, n, index, context.labels)
            l = lb.candidate([_dot(F, row, fcol) for row in left_rows])
            r = rb.candidate([_dot(F, row, gcol) for row in right_rows])
            counters['projective_candidates'] += 1
            if l is None or r is None:
                continue
            lb.commit(l); rb.commit(r)
            for i in range(n):
                X[i].append(fcol[i]); Y[i].append(gcol[i])
            nodes.append({'projective_index': index,
                          'kind': ('1:0' if index == 0 else '0:1' if index == 1 else '1:c'),
                          'label_id': None if index < 2 else context.label_ids[index-2]})
            if len(nodes) == s:
                break
        if len(nodes) != s:
            raise ArithmeticError('proved projective repair scan exhausted')
    counters['accepted_nodes'] = len(nodes)
    E0 = [[_dot(F, X[i], Y[j]) for j in range(n)] for i in range(n)]
    H = [[list(a)+[F.zero() for _ in range(max(0, k+1-len(a)))] for a in row] for row in A]
    for i in range(n):
        for j in range(n):
            H[i][j][k] = F.add(H[i][j][k], E0[i][j])
            H[i][j] = trim(F, H[i][j])
    return {'status': 'REPAIRED_SYMBOL', 'field': F, 'context': context,
            'n': n, 'k': k, 'normal_rank': rho, 'nullity': s,
            'x0': x0, 'point_id': context.label_ids[point_id], 'minor_value': values[point_id],
            'minor_values': values, 'at_point': at_point, 'X': X, 'Y': Y,
            'E0': E0, 'H': H, 'nodes': nodes, 'stats': counters,
            'premise': 'supplied certified maximum minor and its actual polynomial determinant'}
