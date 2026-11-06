# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.kn_modular_vector import ModularVector


def _power(F, value, exponent):
    result = F.one()
    while exponent:
        if exponent & 1:
            result = F.mul(result, value)
        exponent //= 2
        if exponent:
            value = F.mul(value, value)
    return result


class WeightedRingCarrier:
    def __init__(self, H, k, chart):
        self.context = chart['context']
        self.field = self.context.field
        self.n = len(H)
        self.k = k
        n, F = self.n, self.field
        if type(k) is not int or k < 1 or n < 3 or not 2*k < n:
            raise ValueError('narrow regime n>=3 and 1<=2k<n required')
        if (self.context.metadata.get('n') != n or self.context.metadata.get('k') != k
                or chart.get('field') is not F or chart.get('status') != 'UNIT_CHART'):
            raise ValueError('chart and carrier must use same attempt/dimensions')
        candidate = chart['candidate_id']
        if type(candidate) is not int or not 0 <= candidate < len(self.context.labels):
            raise ValueError('invalid internal chart catalog ID')
        self.c = chart['c']
        if not F.is_zero(F.sub(self.c, self.context.labels[candidate])):
            raise ValueError('chart value differs from internally constructed catalog')
        self.c_inverse = F.inv(self.c)
        self.modulus = [F.neg(self.c)]+[F.zero() for _ in range(n-1)]+[F.one()]
        self.stats = {'selected_coefficient_products': 0, 'selected_stein_additions': 0,
                      'selected_residues': 0}
        self.service = ModularVector(F, self.stats)
        if any(len(row) != n for row in H):
            raise ValueError('square polynomial H required')
        H = [[self.service.trim(list(value)) for value in row] for row in H]
        if any(len(value) > 2*k+1 for row in H for value in row):
            raise ValueError('H degree exceeds 2k')
        for i in range(1, n):
            for j in range(1, n):
                if self.service.scalar.sub(H[i][j], H[i-1][j-1]):
                    raise ValueError('H coefficients must be spatial Toeplitz')

        rH = [list(value) for value in H[-1]]
        dH = [self.service.scalar.sub(H[i][-1], H[i+1][0]) for i in range(n-1)]+[[]]
        shifted_last = [[] for _ in range(n)]
        shifted_first = [[] for _ in range(n)]
        shifted_last[-1] = [F.zero() for _ in range(k)]+[F.one()]
        shifted_first[0] = [F.zero() for _ in range(k)]+[F.one()]
        cyclic_rH = [rH[-1]]+rH[:-1]
        transposed_H = [list(row) for row in zip(*H)]
        bounds = [2*k]*n
        def solve(matrix, vector):
            return self.service.solve_mod(matrix, vector, self.modulus,
                                          column_bounds=bounds, degree_budget=2*k*n)
        alpha = solve(H, shifted_last)
        gamma = solve(H, dH)
        beta = solve(transposed_H, cyclic_rH)
        delta = solve(transposed_H, shifted_first)
        self.g = ([alpha[-1]]+alpha[:-1], [gamma[-1]]+gamma[:-1])
        self.h = (beta, delta)
        self.coefficients = {}

        for lag in range(-2*k, 2*k):
            residue = lag % n
            if residue not in self.coefficients:
                self.coefficients[residue] = self._coefficient(residue)
        self.stats['selected_residues'] = len(self.coefficients)
        self.metadata = {'n': n, 'k': k, 'requested_lags': list(range(-2*k, 2*k)),
                         'selected_residues': list(self.coefficients),
                         'ring_vector_count': 4, 'modular_vector_calls': 4,
                         'H_column_budget': 2*k*n, 'RHS_degree_bound': 2*k,
                         'source_scope': 'supplied repaired H and certified matrix-unit chart; no determinant verification',
                         'transpose_scope': 'ordinary ring transpose differs from literal weighted physical transpose',
                         'deferred': 'ring actions, repair projections, merged correction and global restart/descent'}

    def _coefficient(self, residue):
        F, n = self.field, self.n
        displacement = [[F.zero() for _ in range(n)] for _ in range(n)]
        for g, h in zip(self.g, self.h):
            left = [[polynomial[j] if j < len(polynomial) else F.zero()
                     for j in range(n)] for polynomial in g]

            
            right = []
            for j in range(n):
                index = (residue-j) % n
                row = [polynomial[index] if index < len(polynomial) else F.zero() for polynomial in h]
                if j > residue:
                    row = [F.mul(self.c, value) for value in row]
                right.append(row)
            product = uniform.matrix_product(F, left, right, stats=self.stats)
            self.stats['selected_coefficient_products'] += 1
            displacement = [[F.add(a, b) for a, b in zip(x, y)]
                            for x, y in zip(displacement, product)]
        coefficient = [[F.zero() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            coefficient[i][0] = displacement[i][0]
        for j in range(1, n):
            for i in range(n):
                coefficient[i][j] = F.add(displacement[i][j], coefficient[(i-1) % n][j-1])
                self.stats['selected_stein_additions'] += 1
        return coefficient

    def coefficient(self, residue, *, ordinary_transpose=False):
        if type(residue) is not int or not 0 <= residue < self.n:
            raise ValueError('canonical coefficient residue required')
        if residue not in self.coefficients:
            raise KeyError('coefficient was not among selected residues')
        matrix = self.coefficients[residue]
        return [list(row) for row in zip(*matrix)] if ordinary_transpose else [list(row) for row in matrix]

    def weighted_channel(self, lag, *, ordinary_transpose=False):
        """K(d)=c^(-floor(d/n))*G[d mod n]; Python // is true negative floor."""
        if type(lag) is not int:
            raise ValueError('integer physical lag required')
        quotient, residue = divmod(lag, self.n)
        factor = _power(self.field, self.c_inverse if quotient >= 0 else self.c, abs(quotient))
        matrix = self.coefficient(residue, ordinary_transpose=ordinary_transpose)
        return [[self.field.mul(factor, value) for value in row] for row in matrix]

    def physical_block(self, row, column, *, transposed=False):
        """Selected C^-1 block; literal C^-t reverses lag AND inner transpose.

        This is a block lookup only; no full physical matrix/action is formed.
        Missing selected residues are refused instead of acquired on demand.
        """
        if (type(row) is not int or type(column) is not int
                or not 0 <= row < self.n or not 0 <= column < self.n):
            raise ValueError('physical outer block indices out of range')
        return self.weighted_channel(column-row if transposed else row-column,
                                     ordinary_transpose=transposed)


def acquire_weighted_carrier(H, k, chart):
    return WeightedRingCarrier(H, k, chart)
