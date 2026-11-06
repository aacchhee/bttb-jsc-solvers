# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata as dense_metadata
from bttb_solvers._internal.halfstrip_core.uniform_terminal import matrix_product
from bttb_solvers._internal.spatial_transforms import row_product
from bttb_solvers._internal.weighted_carrier import WeightedRingCarrier, _power


class WeightedCarrierActions:
    def __init__(self, carrier):
        if not isinstance(carrier, WeightedRingCarrier):
            raise ValueError('actual weighted ring carrier required')
        self.carrier, self.context, self.field = carrier, carrier.context, carrier.field
        self.n, self.k, self.c = carrier.n, carrier.k, carrier.c
        self.multiplier = carrier.service.mul_scalar

        
        self._factors = []
        for raw_g, raw_h in zip(carrier.g, carrier.h):
            g, h = self._rows(raw_g), self._rows(raw_h)
            self._factors.append((g, h, list(reversed(g)), list(reversed(h))))
        self.stats = {'ring_forward_actions': 0, 'ring_transpose_actions': 0,
                      'physical_forward_actions': 0, 'physical_transpose_actions': 0,
                      'bivariate_products': 0, 'largest_bivariate_input_cells': 0}

    def _rows(self, vector):
        n, F = self.n, self.field
        if len(vector) != n or any(len(poly) > n for poly in vector):
            raise ValueError('n ring-vector polynomials of degree below n required')
        return [list(poly)+[F.zero() for _ in range(n-len(poly))] for poly in vector]

    def _product(self, left, right):

        
        self.stats['bivariate_products'] += 1
        self.stats['largest_bivariate_input_cells'] = max(
            self.stats['largest_bivariate_input_cells'],
            sum(map(len, left))+sum(map(len, right)))
        return row_product(self.field, left, right, 2*self.n-1, self.multiplier)

    def _outer_fold(self, rows):
        F, n = self.field, self.n
        return [[F.add(poly[t] if t < len(poly) else F.zero(),
                       F.mul(self.c, poly[t+n]) if t+n < len(poly) else F.zero())
                 for t in range(n)] for poly in rows]

    def _cyclic(self, g, vector, transpose, reversed_g=None):
        F, n, full = self.field, self.n, 2*self.n-1
        if not transpose:
            product = self._product(g, vector)
            rows = [[F.add(product[i][d], product[i+n][d] if i+n < len(product) else F.zero())
                     for d in range(full)] for i in range(n)]
        else:
            product = self._product(vector, reversed_g if reversed_g is not None else list(reversed(g)))
            rows = [[F.add(product[n-1+i][d], product[i-1][d] if i else F.zero())
                     for d in range(full)] for i in range(n)]
        return self._outer_fold(rows)

    def _upper(self, h, vector, transpose, reversed_h=None):
        n = self.n
        if not transpose:
            product = self._product(vector, reversed_h if reversed_h is not None else list(reversed(h)))
            return self._outer_fold(product[n-1:2*n-1])
        return self._outer_fold(self._product(h, vector)[:n])

    def ring_action(self, vector, *, transpose=False):
        """Qv or ordinary Q^t v in A[z]/(z^n-c), exactly n coefficients/row."""
        F, n = self.field, self.n
        vector = self._rows(vector)
        out = [[F.zero() for _ in range(n)] for _ in range(n)]
        for g, h, reversed_g, reversed_h in self._factors:
            if transpose:
                value = self._upper(h, self._cyclic(g, vector, True, reversed_g), True)
            else:
                value = self._cyclic(g, self._upper(h, vector, False, reversed_h), False)
            out = [[F.add(a, b) for a, b in zip(x, y)] for x, y in zip(out, value)]
        key = 'ring_transpose_actions' if transpose else 'ring_forward_actions'
        self.stats[key] += 1
        return out

    def physical_action(self, rhs, *, transpose=False):
        """Complete C^-1 or literal C^-t action, flat outer-time-major input.

        Literal transpose reverses OUTER positions around ordinary Q^t(S_c).
        It does not substitute z^-1 in the quotient ring.
        """
        n = self.n
        if len(rhs) != n*n:
            raise ValueError('physical RHS must have n*n native entries')
        times = list(range(n-1, -1, -1)) if transpose else list(range(n))
        vector = [[rhs[t*n+i] for t in times] for i in range(n)]
        value = self.ring_action(vector, transpose=transpose)
        result = [value[i][t] for t in times for i in range(n)]
        key = 'physical_transpose_actions' if transpose else 'physical_forward_actions'
        self.stats[key] += 1
        return result


class ProjectedCarrier(WeightedCarrierActions):
    def __init__(self, carrier, X, Y):
        super().__init__(carrier)
        n, F = self.n, self.field
        if len(X) != n or len(Y) != n:
            raise ValueError('repair factors must have n rows')
        s = len(X[0])
        if s > n or any(len(row) != s for row in X) or any(len(row) != s for row in Y):
            raise ValueError('matching n-by-s repair factors with 0<=s<=n required')
        self.s, self.X, self.Y = s, [list(row) for row in X], [list(row) for row in Y]
        self.R = [[[F.zero() for _ in range(s)] for _ in range(n)] for _ in range(n)]
        self.L = [[[F.zero() for _ in range(n)] for _ in range(s)] for _ in range(n)]
        self.Z = [[[F.zero() for _ in range(s)] for _ in range(s)] for _ in range(n)]
        for a in range(s):
            right = self.ring_action([[self.X[i][a]] for i in range(n)])
            left = self.ring_action([[self.Y[i][a]] for i in range(n)], transpose=True)
            for t in range(n):
                for i in range(n):
                    self.R[t][i][a], self.L[t][a][i] = right[i][t], left[i][t]
        self.stats['repair_forward_column_actions'] = s
        self.stats['repair_transpose_column_actions'] = s
        self.stats['Z_dense_products'] = 0
        self.stats['Z_product_stats'] = {}
        if s:

            
            padded_Yt = [[self.Y[j][i] if i < s else F.zero() for j in range(n)]
                         for i in range(n)]
            for block in range(s):
                right = [[self.R[(block*n+j)//s][i][(block*n+j)%s] for j in range(n)]
                         for i in range(n)]
                product = matrix_product(F, padded_Yt, right, stats=self.stats['Z_product_stats'])
                for j in range(n):
                    t, b = divmod(block*n+j, s)
                    for a in range(s):
                        self.Z[t][a][b] = product[a][j]
                self.stats['Z_dense_products'] += 1
        self.metadata = {'n': n, 'k': self.k, 's': s,
                         'projection_coefficient_cells': 2*n*n*s+n*s*s,
                         'all_projection_residues': n,
                         'selected_inverse_residues': tuple(carrier.coefficients),
                         'source_association': 'same carrier object; actual X/Y and raw repair provenance are caller prerequisites',
                         'physical_transpose': 'outer reversal around ordinary ring transpose',
                         'cost_scope': 'soft O(n^2*s)+O(s*n^tau) setup; soft O(n^2) one physical action; narrow s<=2k gives soft O(k*n^tau)',
                         **dense_metadata()}

    def _weighted(self, sequence, lag):
        if type(lag) is not int:
            raise ValueError('integer weighted projection lag required')
        quotient, residue = divmod(lag, self.n)
        scalar = _power(self.field, self.carrier.c_inverse if quotient >= 0 else self.c,
                        abs(quotient))
        return [[self.field.mul(scalar, value) for value in row] for row in sequence[residue]]

    def weighted_R(self, lag):
        return self._weighted(self.R, lag)

    def weighted_L(self, lag):
        return self._weighted(self.L, lag)

    def weighted_Z(self, lag):
        return self._weighted(self.Z, lag)


def acquire_weighted_projections(carrier, X, Y):
    return ProjectedCarrier(carrier, X, Y)
