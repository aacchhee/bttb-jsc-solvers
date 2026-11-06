# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""KN supplied-generator algebra.

Role: Trial/shared. Solve a polynomial-matrix/vector problem modulo a supplied polynomial or z**n-c for carrier and projection operations.

Entry points: ModularVector.solve_mod.

References: Khichane and Neiger (2026); see REFERENCES.md for the precise
approximant/kernel sources and preprint scope.
Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.kn_services import Services
from bttb_solvers._internal.kn_polynomial import count
from bttb_solvers._internal.fast_scalar import FastScalar


DEPENDENCY_STATUS = 'SCALAR_checked_AND_MODULAR_checked_ACCEPTED'


class ModularVector(Services):
    """One rational vector modulo an arbitrary polynomial, no H(0) condition.

    Caller promises gcd(det(H),M)=1; this service does not compute det(H) or
    certify that promise. A full kernel's scalar denominator is inverted
    modulo M, and its original guard/polynomial-nonunit exceptions escape.
    checked accepts the actual Xi/input-volume ledger and banded specialization;
    no full raw solver or measured performance follows.
    """
    def __init__(self, field, stats=None):
        super().__init__(field, stats=stats)
        self.scalar = FastScalar(field)  

    def solve_mod(self, H, vector, modulus, *, column_bounds=None,
                  degree_budget=None):
        """Return H^-1 vector mod M under the caller's matrix-unit promise.

        degree_budget validates the H-column envelope m, not the augmented
        kernel shift sum Xi=sum(column_bounds)+max(0,deg(vector)). The general
        kernel cost includes Xi, even when the RHS degree is nearly deg(M).
        A small H-column budget does not make that kernel cost beta*M(deg M).
        """
        beta, cols = self.shape(H)
        if beta < 1 or cols != beta or len(vector) != beta:
            raise ValueError('positive square H and matching polynomial vector required')
        H = [[self.trim(a) for a in row] for row in H]
        vector = [self.trim(a) for a in vector]
        modulus = self.trim(modulus)
        degree = len(modulus)-1
        if degree < beta:
            raise ValueError('KN Lemma24 requires modulus degree at least matrix order')
        if any(len(a) > degree for a in vector):
            raise ValueError('RHS polynomial degree must be below modulus degree')
        actual = self.column_degrees(H)
        bounds = [max(0, d) for d in actual] if column_bounds is None else list(column_bounds)
        if len(bounds) != beta or any(type(s) is not int or s < max(0, d)
                                      for s, d in zip(bounds, actual)):
            raise ValueError('nonnegative column bounds must bound actual degrees')
        budget = max(beta, sum(bounds)) if degree_budget is None else degree_budget
        if type(budget) is not int or budget < max(beta, sum(bounds)):
            raise ValueError('degree budget must cover column bounds and matrix order')
        augmented = [list(row)+[self.scale(value, self.F.neg(self.F.one()))]
                     for row, value in zip(H, vector)]
        rhs_bound = max(0, max((len(a)-1 for a in vector), default=-1))
        augmented_shift_sum = sum(bounds)+rhs_bound
        basis, _ = self.minimal_nullspace_basis(augmented, bounds+[rhs_bound])
        if len(basis) != beta+1 or any(len(row) != 1 for row in basis):
            raise ValueError('promised nonsingular polynomial H did not give a rank-one full kernel')
        numerator = [self.trim(row[0]) for row in basis[:-1]]
        denominator = self.trim(basis[-1][0])
        if not denominator:
            raise ValueError('polynomial H is singular: kernel denominator is zero')
        inverse = self.scalar.inverse_mod(denominator, modulus)
        result = [self.scalar.divmod(self.mul(a, inverse), modulus)[1] for a in numerator]
        count(self.stats, 'kn_arbitrary_modulus_vector_calls')
        count(self.stats, 'kn_modular_component_products', beta)
        count(self.stats, 'kn_modular_scalar_inversions')
        count(self.stats, 'kn_modular_augmented_shift_sum', augmented_shift_sum)
        count(self.stats, 'kn_modular_input_coefficient_cells',
              sum(len(a) for row in H for a in row)+sum(map(len, vector))+len(modulus))
        return result

    def solve_zn_minus_c(self, H, vector, degree, c, *, column_bounds=None,
                         degree_budget=None):
        """Literal z^degree-c specialization, including inseparable moduli."""
        if type(degree) is not int or degree < 1:
            raise ValueError('positive integer modulus degree required')
        modulus = [self.F.neg(c)]+[self.F.zero() for _ in range(degree-1)]+[self.F.one()]
        return self.solve_mod(H, vector, modulus, column_bounds=column_bounds,
                              degree_budget=degree_budget)
