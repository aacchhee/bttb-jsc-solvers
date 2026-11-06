# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Field setup and common representation tools.

Role: Shared. Fast scalar-polynomial arithmetic used by label acquisition and physical front ends; reports nonunits to the owning restart loop.

Entry points: FastScalar; PolynomialNonUnit.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.banded_bttb.polynomial import PolynomialArithmetic, trim
from bttb_solvers._internal.halfstrip_core.uniform_terminal import make_multiplier


HGCD_LEAF_DEGREE = 16


class PolynomialNonUnit(ArithmeticError):
    """Nonconstant monic gcd obstructs polynomial modular inversion."""
    def __init__(self, gcd):
        self.gcd = list(gcd)
        super().__init__('polynomial is not a unit modulo supplied modulus')


class FastScalar(PolynomialArithmetic):
    def __init__(self, field):
        self.uniform_multiply = make_multiplier(field)
        super().__init__(field, multiply=lambda _field, a, b: self.uniform_multiply(a, b))
        self.hgcd_stats = {'calls': 0, 'recursive_calls': 0,
                           'bridge_divisions': 0, 'leaf_divisions': 0,
                           'outer_divisions': 0, 'maximum_leaf_degree': 0,
                           'recursive_child_degrees': [],
                           'outer_degree_drops': []}

    def scale(self, a, scalar):
        return trim(self.F, [self.F.mul(value, scalar) for value in a])

    def identity(self):
        return [[[self.F.one()], []], [[], [self.F.one()]]]

    def _matrix_product(self, left, right):
        return [[self.add(self.mul(left[i][0], right[0][j]),
                          self.mul(left[i][1], right[1][j]))
                 for j in range(2)] for i in range(2)]

    def apply_matrix(self, matrix, a, b):
        return tuple(self.add(self.mul(row[0], a), self.mul(row[1], b)) for row in matrix)

    def _step(self, quotient, remainder):

        
        step = [[[], [self.F.one()]],
                [[self.F.one()], self.scale(quotient, self.F.neg(self.F.one()))]]
        return step, remainder

    def _half_child(self, parent_degree, a, b):
        child_degree = len(a)-1
        self.hgcd_stats['recursive_child_degrees'].append((parent_degree, child_degree))
        if child_degree > parent_degree//2:
            raise ArithmeticError('Yap half-size recursive-degree invariant failed')
        return self._half_gcd(a, b)

    def half_gcd(self, a, b):
        """Matrix H with H(a,b)=(c,d), deg(d)<ceil(deg(a)/2).

        Requires deg(a)>deg(b); the zero b is allowed. Rows remain actual
        native-polynomial cofactors of the supplied pair.
        """
        a, b = trim(self.F, a), trim(self.F, b)
        if b and len(a) <= len(b):
            raise ValueError('half_gcd requires strictly decreasing degrees')
        return self._half_gcd(a, b)

    def _half_gcd(self, a, b):
        self.hgcd_stats['calls'] += 1
        degree = len(a)-1
        middle = (degree+1)//2
        if not b or len(b)-1 < middle:
            return self.identity()
        if degree <= HGCD_LEAF_DEGREE:
            self.hgcd_stats['maximum_leaf_degree'] = max(self.hgcd_stats['maximum_leaf_degree'], degree)
            result = self.identity()
            while b and len(b)-1 >= middle:
                quotient, remainder = self.divmod(a, b)
                step, remainder = self._step(quotient, remainder)
                result = self._matrix_product(step, result)
                a, b = b, remainder
                self.hgcd_stats['leaf_divisions'] += 1
            return result
        self.hgcd_stats['recursive_calls'] += 1
        first = self._half_child(degree, a[middle:], b[middle:])
        t, s = self.apply_matrix(first, a, b)
        if not s or len(s)-1 < middle:
            return first
        quotient, remainder = self.divmod(t, s)
        step, remainder = self._step(quotient, remainder)
        self.hgcd_stats['bridge_divisions'] += 1
        combined = self._matrix_product(step, first)
        if not remainder:
            return combined
        shift = 2*middle-(len(s)-1)
        if shift < 0:
            raise ArithmeticError('published half-GCD shift invariant failed')
        second = self._half_child(degree, s[shift:], remainder[shift:])
        return self._matrix_product(second, combined)

    def xgcd(self, a, b):
        """Return monic(g),u,v with u*a+v*b=g. Both zero gives([],[],[])."""
        a, b = trim(self.F, a), trim(self.F, b)
        transform = self.identity()
        while b:
            old_degree = max(len(a), len(b))-1

            
            if len(a) > len(b):
                half = self._half_gcd(a, b)
                a, b = self.apply_matrix(half, a, b)
                transform = self._matrix_product(half, transform)
                if b and len(b)-1 >= (old_degree+1)//2:
                    raise ArithmeticError('half-GCD failed its degree threshold')
            if not b:
                break
            quotient, remainder = self.divmod(a, b)
            step, remainder = self._step(quotient, remainder)
            transform = self._matrix_product(step, transform)
            a, b = b, remainder
            self.hgcd_stats['outer_divisions'] += 1
            self.hgcd_stats['outer_degree_drops'].append((old_degree, len(a)-1))
        if not a:
            return [], [], []
        scalar = self.F.inv(a[-1])
        return self.scale(a, scalar), self.scale(transform[0][0], scalar), self.scale(transform[0][1], scalar)

    def gcd(self, a, b):
        return self.xgcd(a, b)[0]

    def inverse_mod(self, value, modulus):
        modulus = trim(self.F, modulus)
        if len(modulus) < 2:
            raise ValueError('positive-degree polynomial modulus required')
        value = self.divmod(value, modulus)[1]
        gcd, inverse, _other = self.xgcd(value, modulus)
        if len(gcd) != 1:
            raise PolynomialNonUnit(gcd)
        return self.divmod(inverse, modulus)[1]
