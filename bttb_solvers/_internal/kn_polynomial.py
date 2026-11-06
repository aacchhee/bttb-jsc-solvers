# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""KN supplied-generator algebra.

Role: Shared. Khichane-Neiger-related scalar polynomial, matrix and degree-budget utilities, reused by both front-end and terminal components.

Entry points: Polynomial.

References: Khichane and Neiger (2026); see REFERENCES.md for the precise
approximant/kernel sources and preprint scope.
Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform


def count(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


class Polynomial:
    def __init__(self, field, stats=None):
        self.F, self.stats = field, stats
        self.mul_scalar = uniform.make_multiplier(field)

    def trim(self, a):
        a = list(a)
        while a and self.F.is_zero(a[-1]):
            a.pop()
        return a

    def add(self, a, b):
        F = self.F
        return self.trim([F.add(a[i] if i<len(a) else F.zero(),
                                b[i] if i<len(b) else F.zero())
                          for i in range(max(len(a),len(b)))])

    def scale(self, a, c):
        return self.trim([self.F.mul(x,c) for x in a])

    def shift(self, a, k):
        if type(k) is not int or k<0:
            raise ValueError('nonnegative polynomial shift required')
        return [self.F.zero() for _ in range(k)]+list(a) if a else []

    def mul(self, a, b):
        return self.trim(self.mul_scalar(a,b)) if a and b else []

    def inverse(self, value):
        if self.F.is_zero(value):
            raise ZeroDivisionError('KN zero denominator')
        return self.F.inv(value)

    def identity(self, n):
        return [[[self.F.one()] if i==j else [] for j in range(n)] for i in range(n)]

    @staticmethod
    def shape(A):
        rows, cols = len(A), len(A[0]) if A else 0
        if any(len(row)!=cols for row in A):
            raise ValueError('ragged polynomial matrix')
        return rows,cols

    @staticmethod
    def transpose(A):
        return [list(row) for row in zip(*A)]

    def to_degree(self, A, limit=None):
        rows,cols = self.shape(A)
        if limit is not None and (type(limit) is not int or limit<1):
            raise ValueError('positive integer conversion limit required')
        degree = max((len(a) for row in A for a in row),default=0)
        if limit is not None:

            
            degree = min(degree,limit)

        degree = max(1,degree)
        count(self.stats,'kn_copy_cells',degree*rows*cols)
        count(self.stats,'kn_degree_input_cells',degree*rows*cols)
        return [[[A[i][j][k] if k<len(A[i][j]) else self.F.zero()
                  for j in range(cols)] for i in range(rows)] for k in range(degree)]

    def from_degree(self, A, rows, cols):
        count(self.stats,'kn_copy_cells',len(A)*rows*cols)
        return [[self.trim([C[i][j] for C in A]) for j in range(cols)] for i in range(rows)]

    def pm_mul(self, A, B, limit=None):
        rows,inner = self.shape(A)
        binner,cols = self.shape(B)
        if inner!=binner:
            raise ValueError('incompatible polynomial matrix dimensions')
        if limit is not None and (type(limit) is not int or limit<0):
            raise ValueError('nonnegative integer coefficient limit required')
        if limit==0:
            return [[[] for _ in range(cols)] for _ in range(rows)]
        C = uniform.polynomial_matrix_product(self.F,self.to_degree(A,limit),
                self.to_degree(B,limit),self.mul_scalar,limit,stats=self.stats)
        return self.from_degree(C,rows,cols)

    def inverse_constant(self, A):
        n = len(A)
        if any(len(row)!=n for row in A):
            raise ValueError('square constant inverse required')
        if not n:
            return []
        result = uniform.gauss_jordan(self.F,A,stats=self.stats)
        if result['rank']!=n:
            raise ValueError('singular constant matrix')
        inv = self.inverse(result['d'])
        p = result['permutation']
        out = [[self.F.zero() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):

                out[i][p[j] if p is not None else j] = self.F.mul(inv,result['columns'][i][j])
        count(self.stats,'kn_copy_cells',n*n)
        return out

    def pm_basis(self, A, order, shift):
        if type(order) is not int or order<0 or len(shift)!=len(A) or any(type(s) is not int for s in shift):
            raise ValueError('integer order and one integer shift per row required')
        if order==0:
            return self.identity(len(A))
        result = uniform.pm_basis(self.F,self.to_degree(A,order),order,shift,
                                  self.mul_scalar,stats=self.stats)
        return self.from_degree(result['basis'],len(A),len(A))

    @staticmethod
    def shifted_degrees(A, s):
        return [max((len(a)-1+s[j] for j,a in enumerate(row) if a),default=-10**9)
                for row in A]
