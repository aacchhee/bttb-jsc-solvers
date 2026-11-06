# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial/shared. Assemble and expose the weighted wrap/repair correction and its generator-compatible maps.

Entry points: WeightedCorrection; acquire_weighted_correction.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform


class WeightedCorrection:
    def __init__(self, raw, projected):
        self.projected = projected
        self.carrier = projected.carrier
        F, n, k, s = self.carrier.field, self.carrier.n, self.carrier.k, projected.s
        self.field, self.n, self.k, self.s = F, n, k, s
        self.c, self.c_inverse = self.carrier.c, self.carrier.c_inverse
        ell, r = 2*k, 2*k*n+n*s
        self.ell, self.order = ell, r
        if not 0 <= s <= n or n*s > 2*k*(n-s):
            raise ValueError('paid narrow normal-rank repair dimension inequality required')
        if any((a, b) not in raw for a in range(-k, k+1) for b in range(1-n, n)):
            raise ValueError('complete original narrow Toeplitz lag input required')
        self.stats = {'matrix_polynomial_products': 0, 'interior_window_batches': 0,
                      'interior_output_windows': 0, 'boundary_interval_products': 0}
        self.multiply = uniform.make_multiplier(F)
        self.positions = list(range(k))+list(range(n-k, n))
        K, L, R, Z = (self.carrier.weighted_channel, projected.weighted_L,
                       projected.weighted_R, projected.weighted_Z)
        zero = self.zero
        def scale(value, matrix):
            return [[F.mul(value, x) for x in row] for row in matrix]
        self.u = {}
        for d in range(1-ell, ell):
            if d <= -k:
                lag, factor = d+ell, self.c
            elif d >= k:
                lag, factor = d-ell, self.c_inverse
            else:
                continue
            self.u[d] = scale(factor, [[raw[lag, i-j] for j in range(n)] for i in range(n)])
        def u(d): return self.u.get(d, zero(n, n))
        self.a = [self.sub(scale(self.c, u(ell-1-q)), u(-q-1)) if q < ell-1 else zero(n,n)
                  for q in range(ell)]
        self.b = [zero(n,n)]+[self.sub(u(p-ell), scale(self.c, u(p))) for p in range(1,ell)]
        self.v = [u(q) for q in range(ell)]
        Qh = self.stack([K(p) for p in self.positions]+[L(i) for i in range(n)])
        Qb, Qv = self.q_product(self.b), self.q_product(self.v)
        D = self.stack([self.sub(K(p-k), K(p-(n-k))) for p in self.positions]
                       +[self.sub(L(i-k), L(i-(n-k))) for i in range(n)])
        krow = [self.sub(K(k-1-q), K(n-k-1-q)) for q in self.positions]
        erb = self.hjoin(self.row_u(krow)+[self.sub(R(k-1-j), R(n-k-1-j)) for j in range(n)])
        el = zero(r,n)
        for i in range(n): el[k*n+i][i] = F.one()
        self.Gc = self.hjoin([self.neg(el), self.neg(D), self.neg(Qh), self.neg(Qb)])
        tail = zero(n,n*s)
        hz = [zero(n,n) for _ in range(ell)]
        hz[-1] = self.identity(n)
        self.HcT = self.stack([erb, self.hjoin([u(k-1-q) for q in range(ell)]+[tail]),
                              self.hjoin(self.a+[tail]), self.hjoin(hz+[tail])])
        hu = self.row_u([K(n-1-q) for q in self.positions])
        fu = self.row_u([K(-q) for q in self.positions])
        ju = self.row_u([L(n-1-q) for q in self.positions]) if s else []
        end_boundary = self.hjoin([self.sub(self.identity(n) if q==ell-1 else zero(n,n), hu[q])
                                  for q in range(ell)]+[self.neg(R(n-1-j)) for j in range(n)])
        end_repair = (self.hjoin([self.neg(block) for block in ju]
                         +[self.sub(self.identity(s) if j==n-1 else zero(s,s), Z(n-1-j)) for j in range(n)])
                      if s else [])
        start_boundary = self.neg(Qv)
        for i in range(n): start_boundary[i][i] = F.add(start_boundary[i][i], F.one())
        start_repair = self.stack([self.neg(R(p)) for p in self.positions]
                         +[self.sub(self.identity(s) if i==0 else zero(s,s), Z(i)) for i in range(n)])
        self.FIRST = self.hjoin([self.sub(self.identity(n) if q==0 else zero(n,n), fu[q])
                                for q in range(ell)]+[self.neg(R(-j)) for j in range(n)])

        self.permutation = [p*n+a for a in range(n) for p in range(ell)]
        self.permutation += [ell*n+i*s+b for b in range(s) for i in range(n)]
        pi = self.permutation
        self.REND = [[row[j] for j in pi] for row in end_boundary+end_repair]
        cstart = self.hjoin([start_boundary,start_repair])
        self.CSTART = [list(cstart[i]) for i in pi]
        self.Gstar = [list(self.Gc[i]) for i in pi]
        self.HstarT = [[row[j] for j in pi] for row in self.HcT]
        starts = [a*ell for a in range(n)]+[ell*n+b*n for b in range(s)]
        ends = [(a+1)*ell-1 for a in range(n)]+[ell*n+(b+1)*n-1 for b in range(s)]
        self.starts, self.ends = starts, ends
        h = n+s
        for j in range(h):
            column = [F.zero() for _ in range(r)]
            if j < h-1:
                column[starts[j+1]] = F.one()
                column[starts[j]] = F.neg(self.c)
            else:
                column[starts[0]] = self.c
                column[starts[-1]] = F.sub(column[starts[-1]], self.c)
            self.append_pair(column, self.REND[j])
        for j in range(h):
            column = [F.mul(self.c,self.CSTART[i][j]) for i in range(r)]
            if j < h-1:
                column = [F.sub(value,self.CSTART[i][j+1]) for i,value in enumerate(column)]
            row = [F.zero() for _ in range(r)]; row[ends[j]] = F.one()
            self.append_pair(column,row)
        self.width = 6*n+2*s

        shifted = [list(self.Gstar[i+1]) for i in range(r-1)]
        shifted.append([F.mul(self.c_inverse,x) for x in self.Gstar[0]])
        self.G0 = list(reversed(shifted))
        self.H0T = [list(reversed(row)) for row in self.HstarT]

        self.first_row = [self.FIRST[0][j] for j in pi]
        right = [F.zero()]+list(reversed(self.first_row))[:-1]
        for i in range(r): self.G0[i].append(self.c_inverse if i==0 else F.zero())
        self.H0T.append(right)
        self.H0 = [list(row) for row in zip(*self.H0T)]
        self.metadata = {'order':r,'cyclic_width':4*n,'scalar_width':self.width,
                         'stein_width':self.width+1,'degree_budget':r,
                         'source_scope':'same repaired carrier/projections and actual original raw lags',
                         'solver_scope':'explicit factors only; no backend or physical recovery'}

    def zero(self, rows, cols):
        return [[self.field.zero() for _ in range(cols)] for _ in range(rows)]

    def identity(self, n):
        return [[self.field.one() if i==j else self.field.zero() for j in range(n)] for i in range(n)]

    def sub(self, A, B):
        return [[self.field.sub(a,b) for a,b in zip(x,y)] for x,y in zip(A,B)]

    def neg(self, A):
        return [[self.field.neg(a) for a in row] for row in A]

    @staticmethod
    def stack(blocks): return [list(row) for block in blocks for row in block]

    @staticmethod
    def hjoin(blocks):
        if not blocks: return []
        return [[value for block in blocks for value in block[i]] for i in range(len(blocks[0]))]

    def product(self, A, B):
        self.stats['matrix_polynomial_products'] += 1
        result = uniform.polynomial_matrix_product(self.field,A,B,self.multiply,stats=self.stats)
        return result

    def coefficient(self, polynomial, degree, rows=None):
        rows = self.n if rows is None else rows
        return polynomial[degree][:rows] if degree < len(polynomial) else self.zero(rows,self.n)

    def row_u(self, row):
        """Short block row times U, multiplication order W_p u_(p-q)."""
        n, ell = self.n, self.ell
        rows = len(row[0])
        padded = [block+self.zero(n-rows,n) for block in row]
        right = [self.u.get(-h,self.zero(n,n)) for h in range(1-ell,ell)]
        result = self.product(padded,right)
        return [self.coefficient(result,q+ell-1,rows) for q in range(ell)]

    def q_product(self, values):
        """Four short corners plus row-packed interior windows; no dense U."""
        F,n,k,s = self.field,self.n,self.k,self.s
        K,L = self.carrier.weighted_channel,self.projected.weighted_L
        offsets = [0,n-k]
        boundary = [self.zero(n,n) for _ in range(2*k)]
        interior = [self.zero(s,n) for _ in range(n)]
        for b,ob in enumerate(offsets):
            right = values[b*k:(b+1)*k]
            for a,oa in enumerate(offsets):
                left = [K(oa-ob+d) for d in range(1-k,k)]
                result = self.product(left,right)
                self.stats['boundary_interval_products'] += 1
                for i in range(k):
                    block = self.coefficient(result,i+k-1)
                    boundary[a*k+i] = [[F.add(x,y) for x,y in zip(u,v)]
                                        for u,v in zip(boundary[a*k+i],block)]
            if not s: continue
            windows = [(a,min(k,n-a)) for a in range(0,n,k)]
            batch_size = n//s
            for first in range(0,len(windows),batch_size):
                batch = windows[first:first+batch_size]
                degree_count = max(d+k-1 for a,d in batch)
                left = []
                for t in range(degree_count):
                    coefficient = []
                    for a,d in batch:
                        coefficient += L(a-k+1+t-ob) if t < d+k-1 else self.zero(s,n)
                    coefficient += self.zero(n-len(coefficient),n)
                    left.append(coefficient)
                result = self.product(left,right)
                self.stats['interior_window_batches'] += 1
                self.stats['interior_output_windows'] += len(batch)
                for group,(a,d) in enumerate(batch):
                    for i in range(d):
                        block = self.coefficient(result,k-1+i)[group*s:(group+1)*s]
                        interior[a+i] = [[F.add(x,y) for x,y in zip(u,v)]
                                         for u,v in zip(interior[a+i],block)]
        return self.stack(boundary+interior)

    def append_pair(self, column, row):
        for i,value in enumerate(column): self.Gstar[i].append(value)
        self.HstarT.append(list(row))


def acquire_weighted_correction(raw, projected):
    return WeightedCorrection(raw, projected)
