# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured half-strip comparison.

Role: Trial/shared. Fixed-window scalar series operations, polynomial factor trimming and permutation signs for the array-based joint representation.

Entry points: SeriesOps; trim_factor.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier


class SeriesOps:
    def __init__(self, F):
        self.F, self.multiply = F, make_multiplier(F)
    def zero(self, N): return [self.F.zero() for _ in range(N)]
    def unit(self, N): return [self.F.one()] + self.zero(N-1)
    def add(self, a, b): return [self.F.add(x, y) for x, y in zip(a, b)]
    def sub(self, a, b): return [self.F.sub(x, y) for x, y in zip(a, b)]
    def neg(self, a): return [self.F.neg(x) for x in a]
    def mul(self, a, b, N):
        out = self.multiply(a, b)[:N]
        return out + self.zero(N-len(out))
    def inv(self, a, N):
        b = [self.F.inv(a[0])]
        while len(b) < N:
            t = min(2*len(b), N)
            error = self.neg(self.mul(a[:t], b, t))
            error[0] = self.F.add(error[0], self.F.add(self.F.one(), self.F.one()))
            b = self.mul(b, error, t)
        return b
    def valuation(self, a, threshold):
        for i in range(min(len(a), threshold)):
            if not self.F.is_zero(a[i]): return i
        return threshold
    def divide_shift(self, a, shift, N):
        if len(a) < shift+N:
            raise ValueError("division reads beyond known precision")
        if any(not self.F.is_zero(x) for x in a[:shift]):
            raise ArithmeticError("nonanalytic exact shift")
        return list(a[shift:shift+N])
    def delay(self, a, shift, N):
        if shift < 0: return self.divide_shift(a, -shift, N)
        needed = max(0, N-shift)
        if len(a) < needed: raise ValueError("delay reads beyond known precision")
        return (self.zero(min(shift, N)) + list(a[:needed]))[:N]
    def dot(self, a, b, N):
        result = self.zero(N)
        for x, y in zip(a, b): result = self.add(result, self.mul(x, y, N))
        return result
    def right(self, factor, matrix, N, *, inverse_transpose=False):
        r = len(matrix)
        return [[self.dot(row, [matrix[c][b] if inverse_transpose else matrix[b][c]
                               for b in range(r)], N)
                 for c in range(r)] for row in factor]


def trim_factor(factor, N):
    return [[list(poly[:N]) for poly in row] for row in factor]


def permutation_sign(F, old, new):
    if len(old) != len(new) or set(old) != set(new):
        raise ValueError("integer-ID permutation required")
    positions = {i: a for a, i in enumerate(old)}
    gather = [positions[i] for i in new]
    seen, cycles = set(), 0
    for i in range(len(gather)):
        if i in seen: continue
        cycles += 1
        j = i
        while j not in seen:
            seen.add(j)
            j = gather[j]
    return F.neg(F.one()) if (len(gather)-cycles) % 2 else F.one()
