# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from flint import fq_default_ctx, fq_default, fq_default_poly_ctx
from bttb_solvers._internal.banded_bttb.field import Field, ELEMENTS_GUARD, is_prime
from bttb_solvers._internal.banded_bttb.errors import ZeroDivisionInField


_CTX = {}


def _contexts(p, d):
    """One (fq_default_ctx, fq_default_poly_ctx) per (p, d), kept alive for the whole process.  python-flint 0.9.0
    segfaults when a field object holding contexts is reclaimed by the cyclic garbage collector while elements of that
    context are still alive (reproduced: seg3-style script in RESULTS.md); contexts are therefore never freed, and
    FqField has no reference cycles (no bound methods stored on the instance)."""
    c = _CTX.get((p, d))
    if c is None:
        ctx = fq_default_ctx(p, d)
        c = _CTX[(p, d)] = (ctx, fq_default_poly_ctx(ctx))
    return c


class FqField(Field):
    """GF(p^d), El = flint.fq_default.  Counting: `count_ops=False` mirrors the benchmark's lean
    `PrimeField(..., count_ops=False)`; `FqField(..., count_ops=True)` ticks the OpCounter like the checked fields.
    Native polynomial products: `FqFieldNative` (class-level hooks, as in PrimeField's lean flint hooks)."""

    def __init__(self, p, d, cnt=None, *, count_ops=False, native_poly=False):
        if not is_prime(p) or d < 1:
            raise ValueError('prime p and degree d >= 1 required')
        if native_poly and type(self) is FqField:
            raise ValueError('use FqFieldNative for native polynomial products')
        self.ctx, self.pctx = _contexts(p, d)
        self.p, self.e, self.q, self.cnt = p, d, p ** d, cnt
        self.modulus = self.ctx.modulus()
        self._zero, self._one = self.ctx.zero(), self.ctx.one()
        self.count_ops = count_ops
        if count_ops and cnt is None:
            raise ValueError('counted field needs an OpCounter')


    def coerce(self, x):
        if type(x) is fq_default:
            return x
        if isinstance(x, tuple):
            return self.ctx(list(x))
        return self.ctx(int(x))

    def zero(self):
        return self._zero

    def one(self):
        return self._one

    def add(self, a, b):
        if self.count_ops:
            self.cnt.c_add()
        return a + b

    def sub(self, a, b):
        if self.count_ops:
            self.cnt.c_add()
        return a - b

    def neg(self, a):
        if self.count_ops:
            self.cnt.c_add()
        return -a

    def mul(self, a, b):
        if self.count_ops:
            self.cnt.c_mul()
        return a * b

    def inv(self, a):
        a = self.coerce(a)
        if a.is_zero():
            raise ZeroDivisionInField('inverse of 0 in %s' % self.name())
        if self.count_ops:
            self.cnt.c_inv()
        return 1 / a

    def pow(self, a, k):
        if k < 0:
            return self.inv(a) ** (-k)
        return a ** k

    def eq(self, a, b):
        return a == b

    def is_zero(self, a):
        return a.is_zero()

    def from_int(self, i):
        c, t = [], int(i)
        for _ in range(self.e):
            c.append(t % self.p)
            t //= self.p
        return self.ctx(c)

    def to_int(self, a):
        v = 0
        for c in reversed(a.to_list()):
            v = v * self.p + int(c)
        return v

    def to_list(self, a):
        return [int(c) for c in a.to_list()]

    def in_prime_subfield(self, a):
        return all(int(c) == 0 for c in a.to_list()[1:])

    def to_str(self, a):
        return str(a)

    def elements(self):
        if self.q > ELEMENTS_GUARD:
            raise ValueError('elements() refused at q = %d > %d' % (self.q, ELEMENTS_GUARD))
        return (self.from_int(i) for i in range(self.q))

    def name(self):
        return 'GF(%d^%d) [fq_default]' % (self.p, self.e)


class FqFieldNative(FqField):
    """FqField whose polynomial products (make_multiplier leaf and >64-coefficient products) go to fq_default_poly:
    the same optional-adapter contract (`polynomial_leaf`, `polynomial_native`: exact full convolution) that the
    checked PrimeField exposes in its lean flint mode.  Hooks are class-level (no per-instance bound methods)."""

    def __init__(self, p, d, cnt=None, *, count_ops=False, native_poly=True):
        FqField.__init__(self, p, d, cnt, count_ops=count_ops, native_poly=False)

    def _poly_product(self, a, b):
        if not a or not b:
            return []
        c = (self.pctx(list(a)) * self.pctx(list(b))).coeffs()
        n = len(a) + len(b) - 1
        return c + [self._zero] * (n - len(c))

    def polynomial_leaf(self, a, b):
        return self._poly_product(a, b)

    def polynomial_native(self, a, b):
        return self._poly_product(a, b)
