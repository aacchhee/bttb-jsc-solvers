# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .errors import (NotAPrimePower, ZeroDivisionInField,
                     CharacteristicDividesOrder, NoRootOfUnity, NeedsGenerator)

ELEMENTS_GUARD = 1 << 16          


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def prime_power(q):
    """(p, e) with q = p**e, else raise NotAPrimePower."""
    if q < 2:
        raise NotAPrimePower(q)
    p = 2
    while p * p <= q:
        if q % p == 0:
            break
        p += 1
    else:
        p = q
    if not is_prime(p):
        raise NotAPrimePower(q)
    e, t = 0, q
    while t % p == 0:
        t //= p
        e += 1
    if t != 1:
        raise NotAPrimePower(q)
    return p, e


TRIAL_DIVISION_BOUND = 10 ** 7    


def factor_small(x, bound=TRIAL_DIVISION_BOUND):
    """Distinct prime factors of x by trial division up to `bound`.
    raises NeedsGenerator if a cofactor survives - the factorisation of q-1 is
    exactly the CONDITIONAL preprocessing of T05 Open item 4."""
    fs, d, t = [], 2, x
    while d <= bound and d * d <= t:
        if t % d == 0:
            fs.append(d)
            while t % d == 0:
                t //= d
        d += 1 if d == 2 else 2
    if t > 1:
        if t <= bound or is_prime(t):
            fs.append(t)
        else:
            raise NeedsGenerator(
                "cofactor %d of %d not factored below trial-division bound %d"
                % (t, x, bound))
    return fs


class Field(object):
    """Abstract; never instantiated.  Subclasses set p, e, q and cnt."""

    p = e = q = None
    cnt = None
    modulus = None


    def zero(self):
        raise NotImplementedError

    def one(self):
        raise NotImplementedError

    def add(self, a, b):
        raise NotImplementedError

    def sub(self, a, b):
        raise NotImplementedError

    def neg(self, a):
        raise NotImplementedError

    def mul(self, a, b):
        raise NotImplementedError

    def inv(self, a):
        raise NotImplementedError

    def div(self, a, b):
        return self.mul(a, self.inv(b))       

    def pow(self, a, k):
        """Square and multiply.  Counts exactly the muls it performs, because
        it performs them through self.mul and nothing else."""
        if k < 0:
            return self.pow(self.inv(a), -k)
        r, b = self.one(), a
        while k:
            if k & 1:
                r = self.mul(r, b)
            k >>= 1
            if k:
                b = self.mul(b, b)
        return r


    def eq(self, a, b):
        return a == b

    def is_zero(self, a):
        return a == self.zero()

    def from_int(self, i):
        raise NotImplementedError

    def to_str(self, a):
        raise NotImplementedError

    def elements(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class PrimeField(Field):
    """F_p, El = int in [0, p)."""

    def __init__(self, p, cnt=None, *, count_ops=True):
        if not is_prime(p):
            raise NotAPrimePower(p)
        self.p, self.e, self.q, self.cnt = p, 1, p, cnt
        self.count_ops = count_ops
        if not count_ops:
            for name in ("add", "sub", "neg", "mul", "inv"):
                setattr(self, name, getattr(self, "_lean_" + name))
            from bttb_solvers._internal.halfstrip_core.flint_backend import selection
            if selection() == 'flint':

                

                self.polynomial_leaf = self._flint_polynomial_leaf
                self.polynomial_native = self._flint_polynomial_native

    def zero(self):
        return 0

    def one(self):
        return 1 % self.p

    def add(self, a, b):
        self.cnt.c_add()
        return (a + b) % self.p

    def sub(self, a, b):
        self.cnt.c_add()
        return (a - b) % self.p

    def neg(self, a):
        self.cnt.c_add()
        return (-a) % self.p

    def mul(self, a, b):
        self.cnt.c_mul()
        return (a * b) % self.p

    def inv(self, a):
        if a % self.p == 0:
            raise ZeroDivisionInField("inverse of 0 in F_%d" % self.p)
        self.cnt.c_inv()
        return pow(a, self.p - 2, self.p)

    def _flint_polynomial_leaf(self, a, b):
        if not a or not b:
            return []
        if max(len(a),len(b)) > 64:
            raise ValueError('fixed polynomial leaf exceeded')
        from bttb_solvers._internal.halfstrip_core.flint_backend import product
        result = product(self,a,b)
        if result is not None:
            return result
        out=[self.zero()]*(len(a)+len(b)-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b):
                out[i+j]=self.add(out[i+j],self.mul(x,y))
        return out

    def _flint_polynomial_native(self, a, b):
        if not a or not b:
            return []

        

        from bttb_solvers._internal.halfstrip.allfield_poly import NATIVE_ACCELERATOR_MAX
        if max(len(a),len(b)) > NATIVE_ACCELERATOR_MAX:
            return None
        from bttb_solvers._internal.halfstrip_core.flint_backend import product
        return product(self,a,b)

    def _lean_add(self, a, b): return (a+b) % self.p
    def _lean_sub(self, a, b): return (a-b) % self.p
    def _lean_neg(self, a): return -a % self.p
    def _lean_mul(self, a, b): return a*b % self.p
    def _lean_inv(self, a):
        if a % self.p == 0:
            raise ZeroDivisionInField("inverse of 0 in F_%d" % self.p)
        return pow(a, self.p-2, self.p)

    def from_int(self, i):
        return i % self.p

    def to_str(self, a):
        return str(a % self.p)

    def elements(self):
        if self.q > ELEMENTS_GUARD:
            raise ValueError("elements() refused at q = %d > %d" % (self.q, ELEMENTS_GUARD))
        return iter(range(self.p))

    def name(self):
        return "GF(%d)" % self.p


def _pmul(a, b, p):
    if not a or not b:
        return ()
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % p
    return _ptrim(tuple(r))


def _ptrim(a):
    i = len(a)
    while i and a[i - 1] == 0:
        i -= 1
    return a[:i]


def _pmod(a, f, p):
    a = list(a)
    df = len(f) - 1
    ilead = pow(f[df], p - 2, p)
    for i in range(len(a) - 1, df - 1, -1):
        if a[i]:
            c = (a[i] * ilead) % p
            for j in range(df + 1):
                a[i - df + j] = (a[i - df + j] - c * f[j]) % p
    return _ptrim(tuple(a))


def _pgcd(a, b, p):
    a, b = _ptrim(tuple(a)), _ptrim(tuple(b))
    while b:
        a, b = b, _pmod(a, b, p)
    return a


def _is_irreducible(f, p):
    """f monic of degree e over F_p.  Standard test: x^(p^i) - x shares no
    factor with f for i = 1..e//2, and x^(p^e) == x mod f."""
    e = len(f) - 1
    if e <= 0:
        return False
    if e == 1:
        return True
    x = (0, 1)
    h = x
    for i in range(1, e // 2 + 1):
        h = _ppowmod(h, p, f, p)
        g = _pgcd(f, _ptrim(tuple((h[j] if j < len(h) else 0) - (x[j] if j < len(x) else 0)
                                  for j in range(max(len(h), len(x))))), p)
        g = _ptrim(tuple(c % p for c in g))
        if len(g) - 1 > 0:
            return False
    h = x
    for _ in range(e):
        h = _ppowmod(h, p, f, p)
    return h == x


def _ppowmod(a, k, f, p):
    r, b = (1,), _pmod(a, f, p)
    while k:
        if k & 1:
            r = _pmod(_pmul(r, b, p), f, p)
        k >>= 1
        if k:
            b = _pmod(_pmul(b, b, p), f, p)
    return r


def smallest_irreducible(p, e):
    """Lexicographically smallest monic irreducible of degree e over F_p, in
    the fixed order 'constant coefficient varies fastest'.  Deterministic,
    exhaustive, and CONDITIONAL/excluded from every cost claim."""
    for v in range(p ** e):
        c, t = [], v
        for _ in range(e):
            c.append(t % p)
            t //= p
        f = tuple(c) + (1,)
        if _is_irreducible(f, p):
            return f
    raise NotAPrimePower(p ** e)


class ExtensionField(Field):
    """GF(p^e), El = tuple of e coefficients, low index first."""

    def __init__(self, p, e, cnt, modulus=None):
        if not is_prime(p) or e < 1:
            raise NotAPrimePower((p, e))
        self.p, self.e, self.q, self.cnt = p, e, p ** e, cnt
        if modulus is None:
            modulus = smallest_irreducible(p, e)
        if len(modulus) != e + 1 or modulus[e] % p != 1:
            raise NotAPrimePower("modulus must be monic of degree %d" % e)
        self.modulus = tuple(int(c) % p for c in modulus)
        self._zero = tuple([0] * e)
        self._one = tuple([1 % p] + [0] * (e - 1))


    def _reduce(self, a):
        a = _pmod(_ptrim(tuple(a)), self.modulus, self.p)
        return tuple(list(a) + [0] * (self.e - len(a)))

    def zero(self):
        return self._zero

    def one(self):
        return self._one

    def add(self, a, b):
        self.cnt.c_add()
        return tuple((x + y) % self.p for x, y in zip(a, b))

    def sub(self, a, b):
        self.cnt.c_add()
        return tuple((x - y) % self.p for x, y in zip(a, b))

    def neg(self, a):
        self.cnt.c_add()
        return tuple((-x) % self.p for x in a)

    def mul(self, a, b):
        self.cnt.c_mul()
        return self._reduce(_pmul(_ptrim(a), _ptrim(b), self.p))

    def inv(self, a):
        if _ptrim(a) == ():
            raise ZeroDivisionInField("inverse of 0 in %s" % self.name())
        self.cnt.c_inv()

        p = self.p
        r0, r1 = self.modulus, _ptrim(a)
        s0, s1 = (), (1,)
        while r1:
            qq, rem = _pdivmod(r0, r1, p)
            r0, r1 = r1, rem
            s0, s1 = s1, _ptrim(tuple(
                ((s0[i] if i < len(s0) else 0) - (_pmul(qq, s1, p)[i]
                 if i < len(_pmul(qq, s1, p)) else 0)) % p
                for i in range(max(len(s0), len(_pmul(qq, s1, p))))))
        c = pow(r0[0], p - 2, p)
        res = _ptrim(tuple((x * c) % p for x in s0))
        return tuple(list(res) + [0] * (self.e - len(res)))

    def from_int(self, i):
        c, t = [], int(i)
        for _ in range(self.e):
            c.append(t % self.p)
            t //= self.p
        return tuple(c)

    def to_int(self, a):
        v = 0
        for c in reversed(a):
            v = v * self.p + (c % self.p)
        return v

    def to_str(self, a):
        return "(" + ",".join(str(c % self.p) for c in a) + ")"

    def elements(self):
        if self.q > ELEMENTS_GUARD:
            raise ValueError("elements() refused at q = %d > %d" % (self.q, ELEMENTS_GUARD))
        return (self.from_int(i) for i in range(self.q))

    def name(self):
        return "GF(%d^%d), modulus %s" % (self.p, self.e, self.modulus)


def _pdivmod(a, b, p):
    a = list(a)
    db = len(b) - 1
    il = pow(b[db], p - 2, p)
    qq = [0] * max(0, len(a) - db)
    for i in range(len(a) - 1, db - 1, -1):
        if a[i]:
            c = (a[i] * il) % p
            qq[i - db] = c
            for j in range(db + 1):
                a[i - db + j] = (a[i - db + j] - c * b[j]) % p
    return _ptrim(tuple(qq)), _ptrim(tuple(a))


def field_for(q, cnt, modulus=None):
    p, e = prime_power(q)
    return PrimeField(p, cnt) if e == 1 else ExtensionField(p, e, cnt, modulus)


def multiplicative_generator(F, bound=TRIAL_DIVISION_BOUND):
    """The smallest generator of F* in the field's own element order.

    Deterministic.  For q <= ELEMENTS_GUARD the scan is over elements(); above
    it the scan is over from_int(2), from_int(3), ...  Either way the test
    needs the distinct prime factors of q-1, which is the CONDITIONAL
    preprocessing named in T05 Open item 4: raises NeedsGenerator when trial
    division below `bound` cannot supply them."""
    fs = factor_small(F.q - 1, bound)

    def candidates():
        if F.q <= ELEMENTS_GUARD:
            for x in F.elements():
                if not F.is_zero(x):
                    yield x
        else:
            i = 1
            while True:
                i += 1
                x = F.from_int(i)
                if not F.is_zero(x):
                    yield x

    for g in candidates():
        if all(not F.eq(F.pow(g, (F.q - 1) // f), F.one()) for f in fs):
            return g
    raise NeedsGenerator("no generator found in %s" % F.name())


def root_of_unity(F, m):
    """A primitive m-th root of unity in F.

    raises CharacteristicDividesOrder if F.p | m   -- (H1), REFUSE, no extension helps
    raises NoRootOfUnity(order=ord_m(q))  if m does not divide q-1
    Deterministic: smallest generator g of F*, return g**((q-1)//m)."""
    if m < 1:
        raise ValueError("m >= 1 required")
    if m % F.p == 0:
        raise CharacteristicDividesOrder(F.p, m)
    if m == 1:
        return F.one()
    if (F.q - 1) % m != 0:
        raise NoRootOfUnity(multiplicative_order(F.q % m, m), q=F.q, m=m)
    g = multiplicative_generator(F)
    return F.pow(g, (F.q - 1) // m)


def multiplicative_order(a, m):
    """ord_m(a) by integer arithmetic; None if gcd(a, m) != 1."""
    from math import gcd
    a %= m
    if gcd(a, m) != 1:
        return None
    o, t = 1, a % m
    while t != 1 % m:
        t = (t * a) % m
        o += 1
        if o > m:
            return None
    return o


def required_extension_degree(q, m, k, n):
    """d = min{ e >= 1 : (q**e - 1)/m > 2*k*n }  -- L01 (c), = ceil(log_q(2kmn+2)).

    PURE INTEGER ARITHMETIC: the comparison is (q**e - 1) > 2*k*n*m, which is
    (q**e - 1)/m > 2kn as rationals.  No floating-point log anywhere."""
    thr = 2 * k * n * m
    e, qe = 1, q
    while qe - 1 <= thr:
        e += 1
        qe *= q
    return e


def int_element(F, i):
    """The image of the integer i under Z -> K.  NOT `from_int` for extension
    fields, where `from_int` is the index decoding used by `elements()`."""
    if F.e == 1:
        return i % F.p
    return tuple([i % F.p] + [0] * (F.e - 1))


def ordered_elements(F):
    """The field's canonical element order, LAZILY and without the `elements()`
    size guard: from_int(0), from_int(1), ...  For a prime field that is
    0, 1, 2, ...; for an extension it is the base-p index decoding.  Used by the
    deterministic searches, which consume a prefix and never the whole field."""
    for i in range(F.q):
        yield F.from_int(i)
