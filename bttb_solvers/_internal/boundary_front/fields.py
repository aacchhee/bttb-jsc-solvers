# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""BFM local-front implementation.

Role: Main/shared. Native eight-operation fields and adapters, including LeanAttrField for the GBE comparator.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
class PrimeNative:
    def __init__(self, p):
        if type(p) is not int or p < 2:
            raise ValueError("prime modulus required (primality is not certified)")
        self.p = p
        self.name = "F" + str(p)

    def zero(self): return 0
    def one(self): return 1
    def add(self, a, b): return (a + b) % self.p
    def sub(self, a, b): return (a - b) % self.p
    def neg(self, a): return (-a) % self.p
    def mul(self, a, b): return (a * b) % self.p

    def inv(self, a):
        if a % self.p == 0:
            raise ZeroDivisionError("inverse of zero")
        return pow(a, -1, self.p)

    def is_zero(self, a): return a == 0
    def element(self, v): return int(v) % self.p
    def random(self, rng): return rng.randrange(self.p)


class ExtNative:
    """GF(p^e) = F_p[x]/(modulus), modulus monic low-to-high of length e+1.

    Elements are tuples of e residues (low degree first).  Schoolbook
    arithmetic; inverse by Fermat power.  Small fields only.
    """
    def __init__(self, p, modulus):
        self.p, self.modulus = p, tuple(modulus)
        self.e = len(self.modulus) - 1
        if self.e < 2 or self.modulus[-1] != 1:
            raise ValueError("monic modulus of degree >= 2 required")
        self.name = "F%d^%d" % (p, self.e)
        self._zero = (0,) * self.e
        self._one = (1,) + (0,) * (self.e - 1)

    def zero(self): return self._zero
    def one(self): return self._one
    def add(self, a, b): return tuple((x + y) % self.p for x, y in zip(a, b))
    def sub(self, a, b): return tuple((x - y) % self.p for x, y in zip(a, b))
    def neg(self, a): return tuple((-x) % self.p for x in a)

    def mul(self, a, b):
        p, e = self.p, self.e
        out = [0] * (2 * e - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    out[i + j] = (out[i + j] + x * y) % p
        for i in range(2 * e - 2, e - 1, -1):
            c = out[i]
            if c:
                for j in range(e):
                    out[i - e + j] = (out[i - e + j] - c * self.modulus[j]) % p
        return tuple(out[:e])

    def is_zero(self, a): return not any(a)

    def inv(self, a):
        if not any(a):
            raise ZeroDivisionError("inverse of zero")
        out, base, exponent = self._one, a, self.p ** self.e - 2
        while exponent:
            if exponent & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            exponent >>= 1
        return out

    def element(self, v):
        if isinstance(v, tuple):
            return tuple(x % self.p for x in v)
        return (int(v) % self.p,) + (0,) * (self.e - 1)

    def random(self, rng): return tuple(rng.randrange(self.p) for _ in range(self.e))


def make_field(spec):
    """spec: 'F2', 'F998244353', 'F4', 'F8', 'F9', or a dict
    {'kind':'prime','p':p} / {'kind':'extension','p':p,'modulus':[...]}."""
    if isinstance(spec, dict):
        if spec["kind"] == "prime":
            return PrimeNative(spec["p"])
        if spec["kind"] == "extension":
            return ExtNative(spec["p"], spec["modulus"])
        raise ValueError("field kind")
    named = {"F4": (2, (1, 1, 1)), "F8": (2, (1, 1, 0, 1)), "F9": (3, (1, 0, 1)),
             "F16": (2, (1, 1, 0, 0, 1))}
    if spec in named:
        f = ExtNative(*named[spec])
        f.name = spec
        return f
    if isinstance(spec, str) and spec.startswith("F") and spec[1:].isdigit():
        return PrimeNative(int(spec[1:]))
    raise ValueError("unknown field " + repr(spec))


class Counted:
    """Wrap a native field; count calls per (stage, operation).  Counting wrapper
    for the package's own phase accounting (not used when the caller already
    supplies a counting base)."""
    def __init__(self, base):
        self._base = base
        self.stage = "setup"
        self.counts = {}
        for name in ("zero", "one", "add", "sub", "neg", "mul", "inv", "is_zero"):
            setattr(self, name, self._wrap(name, getattr(base, name)))

    def _wrap(self, name, method):
        counts = self.counts

        def call(*args):
            key = self.stage + ":" + name
            counts[key] = counts.get(key, 0) + 1
            return method(*args)
        return call


class AttrField:
    """Attribute-style adapter for the unchanged algorithm sources (cf. the
    benchmark's classical_banded.NativeField)."""
    def __init__(self, base):
        self._base = base
        self.stage, self.counts = "native_ingress", {}
        self.zero, self.one = base.zero(), base.one()
        self.name = getattr(base, "name", "native")

    def _tick(self, operation):
        key = self.stage + ":" + operation
        self.counts[key] = self.counts.get(key, 0) + 1

    def zero_test(self, a):
        self._tick("zero"); return self._base.is_zero(a)

    def add(self, a, b):
        self._tick("add"); return self._base.add(a, b)

    def sub(self, a, b):
        self._tick("sub"); return self._base.sub(a, b)

    def neg(self, a):
        self._tick("neg"); return self._base.neg(a)

    def mul(self, a, b):
        self._tick("mul"); return self._base.mul(a, b)

    def inv(self, a):
        self._tick("inv"); return self._base.inv(a)


class LeanAttrField:
    """Attribute-style adapter with NO per-operation bookkeeping: the native bound
    methods are exposed directly (zero/one as constants, is_zero as `zero_test`),
    as the algorithm sources expect.  Timed rootfree runs use this adapter, so that
    both arms are timed on the plain field; `stage` and `counts` exist only to
    satisfy the unchanged sources and are never written to a counter."""
    def __init__(self, base):
        self._base = base
        self.stage, self.counts = "native_ingress", {}
        self.zero, self.one = base.zero(), base.one()
        self.name = getattr(base, "name", "native")
        self.zero_test = base.is_zero
        self.add, self.sub, self.neg = base.add, base.sub, base.neg
        self.mul, self.inv = base.mul, base.inv
