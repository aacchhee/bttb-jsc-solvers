# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from fractions import Fraction


class ExactField:
    def __init__(self):
        self.stage = "setup"
        self.counts = {}

    def tick(self, operation):
        key = self.stage + ":" + operation
        self.counts[key] = self.counts.get(key, 0) + 1

    def zero_test(self, a):
        self.tick("zero")
        return a == self.zero


class PrimeField(ExactField):
    """p must be a supplied prime; this adapter does not certify primality."""
    def __init__(self, p):
        super().__init__()
        if p < 2:
            raise ValueError("prime must be at least two")
        self.p, self.zero, self.one = p, 0, 1
        self.name = "F" + str(p)

    def element(self, a):
        return int(a) % self.p

    def add(self, a, b):
        self.tick("add")
        return (a + b) % self.p

    def sub(self, a, b):
        self.tick("sub")
        return (a - b) % self.p

    def mul(self, a, b):
        self.tick("mul")
        return (a * b) % self.p

    def inv(self, a):
        self.tick("inv")
        if a == 0:
            raise ZeroDivisionError("inverse of zero")
        return pow(a, -1, self.p)


class RationalField(ExactField):
    def __init__(self):
        super().__init__()
        self.zero, self.one = Fraction(0), Fraction(1)
        self.name = "Q"

    def element(self, a):
        return Fraction(a)

    def add(self, a, b):
        self.tick("add")
        return a + b

    def sub(self, a, b):
        self.tick("sub")
        return a - b

    def mul(self, a, b):
        self.tick("mul")
        return a * b

    def inv(self, a):
        self.tick("inv")
        if a == 0:
            raise ZeroDivisionError("inverse of zero")
        return 1 / a


class F4(ExactField):
    """Bits encode F2[t]/(t^2+t+1), an explicit field of four elements."""
    def __init__(self):
        super().__init__()
        self.zero, self.one, self.name = 0, 1, "F4"

    def element(self, a):

        return int(a) % 2

    def add(self, a, b):
        self.tick("add")
        return a ^ b

    def sub(self, a, b):
        self.tick("sub")
        return a ^ b

    def mul(self, a, b):
        self.tick("mul")
        c = 0
        while b:
            if b & 1:
                c ^= a
            b >>= 1
            a <<= 1
            if a & 4:
                a ^= 7
        return c

    def inv(self, a):
        self.tick("inv")
        if a == 0:
            raise ZeroDivisionError("inverse of zero")
        return self.mul(a, a)
