# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Field setup and common representation tools.

Role: Shared. Own scalar embeddings, guarded operations, algebra restrictions and descent for label-based source constructions.

Entry points: OwnedLabelAlgebra.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.exact import EtaleField, Restart, pm, pgcd, pxgcd


class OwnedLabelAlgebra(EtaleField):
    """Same finite-etale arithmetic/Restart contract, no cacheability probes.

    Inherited empty cache dictionaries remain unused for compatibility. The
    inverse-guard transcript is retained; native-base-owned caches are neither
    inspected nor cleared. This is not a complete peak-memory guarantee.
    """
    def mul(self, a, b):
        return self.reduce(pm(self.base, a, b))

    def is_zero(self, a):
        self.guards += 1
        value = self.reduce(a)
        if all(self.base.is_zero(coefficient) for coefficient in value):
            return True
        gcd = pgcd(self.base, value, self.modulus)
        if len(gcd) > 1:
            raise Restart(value, gcd, self.guards)
        return False

    def inv(self, a):
        if self.is_zero(a):
            raise ZeroDivisionError
        self.inverse_guards.append({'denominator': a, 'unit': True,
                                    'gcd': [self.base.one()]})
        _gcd, inverse = pxgcd(self.base, a, self.modulus)
        return self.reduce(inverse)
