# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""BFM local-front implementation.

Role: Main/shared. Independent physical residual actions selected by the prepared BFM interface.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from .kernel import matmul


def residual_classical(F, blocks, m, k, s, x, rhs, transposed=False):
    ok = True
    for i in range(m):
        for a in range(s):
            value = F.zero()
            for j in range(max(0, i - k), min(m, i + k + 1)):
                lag = i - j
                for b in range(s):
                    coeff = blocks[-lag][b][a] if transposed else blocks[lag][a][b]
                    value = F.add(value, F.mul(coeff, x[j * s + b]))
            if not F.is_zero(F.sub(value, rhs[i * s + a])):
                ok = False
    return ok


def residual_fast(F, blocks, m, k, s, x, rhs):
    """Check T x = rhs for the forward block matrix: X has the blocks of x as columns."""
    X = [[x[j * s + b] for j in range(m)] for b in range(s)]
    acc = [[F.zero() for _ in range(s)] for _ in range(m)]       
    for lag in range(-k, k + 1):
        Y = matmul(F, blocks[lag], X)                              
        for i in range(m):
            j = i - lag
            if 0 <= j < m:
                row = acc[i]
                for a in range(s):
                    row[a] = F.add(row[a], Y[a][j])
    return all(F.is_zero(F.sub(acc[i][a], rhs[i * s + a]))
               for i in range(m) for a in range(s))
