# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Main HSC and HPR series/mosaic implementation.

Role: Shared. Fast local linear-algebra helpers used by carrier/defect calculations.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from .uniform_terminal import (
    gauss_jordan, constant_determinant as determinant, _inverse, _power,
)


def _square_size(A):
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError('square matrix required')
    return n


def rank(F, A, *, stats=None):
    """Rank of a rectangular matrix; [] and any m-by-0 matrix have rank zero."""
    return gauss_jordan(F, A, stats=stats)['rank']


def _inverse_from_result(F, result, n):
    if result['rank'] != n:
        raise ZeroDivisionError('singular matrix')

    

    scale = _inverse(F, result['d'])
    p = result['permutation']
    out = [[F.zero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][p[j] if p is not None else j] = F.mul(
                scale, result['columns'][i][j])
    return out


def inverse(F, A, *, stats=None):
    """Inverse of a square matrix, raising ZeroDivisionError if singular."""
    n = _square_size(A)
    if n == 0:
        return []
    return _inverse_from_result(F, gauss_jordan(F, A, stats=stats), n)


def inverse_and_determinant(F, A, *, stats=None):
    """Return (inverse, determinant) using one compact terminal decomposition."""
    n = _square_size(A)
    if n == 0:
        return [], F.one()
    result = gauss_jordan(F, A, stats=stats)
    out = _inverse_from_result(F, result, n)

    denominator = F.mul(result['eta'], result['epsilon'])
    value = F.mul(_power(F, result['d'], n), _inverse(F, denominator))
    return out, value
