# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.uniform_terminal import gauss_jordan, matrix_product, _inverse


def _shape(A, ncols):
    if ncols is not None and (type(ncols) is not int or ncols < 0):
        raise ValueError('ncols must be a nonnegative integer')
    m = len(A)
    n = len(A[0]) if m else (0 if ncols is None else ncols)
    if (ncols is not None and ncols != n) or any(len(row) != n for row in A):
        raise ValueError('rectangular matrix with matching ncols required')
    return m, n


def rank_profile_kernel(F, A, *, kernel=False, ncols=None, stats=None):
    """Return rank, original column profile, reduced equations and optional K.

    K has shape n by n-rank, spans the full right kernel, and has the identity
    on the increasing free-variable rows. With kernel=False its value is None.
    [] means 0-by-0 unless ncols explicitly supplies its column dimension.
    A nonempty list of empty rows means m-by-0. Input rows are not mutated.
    """
    m, n = _shape(A, ncols)
    result = gauss_jordan(F, A, stats=stats)
    r = result['rank']
    equations, profile = [], []
    if r:
        p = result['permutation']

        
        top = [list(row) for row in result['columns'][:r]]
        gathered = [A[p[i] if p is not None else i] for i in range(r)]
        equations = matrix_product(F, top, gathered, stats=stats)
        scale = _inverse(F, result['d'])
        equations = [[F.mul(scale, value) for value in row] for row in equations]
        start = 0
        for row in equations:
            pivot = next((j for j in range(start, n) if not F.is_zero(row[j])), None)
            if pivot is None:
                raise ArithmeticError('compact GJ returned a zero pivot equation')
            profile.append(pivot)
            start = pivot+1
    chosen = set(profile)  
    free = [j for j in range(n) if j not in chosen]
    K = None
    if kernel:
        K = [[F.zero() for _ in free] for _ in range(n)]
        for j, col in enumerate(free):
            K[col][j] = F.one()
            for i, pivot in enumerate(profile):
                K[pivot][j] = F.neg(equations[i][col])
    return {'rank': r, 'column_profile': profile, 'free_columns': free,
            'reduced_equations': equations, 'kernel': K, 'shape': (m, n)}


def row_rank_profile(F, A, *, ncols=None, stats=None):
    """Lexicographically earliest independent ORIGINAL rows, via transpose.

    The compact GJ row gather is not an original-row rank profile. Empty
    shapes follow rank_profile_kernel, including an explicit 0-by-n input.
    """
    m, n = _shape(A, ncols)
    transpose = [[A[i][j] for i in range(m)] for j in range(n)]
    return rank_profile_kernel(F, transpose, ncols=m, stats=stats)['column_profile']
