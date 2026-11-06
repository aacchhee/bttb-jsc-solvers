# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial. Dispatch the dedicated raw-lag solve to narrow or wide physical acquisition, paying front end, generator acquisition and both queries.

Entry points: solve_banded.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.narrow_physical_solver import solve_narrow
from bttb_solvers._internal.wide_physical_solver import solve_wide, SingularPhysicalMatrix


def solve_banded(base, n, k, raw, forward_rhs, transpose_rhs, *, compress_generators=True):
    """Original native lags and two independent physical RHS -> native answers.

    No half-strip core, determinant, marks or reusable prepared solver API.
    Physical singularity raises SingularPhysicalMatrix before either query.
    """
    if type(compress_generators) is not bool:
        raise ValueError('compress_generators must be a Boolean route choice')
    if type(n) is not int or type(k) is not int or n < 2 or not 1 <= k < n:
        raise ValueError('integer n>=2 and 1<=k<n required')
    keys = tuple((a,b) for a in range(-k,k+1) for b in range(1-n,n))
    if any(key not in raw for key in keys):
        raise ValueError('complete original raw lag mapping required')
    cached = {key: raw[key] for key in keys}
    forward, transposed = tuple(forward_rhs), tuple(transpose_rhs)
    if len(forward) != n*n or len(transposed) != n*n:
        raise ValueError('both physical RHS must have n*n native entries')
    if 2*k < n:
        return solve_narrow(base,n,k,cached,forward,transposed,
                            compress_generators=compress_generators)
    result = solve_wide(base,n,k,cached,forward,transposed,
                        compress_generators=compress_generators)
    result['metadata'].update(dimension_branch='2k>=n', attempts=1, restarts=[])
    return result
