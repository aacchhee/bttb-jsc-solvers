# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Main HSC and HPR series/mosaic implementation.

Role: Shared. Context-local exact native polynomial, dot-product and update kernels; no whole-solver selection.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from contextlib import contextmanager
from contextvars import ContextVar

_BACKEND = ContextVar('halfstrip_core_flint_backend', default='python')
_LEDGER = ContextVar('halfstrip_core_flint_ledger',default=None)

def record(kernel, accelerated):
    ledger=_LEDGER.get()
    if ledger is not None:
        key=kernel+('_flint_calls' if accelerated else '_python_fallback_calls')
        ledger[key]=ledger.get(key,0)+1

def metadata():
    return dict(_LEDGER.get() or {'requested':selection()})


def selection():
    return _BACKEND.get()


@contextmanager
def kernel_scope(backend='python'):
    if backend not in ('python', 'flint'):
        raise ValueError('kernel backend must be python or flint')
    if backend == 'flint':
        import flint  
    token = _BACKEND.set(backend)
    ledger={'requested':backend}
    ledger_token=_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _LEDGER.reset(ledger_token)
        _BACKEND.reset(token)


def prime_modulus(F):
    """Conservative identity check, deliberately not duck-typing wrappers."""
    from .exact import PrimeField
    from .field_setup import BaseFieldView

    

    import sys
    classical = getattr(sys.modules.get('bttb_solvers._internal.hsc_solver'),'_ClassicalField',None)
    if classical is not None and type(F) is classical:
        F = getattr(F.mul,'__self__',None)
    from bttb_solvers._internal.boundary_front.fields import PrimeNative
    if type(F) is BaseFieldView:
        F = F._base
    from bttb_solvers._internal.banded_bttb.field import PrimeField as BandedPrime
    if type(F) is BandedPrime:
        if F.count_ops:
            return None
    elif type(F) not in (PrimeField, PrimeNative):
        return None
    p = F.p

    
    import struct
    return p if type(p) is int and 1 < p < 1 << (8*struct.calcsize('P')) else None


def enabled_prime(F):
    return prime_modulus(F) if selection() == 'flint' else None


def _matrix(rows,p):
    from flint import nmod_mat
    return nmod_mat(len(rows),len(rows[0]) if rows else 0,
                    [v for row in rows for v in row],p)


def _rows(matrix):
    columns=matrix.ncols()
    values=[int(v) for v in matrix.entries()]
    return [values[i*columns:(i+1)*columns] for i in range(matrix.nrows())]


def matmul(F, A, B):
    """None requests the unchanged Python fallback; lists remain the API."""
    p = enabled_prime(F)
    record('matmul',p is not None)
    if p is None:
        return None
    from flint import nmod_mat
    return _rows(_matrix(A,p)*_matrix(B,p))


def product(F, a, b):
    """Full dense product, padded to the original multiplier's output length."""
    p = enabled_prime(F)
    record('poly',p is not None)
    if p is None:
        return None
    if not a or not b:
        return []
    from flint import nmod_poly
    polynomial = nmod_poly(list(a), p) * nmod_poly(list(b), p)
    out = [int(polynomial[i]) for i in range(len(polynomial))]
    return out + [0] * (len(a)+len(b)-1-len(out))


def normalize(F, panel, columns, rank_failure):
    """The original restricted-pivot schedule; only row updates run in FLINT.

    A full RREF of [panel|I] would pivot retained columns too, changing Q.
    Each rank-one update here preserves the exact original pivot convention.
    """
    if selection() != 'flint':
        return None
    from bttb_solvers._internal.boundary_front.fields import LeanAttrField
    if type(F) is not LeanAttrField:
        record('normalize',False)
        return None
    p = prime_modulus(F._base)
    record('normalize',p is not None)
    if p is None:
        return None
    from flint import nmod_mat
    rows = len(panel)
    if not rows:
        return [], []
    width = len(panel[0])
    a = [row[:] + [int(i == j) for j in range(rows)] for i,row in enumerate(panel)]
    for j in range(columns):
        pivot = next((i for i in range(j,rows) if not F.zero_test(a[i][j])),None)
        if pivot is None:
            raise rank_failure('deficient local column '+str(j))
        a[j],a[pivot]=a[pivot],a[j]
        scale = F.inv(a[j][j])
        a[j]=[(scale*v)%p for v in a[j]]
        factors=[[a[i][j] if i != j else 0] for i in range(rows)]
        a=_rows(_matrix(a,p)-_matrix(factors,p)*_matrix([a[j]],p))
    return [row[:width] for row in a],[row[width:] for row in a]


def lu_update(F, M, c, inverse):
    """Update trailing rows of the retained LU, preserving pivot schedule."""
    p = enabled_prime(F)
    record('lu_update',p is not None)
    if p is None:
        return False
    n=len(M)
    if c+1 == n:
        return True
    from flint import nmod_mat
    factors=[]
    for r in range(c+1,n):
        f=M[r][c]*inverse % p
        M[r][c]=f
        factors.append([f])
    tail=[M[c][c+1:]]
    block=[row[c+1:] for row in M[c+1:]]
    updated=_rows(_matrix(block,p)-_matrix(factors,p)*_matrix(tail,p))
    for r,row in enumerate(updated,c+1):
        M[r][c+1:]=[int(v) for v in row]
    return True


def dot(F, a, b):
    p=enabled_prime(F)
    record('dot',p is not None)
    if p is None:
        return None
    if not a:
        return 0
    from flint import nmod_mat
    return int((_matrix([list(a)],p)*_matrix([[v] for v in b],p))[0,0])
