# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Matrix convention and independent exact residual checks."""
from .api import PRIME, _validate


def banded_to_halfstrip(lags,n,k):
    """Pad outer lags k+1,...,n-1 by zero without changing the matrix."""
    rows=_validate(lags,n,k)
    return rows+[[0]*(2*n-1) for _ in range(n-k-1)]


def dense_matrix(lags,n,width,*,kind='banded',p=PRIME):
    """Build a python-flint matrix for testing, paying O(n**4) storage."""
    from flint import nmod_mat
    rows=_validate(lags,n,width,banded=kind=='banded')
    A=nmod_mat(n*n,n*n,p)
    for t in range(n):
        for u in range(max(0,t+width-len(rows)+1),min(n,t+width+1)):
            lag=rows[t-u+width]
            for a in range(n):
                for b in range(n):A[t*n+a,u*n+b]=int(lag[n-1+a-b])%p
    return A


def action(lags,n,width,vector,*,kind='banded',p=PRIME,transpose=False):
    """Apply the raw lag matrix with exact polynomial products, independently.

    Uses O(n**2) vector space rather than a physical dense matrix. Valid for
    both matrix classes and arbitrary literal transposes.
    """
    from flint import nmod_poly
    rows=_validate(lags,n,width,banded=kind=='banded')
    if len(vector)!=n*n:raise ValueError('vector must have n*n entries')
    if transpose:
        rows=[row[::-1] for row in rows[::-1]]
        width=len(rows)-1-width
    kernels=[nmod_poly([int(v)%p for v in row],p) for row in rows]
    blocks=[nmod_poly([int(v)%p for v in vector[u*n:(u+1)*n]],p) for u in range(n)]
    out=[]
    for t in range(n):
        acc=nmod_poly([],p)
        for d,kernel in enumerate(kernels):
            u=t-d+width
            if 0<=u<n:acc+=kernel*blocks[u]
        coeffs=acc.coeffs()
        out.extend(int(coeffs[n-1+a]) if n-1+a<len(coeffs) else 0 for a in range(n))
    return out
