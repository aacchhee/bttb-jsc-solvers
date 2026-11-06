# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Deterministic problem inputs used for the paper's measurements.

The fixed byte domain below is part of the input seed. Do not change it:
its only role is preserving the exact coefficient/RHS streams in the paper.
"""
import hashlib
from bttb_solvers import PRIME

_INPUT_DOMAIN=bytes((100,117,97,108,45,115,121,118,101,114,116,45,118,50,47))


def fixture(case):
    n,w,R,seed=case['n'],case['width'],case['pairs'],case['seed']
    if case['family'].startswith('halfstrip-'):
        from bttb_solvers._internal.halfstrip_core.reuse import random_failed_h0_lags,_rhs_vectors,_field_spec
        name,base=_field_spec('F'+str(PRIME))
        deficient=case['family']=='halfstrip-rank-deficient'
        rows=random_failed_h0_lags(n,w,'rank-nm1' if deficient else 'generic',seed)
        if not deficient:rows[0]=[int(i==n-1) for i in range(2*n-1)]
        rhs=_rhs_vectors(name,base,n,2*R,seed)
        return rows,rhs[:R],rhs[R:],'halfstrip'
    prefix=_INPUT_DOMAIN+f'{n}/{w}/banded/common_h0_triangular/{seed}'.encode()
    def draw(counter):
        return int.from_bytes(hashlib.sha256(prefix+counter.to_bytes(8,'big')).digest()[:8],'big')%PRIME
    rows=[[draw(d*(2*n-1)+i) for i in range(2*n-1)] for d in range(2*w+1)]
    rows[0][:n-1]=[0]*(n-1);rows[0][n-1]=1;rows[0][n]=1
    def rhs(direction):
        return [[int.from_bytes(hashlib.sha256(prefix+b'/rhs/'+direction.encode()+pair.to_bytes(4,'big')+
                    i.to_bytes(8,'big')).digest()[:8],'big')%PRIME for i in range(n*n)] for pair in range(R)]
    return rows,rhs('forward'),rhs('transpose'),'banded'
