# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.kn_services import Services
from bttb_solvers._internal.kn_krylov import row_rank_profile
from bttb_solvers._internal.kn_polynomial import count


def coordinate_completion(K,N,dimension):
    """Return identity-column indices complementing the primitive N(0)."""
    if len(N)!=dimension:
        raise ValueError('kernel specialization dimension mismatch')
    width=len(N[0]) if N else 0
    if any(len(row)!=width for row in N):
        raise ValueError('ragged kernel basis')
    constant=[[a[0] if a else K.F.zero() for a in row] for row in N]
    augmented=[list(row)+[K.F.one() if i==j else K.F.zero()
                         for j in range(dimension)] for i,row in enumerate(constant)]
    profile=row_rank_profile(K,K.transpose(augmented))
    if len(profile)!=dimension or profile[:width]!=list(range(width)):
        raise ArithmeticError('full generating kernel specialization lost primitivity')
    selected=[j-width for j in profile[width:]]
    if len(selected)!=dimension-width:
        raise ArithmeticError('coordinate completion has incorrect size')
    count(K.stats,'banded_completion_copy_cells',dimension*(dimension+width))
    return selected


def certified_maximum_minor(F,A,*,degree_bound=None,stats=None):
    """Square polynomial input -> normal rank, coordinate minor, witnesses.

    Nonsingularity of the returned rho-square minor is certified by the
    algorithm two-kernel construction, BEFORE a determinant routine is used.
    """
    K=Services(F,stats)
    n,columns=K.shape(A)
    if not n or columns!=n:
        raise ValueError('positive square polynomial matrix required')
    input_degree=max((len(a)-1 for row in A for a in row),default=0)
    if degree_bound is None:
        degree_bound=max(0,input_degree)
    if type(degree_bound) is not int or degree_bound<max(0,input_degree):
        raise ValueError('actual nonnegative integer degree bound required')
    N,_=K.minimal_nullspace_basis(A,[degree_bound]*n)
    nullity=len(N[0]) if N else 0
    rho=n-nullity
    J=coordinate_completion(K,N,n)
    if rho==0:
        return {'normal_rank':0,'rows':[],'columns':[],'minor':[],
                'right_kernel':N,'second_kernel':None,
                'empty_minor_determinant':[F.one()],
                'minor_certified_nonsingular':True}
    if rho==n:
        I=list(range(n)); second=None
    else:
        selected=[[row[j] for j in J] for row in A]

        

        second_input=K.transpose(selected)
        second_input += [[[] for _ in range(n)] for _ in range(n-rho)]
        count(stats,'banded_second_kernel_padding_cells',(n-rho)*n)
        second,_=K.minimal_nullspace_basis(second_input,[degree_bound]*n)
        I=coordinate_completion(K,second,n)
        if len(I)!=rho:
            raise ArithmeticError('second full kernel has unexpected nullity')
    minor=[[list(A[i][j]) for j in J] for i in I]
    count(stats,'banded_minor_copy_cells',sum(len(a) for row in minor for a in row))
    return {'normal_rank':rho,'rows':I,'columns':J,'minor':minor,
            'right_kernel':N,'second_kernel':second,
            'minor_certified_nonsingular':True}


def narrow_raw_minor(F,n,k,raw,*,stats=None):
    """Acquire F=z^k a(z) directly from raw square narrow Toeplitz lags."""
    if type(n) is not int or type(k) is not int or not 1<=k<n or 2*k>=n:
        raise ValueError('narrow raw branch requires integers 1<=k<n and 2k<n')
    for outer in range(-k,k+1):
        for inner in range(1-n,n):
            if (outer,inner) not in raw:
                raise ValueError('missing raw lag coefficient')
    A=[[[raw[outer,i-j] for outer in range(-k,k+1)] for j in range(n)] for i in range(n)]
    count(stats,'banded_raw_symbol_copy_cells',(2*k+1)*n*n)
    result=certified_maximum_minor(F,A,degree_bound=2*k,stats=stats)
    result.update({'symbol':A,'n':n,'k':k,'entry_degree_bound':2*k})
    return result
