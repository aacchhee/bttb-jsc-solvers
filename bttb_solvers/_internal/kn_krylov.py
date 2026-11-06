# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""KN supplied-generator algebra.

Role: Trial/shared. Small-order canonical approximant leaf based on exact nilpotent/striped Krylov actions and row-rank profiles.

Entry points: canonical_leaf; row_rank_profile.

References: Khichane and Neiger (2026); see REFERENCES.md for the precise
approximant/kernel sources and preprint scope.
Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.kn_polynomial import count


def row_rank_profile(K, A):
    """Owner's original-row rank-profile extraction, without m-square basis.

    Same U P and pivot-column procedure used by uniform.order_one. The
    transpose makes earliest independent columns be earliest original rows.
    """
    F=K.F
    if not A or not A[0]:
        return []
    T=K.transpose(A)
    result=uniform.gauss_jordan(F,T,stats=K.stats)
    rank=result['rank']
    if not rank:
        return []
    permutation=result['permutation']
    U=[row[:rank] for row in result['columns'][:rank]]
    PA=[T[permutation[i] if permutation is not None else i] for i in range(rank)]
    reduced=uniform.matrix_product(F,U,PA,stats=K.stats)

    profile=[]; start=0
    for row in reduced:
        pivot=next(j for j in range(start,len(A)) if not F.is_zero(row[j]))
        profile.append(pivot); start=pivot+1
    count(K.stats,'kn_rank_profile_scans',start)
    return profile


def nilpotent_action(F, row, orders, power):
    """Row times blockdiag(J_order)^power; only coefficient moves."""
    out=[]; offset=0
    for size in orders:
        if power>=size:
            out.extend(F.zero() for _ in range(size))
        else:
            out.extend(F.zero() for _ in range(power))
            out.extend(row[offset:offset+size-power])
        offset+=size
    return out


def canonical_leaf(K, A, orders, shifts):
    F=K.F; m=len(A); sigma=sum(orders)
    if not sigma:
        return K.identity(m)
    if sigma>m:
        raise ValueError('JNSV small-order leaf requires sigma<=m')
    E=[]
    for row in A:
        coefficients=[]
        for a,order in zip(row,orders):
            coefficients.extend(a[:order])
            coefficients.extend(F.zero() for _ in range(order-min(order,len(a))))
        E.append(coefficients)
    def priority(pair):
        coordinate,power=pair
        return shifts[coordinate]+power,coordinate
    initial=sorted(range(m),key=lambda i:priority((i,0)))
    profile=row_rank_profile(K,[E[i] for i in initial])
    selected=[(initial[i],0) for i in profile]
    M=[E[i] for i,power in selected]

    power=1; bound=max(orders)
    while power<bound and selected:
        shifted=[(i,d+power) for i,d in selected]
        shifted_rows=[nilpotent_action(F,row,orders,power) for row in M]
        candidates=list(zip(selected,M))+list(zip(shifted,shifted_rows))
        candidates.sort(key=lambda item:priority(item[0]))
        profile=row_rank_profile(K,[row for _,row in candidates])
        selected=[candidates[i][0] for i in profile]
        M=[candidates[i][1] for i in profile]
        power*=2
        count(K.stats,'kn_krylov_doublings')
        count(K.stats,'kn_krylov_candidate_cells',len(candidates)*sigma)
    count(K.stats,'kn_fast_krylov_calls')
    count(K.stats,'kn_krylov_initial_cells',m*sigma)
    if not selected:
        return K.identity(m)
    rank=len(selected)
    degrees=[0]*m
    for i,power in selected:
        degrees[i]=max(degrees[i],power+1)
    columns=row_rank_profile(K,K.transpose(M))
    if len(columns)!=rank:
        raise AssertionError('Krylov row/column profile ranks differ')
    C=[[row[j] for j in columns] for row in M]
    first_dependencies=[nilpotent_action(F,E[i],orders,degrees[i]) for i in range(m)]
    D=[[row[j] for j in columns] for row in first_dependencies]
    relations=uniform.matrix_product(F,D,K.inverse_constant(C),stats=K.stats)
    B=[[[] for _ in range(m)] for _ in range(m)]
    for i in range(m):
        B[i][i]=[F.zero() for _ in range(degrees[i])]+[F.one()]
        for coefficient,(j,power) in zip(relations[i],selected):
            if not F.is_zero(coefficient):
                a=B[i][j]
                if len(a)<=power:
                    a.extend(F.zero() for _ in range(power+1-len(a)))
                a[power]=F.sub(a[power],coefficient)
        B[i]=[K.trim(a) for a in B[i]]
    count(K.stats,'kn_krylov_output_cells',m*m+m*sum(degrees))
    return B
