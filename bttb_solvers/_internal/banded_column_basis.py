# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial/shared. Polynomial column/image basis construction for the dedicated front end, with degree and reconstruction bookkeeping.

Entry points: ColumnBasis.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.banded_general_approximant import GeneralApproximant
from bttb_solvers._internal.kn_polynomial import count


class ColumnBasis(GeneralApproximant):
    def _ambient_product(self,A,B,bounds,ambient):
        """Literal zero rows keep packed-product dimensions within paid scope."""
        rows,inner=self.shape(A)
        binner,columns=self.shape(B)
        if inner!=binner or rows>ambient or inner>ambient or columns>ambient:
            raise ValueError('ambient product shape exceeds its dimension budget')
        padded=[list(row) for row in A]+[[[] for _ in range(inner)] for _ in range(ambient-rows)]
        count(self.stats,'column_basis_ambient_padding_cells',(ambient-rows)*inner)
        return self.degree_budget_product(padded,B,bounds)[:rows]

    def reversed_kernel(self,M,shift,budget):
        """2013 Algorithm1: full (M,-shift)-minimal generating right kernel.

        Source precondition: shift>=0, sum shift<=budget, sum shifted row
        degrees<=budget, and the COMPLETE minimal kernel has (-shift)
        degrees<=0. This last property is proved for the two ColumnBasis
        callers; this is not a general negative-shift kernel oracle.
        """
        shift=list(shift); ambient=len(shift)
        if not ambient or type(budget) is not int or budget<0 or any(type(s) is not int or s<0 for s in shift):
            raise ValueError('positive ambient dimension, nonnegative integer shift/budget required')
        if sum(shift)>budget or any(len(row)!=ambient for row in M):
            raise ValueError('reverse-kernel shift budget or shape invalid')
        M=[[self.trim(a) for a in row] for row in M]
        k=len(M)
        working_dimension=max(ambient,k)
        if not k:
            return self.identity(ambient),[-s for s in shift]
        row_bounds=[max(0,d) for d in self.shifted_degrees(M,shift)]
        if sum(row_bounds)>budget:
            raise ValueError('reverse-kernel shifted row-degree budget exceeded')
        if k==1:
            groups=[(row_bounds[0]+1,[0])]
        else:
            levels=(k-1).bit_length()
            buckets=[[] for _ in range(levels)]
            for row,bound in enumerate(row_bounds):
                index=levels-1
                for i in range(1,levels):
                    if bound*(1<<i)>budget:
                        index=i-1;break
                buckets[index].append(row)
            groups=[((budget+(1<<i)-1)//(1<<i)+1,rows)
                    for i,rows in enumerate(buckets) if rows]
        accumulated=self.identity(ambient)
        current_shift=list(shift)
        for order,indices in groups:

            
            uniform_rows=[[self.shift(a,order-row_bounds[i]-1) for a in M[i]] for i in indices]
            transpose_product=self._ambient_product(self.transpose(accumulated),
                    self.transpose(uniform_rows),shift,working_dimension)
            residual=self.transpose(transpose_product)
            row_basis=self.canonical_general(self.transpose(residual),[order]*len(indices),
                                              [-s for s in current_shift])
            basis=self.transpose(row_basis)
            profile=self.column_degrees(basis,[-s for s in current_shift])
            keep=[j for j,d in enumerate(profile) if d<=0]
            if not keep:
                return [[] for _ in range(ambient)],[]
            local=[[row[j] for j in keep] for row in basis]

            

            composed=self._ambient_product(self.transpose(local),self.transpose(accumulated),
                                             current_shift,working_dimension)
            accumulated=self.transpose(composed)
            actual_profile=self.column_degrees(accumulated,[-s for s in shift])
            current_shift=[-d for d in actual_profile]
            if any(s<0 for s in current_shift) or sum(current_shift)>budget:
                raise ArithmeticError('reversed-kernel accumulated degree invariant failed')
            count(self.stats,'column_basis_reversed_groups')
            count(self.stats,'column_basis_reversed_total_order',len(indices)*order)
        return accumulated,[-s for s in current_shift]

    def column_basis_full_row_rank(self,A):
        """Actual image-module basis B, full right kernel N, right factor G.

        Input m-by-n with 1<=m<=n, promised polynomial full row rank. A
        deficient input is rejected from its actual full-kernel dimension.
        The ordinary initial kernel is square zero-row padded before ZLS.
        """
        m,n=self.shape(A)
        if not 1<=m<=n:
            raise ValueError('column basis requires 1<=rows<=columns')
        A=[[self.trim(a) for a in row] for row in A]
        shift=[max(0,d) for d in self.column_degrees(A)]
        budget=sum(shift)
        padded=[list(row) for row in A]+[[[] for _ in range(n)] for _ in range(n-m)]
        N,_=self.minimal_nullspace_basis(padded,shift)
        nullity=len(N[0])
        if nullity!=n-m:
            raise ValueError('polynomial matrix is not full row rank')
        if nullity:
            left_kernel,_=self.reversed_kernel(self.transpose(N),shift,budget)
        else:
            left_kernel=self.identity(n)
        G=self.transpose(left_kernel)
        if len(G)!=m:
            raise ArithmeticError('reverse-kernel right factor has incorrect rank')
        column_bounds=[-d for d in self.shifted_degrees(G,[-s for s in shift])]
        if any(t<0 for t in column_bounds) or sum(column_bounds)>budget:
            raise ArithmeticError('column-basis recovery degree invariant failed')
        signed=[[list(a) for a in left]+[self.scale(a,self.F.neg(self.F.one())) for a in right]
                for left,right in zip(self.transpose(A),self.transpose(G))]
        star_shift=[0]*m+column_bounds
        recovery,_=self.reversed_kernel(signed,star_shift,budget)
        if len(recovery)!=2*m or any(len(row)!=m for row in recovery):
            raise ArithmeticError('column-basis recovery kernel dimension mismatch')
        top=recovery[:m]; bottom=recovery[m:]
        if any(len(self.trim(a))>1 for row in top for a in row):
            raise ArithmeticError('column-basis recovery top block is not constant')
        V=[[a[0] if a else self.F.zero() for a in row] for row in top]
        inverse=self.inverse_constant(V)
        B=self.degree_budget_product([[[a] for a in row] for row in self.transpose(inverse)],
                                     self.transpose(bottom),[0]*m)
        if any(max(0,d)>t for d,t in zip(self.column_degrees(B),column_bounds)):
            raise ArithmeticError('column-basis output column bound exceeded')
        count(self.stats,'column_basis_calls')
        return {'basis':B,'kernel':N,'right_factor':G,
                'recovery_kernel':recovery,'recovery_constant':V,
                'column_bounds':column_bounds,'input_degree_budget':budget,
                'source':'Zhou-Labahn2013 Algorithms1-2 / JNV2020 Algorithms4-6',
                'scope':'candidate full-row-rank image-module basis pending checked'}
