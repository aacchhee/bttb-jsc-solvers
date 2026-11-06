# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core import poly as sparse_poly
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.kn_polynomial import Polynomial, count


class Kernel(Polynomial):
    def column_degrees(self, A, shift=None):
        rows,cols=self.shape(A)
        shift=[0]*rows if shift is None else list(shift)
        if len(shift)!=rows:
            raise ValueError('column-degree shift mismatch')
        return [max((len(A[i][j])-1+shift[i] for i in range(rows) if A[i][j]),default=-1)
                for j in range(cols)]

    def degree_budget_product(self, A, B, column_bounds=None):
        """Coefficient-splitting variant of the source unbalanced-product step.

        Split A columns at block degree b; shift paired B rows, then split
        their columns at the same b. Only bounded-degree matrices reach the
        uniform product. No maximum-degree padded original matrix is built.
        """
        rows,inner=self.shape(A)
        binner,cols=self.shape(B)
        if inner!=binner:
            raise ValueError('degree-budget product dimensions mismatch')
        if not inner or not cols:
            return [[[] for _ in range(cols)] for _ in range(rows)]
        actual=self.column_degrees(A)
        bounds=[max(0,d) for d in actual] if column_bounds is None else list(column_bounds)
        if len(bounds)!=inner or any(type(s) is not int or s<max(0,d) for s,d in zip(bounds,actual)):
            raise ValueError('invalid product column degree bounds')
        output=self.column_degrees(B,bounds)
        budget=max(1,sum(bounds),sum(max(0,d) for d in output))
        block=max(1,(budget+max(1,rows)-1)//max(1,rows))
        achunks=[]
        for i,s in enumerate(bounds):
            for k in range(s//block+1):
                achunks.append((i,k*block))
        bchunks=[]
        for j,d in enumerate(output):
            if d>=0:
                for k in range(d//block+1):
                    bchunks.append((j,k*block))
        if not bchunks:
            return [[[] for _ in range(cols)] for _ in range(rows)]
        Ap=[[self.trim(A[r][i][power:power+block]) for i,power in achunks]
            for r in range(rows)]

        Bp=[]
        for i,power in achunks:
            row=[]
            for j,start in bchunks:
                row.append(self.trim([B[i][j][q-power]
                            if 0<=q-power<len(B[i][j]) else self.F.zero()
                            for q in range(start,start+block)]))
            Bp.append(row)
        count(self.stats,'kn_packed_product_calls')
        count(self.stats,'kn_packed_copy_cells',block*(rows*len(achunks)+len(achunks)*len(bchunks)))
        product=self.pm_mul(Ap,Bp)
        out=[[[self.F.zero() for _ in range(max(0,d+2*block))]
              for d in output] for _ in range(rows)]
        for r in range(rows):
            for k,(j,power) in enumerate(bchunks):
                for q,value in enumerate(product[r][k]):
                    out[r][j][power+q]=self.F.add(out[r][j][power+q],value)
        count(self.stats,'kn_packed_copy_cells',rows*sum(max(0,d+2*block) for d in output))
        return [[self.trim(a) for a in row] for row in out]

    def minimal_nullspace_basis(self, A, shift=None):
        m,n=self.shape(A)
        if m>n:
            raise ValueError('public ZLS contract requires rows<=columns')
        return self._minimal_nullspace_basis(A,shift)

    def _minimal_nullspace_basis(self, A, shift=None):
        """Full generating right kernel, never merely rationally independent.

        Return n by nullity polynomial matrix and actual input-shift degrees.
        All sorting depends on integer degree profiles, never field coercion.
        Zero-dimensional kernels retain n empty rows.
        """
        m,n=self.shape(A)
        if not m:
            raise ValueError('positive row count required')
        degrees=self.column_degrees(A)
        shift=[max(0,x) for x in degrees] if shift is None else list(shift)
        if len(shift)!=n or any(type(s) is not int or s<max(0,d) for s,d in zip(shift,degrees)):
            raise ValueError('nonnegative shift must bound column degrees')
        count(self.stats,'kn_zls_nodes')
        if not n:
            return [],[]
        permutation=sorted(range(n),key=lambda j:(shift[j],j))
        sorted_A=[[row[j] for j in permutation] for row in A]
        sorted_shift=[shift[j] for j in permutation]
        N=self._kernel(sorted_A,sorted_shift)
        out=[[[] for _ in range(len(N[0]))] for _ in range(n)]
        for i,j in enumerate(permutation):
            out[j]=N[i]
        return out,self.column_degrees(out,shift)

    def _kernel(self, A, shift):
        m,n=self.shape(A)
        if not n:
            return []
        if m>n:

            
            cut=m//2
            N1,u=self._minimal_nullspace_basis(A[:cut],shift)
            if not N1 or not N1[0]:
                return [[] for _ in range(n)]
            lower=self.degree_budget_product(A[cut:],N1,shift)
            N2,_=self._minimal_nullspace_basis(lower,u)
            if not N2 or not N2[0]:
                return [[] for _ in range(n)]
            return self.degree_budget_product(N1,N2,u)

        rho=sum(sorted(shift)[-m:])
        order=max(1,3*((rho+m-1)//m))
        count(self.stats,'kn_zls_uniform_order_sum',order)
        row_basis=self.pm_basis(self.transpose(A),order,shift)
        P=self.transpose(row_basis)
        profile=self.column_degrees(P,shift)
        indices=sorted(range(n),key=lambda j:(profile[j],j))
        P=[[row[j] for j in indices] for row in P]
        profile=[profile[j] for j in indices]
        product=self.degree_budget_product(A,P,shift)
        zero=[]; nonzero=[]
        for j in range(n):

            
            if all(not self.trim(product[i][j]) for i in range(m)):
                zero.append(j)
            else:
                nonzero.append(j)
        P1=[[row[j] for j in zero] for row in P]
        if m==1 or not nonzero:
            return P1
        P2=[[row[j] for j in nonzero] for row in P]
        residual=[]
        for row in product:
            out=[]
            for j in nonzero:
                if any(not self.F.is_zero(x) for x in row[j][:order]):
                    raise AssertionError('order basis residual has low coefficients')
                out.append(self.trim(row[j][order:]))
            residual.append(out)
        residual_shift=[profile[j]-order for j in nonzero]
        if any(s<0 for s in residual_shift):
            raise AssertionError('nonkernel column has negative residual shift')
        cut=m//2
        N1,u=self._minimal_nullspace_basis(residual[:cut],residual_shift)
        if not N1 or not N1[0]:
            return P1
        lower=self.degree_budget_product(residual[cut:],N1,residual_shift)
        N2,_=self._minimal_nullspace_basis(lower,u)
        if not N2 or not N2[0]:
            return P1
        Q=self.degree_budget_product(N1,N2,u)
        P2Q=self.degree_budget_product(P2,Q,[profile[j] for j in nonzero])
        return [left+right for left,right in zip(P1,P2Q)]

    def rational_vector_xd(self, P, vector, order, column_bounds=None):
        """KN Lemma24 kernel route for M=x^order, exact functional service.

        This replaces beta² causal recursion algorithmically, while retaining
        the uniform-order ZLS cost limitation stated in this module.
        """
        beta,width=self.shape(P)
        if beta!=width or beta<1 or len(vector)!=beta or type(order) is not int or order<beta:
            raise ValueError('square P and order>=beta required')
        degrees=self.column_degrees(P)
        bounds=[max(0,x) for x in degrees] if column_bounds is None else list(column_bounds)
        if len(bounds)!=beta or any(type(x) is not int or x<max(0,d) for x,d in zip(bounds,degrees)):
            raise ValueError('column degree bounds invalid')
        if sum(bounds)>order or any(len(a)>order for a in vector):
            raise ValueError('KN linear degree budget or RHS order exceeded')

        self.inverse_constant([[a[0] if a else self.F.zero() for a in row] for row in P])
        augmented=[list(row)+[self.scale(v,self.F.neg(self.F.one()))]
                   for row,v in zip(P,vector)]
        vdegree=max((len(a)-1 for a in vector),default=0)
        N,_=self.minimal_nullspace_basis(augmented,bounds+[max(0,vdegree)])
        if not N or any(len(row)!=1 for row in N):
            raise AssertionError('nonsingular P must give a rank-one full kernel')
        numerator=[row[0] for row in N[:-1]]
        denominator=self.trim(N[-1][0])
        if not denominator:
            raise AssertionError('kernel denominator must be nonzero')

        if self.F.is_zero(denominator[0]):
            raise AssertionError('primitive kernel denominator is not an x-unit')
        inverse=sparse_poly.sinv(uniform._FieldOperations(self.F),
                                 {i:x for i,x in enumerate(denominator)},order)
        dense=[inverse.get(i,self.F.zero()) for i in range(order)]
        count(self.stats,'kn_phase2_kernel_vector_calls')
        return [self.trim(self.mul(a,dense)[:order]) for a in numerator]
