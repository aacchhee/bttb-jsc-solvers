# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.banded_column_basis import ColumnBasis
from bttb_solvers._internal.banded_frontend_rank import coordinate_completion
from bttb_solvers._internal.kn_polynomial import count


class NonsingularDeterminant(ColumnBasis):
    def determinant_nonsingular(self,A):
        """Return det(A), under a previously established nonsingularity proof.

        Empty input has determinant1. A scalar returns its actual native
        polynomial. Recursive nonsingularity follows from LNZ Lemma3.1.
        """
        n,columns=self.shape(A)
        if n!=columns:
            raise ValueError('certified-nonsingular determinant requires square input')
        volume=sum(len(a) for row in A for a in row)
        A=[[self.trim(a) for a in row] for row in A]
        count(self.stats,'lnz_nodes')
        count(self.stats,'lnz_input_coefficient_slots',volume)
        count(self.stats,'lnz_node_copy_slots',volume+n*n)
        if not n:
            count(self.stats,'lnz_empty_nodes')
            return [self.F.one()]
        if n==1:
            count(self.stats,'lnz_scalar_nodes')
            return A[0][0]
        m=(n+1)//2
        upper=A[:m];lower=A[m:]
        bounds=[max(0,d) for d in self.column_degrees(A)]
        budget=sum(bounds)
        result=self.column_basis_full_row_rank(upper)
        B1,N,G=result['basis'],result['kernel'],result['right_factor']
        upper_bounds=[max(0,d) for d in self.column_degrees(upper)]
        if upper_bounds!=bounds:

            
            padded=list(upper)+[[[] for _ in range(n)] for _ in range(n-m)]
            count(self.stats,'lnz_kernel_padding_slots',(n-m)*n)
            N,_=self.minimal_nullspace_basis(padded,bounds)
            count(self.stats,'lnz_full_shift_kernel_calls')
        if len(N)!=n or any(len(row)!=n-m for row in N):
            raise ArithmeticError('certified LNZ kernel dimension invariant failed')
        shifted_degrees=self.column_degrees(N,bounds)
        if sum(max(0,d) for d in shifted_degrees)>budget:
            raise ArithmeticError('LNZ full-column-shift kernel budget exceeded')
        B2=self._ambient_product(lower,N,bounds,n)
        if any(len(row)!=n-m for row in B2):
            raise ArithmeticError('certified LNZ lower block shape invariant failed')
        first_budget=sum(max(0,d) for d in self.column_degrees(B1))
        second_budget=sum(max(0,d) for d in self.column_degrees(B2))
        if first_budget>budget or second_budget>budget:
            raise ArithmeticError('LNZ recursive column-degree budget exceeded')

        
        J=coordinate_completion(self,N,n)
        if len(J)!=m:
            raise ArithmeticError('certified LNZ coordinate completion size failed')
        N0=[[a[0] if a else self.F.zero() for a in row] for row in N]
        G0=[[a[0] if a else self.F.zero() for a in row] for row in G]
        Ustar=[[self.F.one() if i==j else self.F.zero() for j in J]+N0[i]
               for i in range(n)]
        numerator_matrix=[[row[j] for j in J] for row in G0]
        numerator=uniform.constant_determinant(self.F,numerator_matrix,stats=self.stats)
        denominator=uniform.constant_determinant(self.F,Ustar,stats=self.stats)
        if self.F.is_zero(numerator):
            raise ArithmeticError('certified LNZ normalization numerator vanished')
        gamma=self.F.mul(numerator,self.inverse(denominator))
        count(self.stats,'lnz_normalization_calls')
        count(self.stats,'lnz_constant_copy_slots',n*(n-m)+m*n+n*n+m*m)

        
        determinant=self.mul(self.determinant_nonsingular(B1),self.determinant_nonsingular(B2))
        return self.scale(determinant,gamma)


def determinant_from_maximum_minor(field,certificate,*,stats=None):
    """Consume the accepted rank/minor result; retain its actual coordinates.

    The certificate must originate from certified_maximum_minor or the
    equivalent accepted proof-carrying repair path, not a caller's guess.
    """
    if certificate.get('minor_certified_nonsingular') is not True:
        raise ValueError('an established nonsingular minor certificate is required')
    rank=certificate.get('normal_rank')
    minor=certificate.get('minor')
    if type(rank) is not int or rank<0 or not isinstance(minor,list):
        raise ValueError('invalid certified minor shape')
    if len(minor)!=rank or any(len(row)!=rank for row in minor):
        raise ValueError('certified rank and minor dimensions disagree')
    return NonsingularDeterminant(field,stats).determinant_nonsingular(minor)
