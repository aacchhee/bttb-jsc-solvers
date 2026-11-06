# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial/shared. General-order canonical approximant operations used by the raw polynomial front end.

Entry points: GeneralApproximant.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.kn_services import Services
from bttb_solvers._internal.kn_polynomial import count


class GeneralApproximant(Services):
    def truncated_shift(self,a,power,limit):
        if not a or power>=limit:return []
        return [self.F.zero() for _ in range(power)]+list(a[:limit-power])

    def residual_product(self,P,A,orders):
        """JNV Lemma2.6(ii): P*A with separate output-column moduli.

        Column-linearize P, shift the paired A rows, truncate BEFORE packing,
        column-linearize that residual, multiply once at bounded degree and
        accumulate directly into output slots. Never pad to max(orders).
        """
        m,inner=self.shape(P)
        rows,n=self.shape(A)
        if m!=inner or rows!=inner or len(orders)!=n or any(type(d) is not int or d<0 for d in orders):
            raise ValueError('square left factor and one nonnegative order per right column required')
        if not n:return [[] for _ in range(m)]
        degrees=[max(0,d) for d in self.column_degrees(P)]
        budget=max(1,m,sum(degrees),sum(orders))
        block=(budget+m-1)//m
        chunks=[(i,power) for i,d in enumerate(degrees) for power in range(0,d+1,block)]
        output_chunks=[(j,power) for j,d in enumerate(orders) for power in range(0,d,block)]
        left=[[self.trim(P[r][i][power:power+block]) for i,power in chunks] for r in range(m)]
        right=[]
        for i,power in chunks:
            row=[]
            for j,start in output_chunks:
                row.append(self.trim([A[i][j][q-power]
                        if q<orders[j] and 0<=q-power<len(A[i][j]) else self.F.zero()
                        for q in range(start,start+block)]))
            right.append(row)
        if not output_chunks:return [[[] for _ in range(n)] for _ in range(m)]
        product=self.pm_mul(left,right)
        out=[[[self.F.zero() for _ in range(d)] for d in orders] for _ in range(m)]
        for r in range(m):
            for k,(j,power) in enumerate(output_chunks):
                for q,value in enumerate(product[r][k]):
                    if power+q<orders[j]:
                        out[r][j][power+q]=self.F.add(out[r][j][power+q],value)
        count(self.stats,'jnv_residual_packed_cells',
              block*(m*len(chunks)+len(chunks)*len(output_chunks))+m*sum(orders))
        return [[self.trim(a) for a in row] for row in out]

    def reduce_columns(self,A,orders,shifts):
        """JNV Algorithm4; orders must be positive and nonincreasing."""
        m,n=self.shape(A)
        if n<m or len(orders)!=n or len(shifts)!=m or any(type(d) is not int or d<1 for d in orders):
            raise ValueError('ReduceColDim requires n>=m and positive orders')
        if any(orders[i]<orders[i+1] for i in range(n-1)):
            raise ValueError('ReduceColDim orders must be nonincreasing')
        if any(type(s) is not int for s in shifts):raise ValueError('integer shifts required')
        rounded=[0]*n
        for j in range(m-1,n):rounded[j]=1<<((orders[j]-1).bit_length())
        increment=rounded[m-1]-orders[m-1]
        for j in range(m-1):rounded[j]=orders[j]+increment
        shifted=[[self.shift(a[:orders[j]],rounded[j]-orders[j])
                  for j,a in enumerate(row)] for row in A]
        top=rounded[m-1]
        keep=[j for j,d in enumerate(rounded) if d>top]
        P=self.pm_basis(shifted,1,shifts)
        power=1
        while power<top:
            active=[j for j,d in enumerate(rounded) if d>=2*power]
            product=self.pm_mul(P,[[row[j] for j in active] for row in shifted],2*power)
            residual=[[self.trim(a[power:2*power]) for a in row] for row in product]
            Pi=self.pm_basis(residual,power,self.shifted_degrees(P,shifts))
            P=self.pm_mul(Pi,P)
            power*=2
        next_orders=[orders[j]-orders[m-1] for j in keep]
        if keep:
            product=self.residual_product(P,[[row[j] for j in keep] for row in A],
                                          [orders[j] for j in keep])
            residual=[[self.trim(a[orders[m-1]:]) for a in row] for row in product]
        else:
            residual=[[] for _ in range(m)]
        count(self.stats,'jnv_reduce_columns_calls')
        return residual,next_orders,self.shifted_degrees(P,shifts),P

    def known_general(self,A,orders,degrees):
        """JNV Algorithm5 with actual ReduceColDim after output linearization."""
        m,n=self.shape(A); sigma=sum(orders)
        if len(degrees)!=m or any(type(d) is not int or d<0 for d in degrees):
            raise ValueError('one nonnegative known pivot degree per row required')
        if not n:return self.identity(m)
        block=max(1,(sigma+m-1)//m)
        expanded=[]; shift=[]; ends=[]; compression=[]
        for i,degree in enumerate(degrees):
            copies=max(1,(degree+block-1)//block)
            for k in range(copies):
                expanded.append([self.truncated_shift(A[i][j],k*block,orders[j]) for j in range(n)])
                shift.append(-block if k<copies-1 else -(degree-(copies-1)*block))
                compression.append((i,k*block))
            ends.append(len(expanded)-1)
        mb=len(expanded)
        permutation=sorted(range(n),key=lambda j:(-orders[j],j))
        expanded=[[row[j] for j in permutation] for row in expanded]
        sorted_orders=[orders[j] for j in permutation]
        if n>=mb:
            residual,res_orders,res_shift,R1=self.reduce_columns(expanded,sorted_orders,shift)
        else:
            residual,res_orders,res_shift,R1=expanded,sorted_orders,shift,self.identity(mb)
        if res_orders:
            columns=[]; linear_orders=[]; extra=[]
            for j,order in enumerate(res_orders):
                alpha=(order-1)//block
                if alpha<=1:
                    columns.append([row[j] for row in residual]); linear_orders.append(order)
                else:
                    start=len(columns)
                    for k in range(alpha):
                        columns.append([row[j][k*block:(k+2)*block] for row in residual])
                        linear_orders.append(2*block if k<alpha-1 else order-(alpha-1)*block)
                    extra.extend(start+k+1 for k in range(alpha-1))
            linear=self.transpose(columns)
            linear.extend([[self.F.one()] if j==column else [] for j in range(len(columns))]
                          for column in extra)
            target=max(linear_orders)
            linear=[[self.shift(a,target-linear_orders[j]) for j,a in enumerate(row)] for row in linear]
            W=self.pm_basis(linear,target,res_shift+[-block]*len(extra))
            R2=[row[:mb] for row in W[:mb]]
            combined=self.pm_mul(R2,R1)
        else:
            combined=R1
        span=max((len(a) for row in combined for a in row),default=1)
        result=[[[self.F.zero() for _ in range(degrees[j]+span)] for j in range(m)] for _ in range(m)]
        for i,index in enumerate(ends):
            for k,(j,power) in enumerate(compression):
                for q,value in enumerate(combined[index][k]):
                    result[i][j][power+q]=self.F.add(result[i][j][power+q],value)
        result=[[self.trim(a) for a in row] for row in result]
        leading=[[row[j][degrees[j]] if len(row[j])>degrees[j] else self.F.zero()
                  for j in range(m)] for row in result]
        inverse=self.inverse_constant(leading)
        count(self.stats,'jnv_known_general_calls')
        return self.degree_budget_product([[[a] for a in row] for row in inverse],result,[0]*m)

    def canonical_general(self,A,orders,shifts):
        """JNV Algorithm6: arbitrary constraint count, orders and shifts."""
        m,n=self.shape(A)
        if not m or len(orders)!=n or len(shifts)!=m or any(type(d) is not int or d<0 for d in orders) or any(type(s) is not int for s in shifts):
            raise ValueError('matching shapes, nonnegative integer orders, integer shifts required')
        active=[j for j,d in enumerate(orders) if d]
        if len(active)!=n:
            return self.canonical_general([[row[j] for j in active] for row in A],
                                          [orders[j] for j in active],shifts)
        sigma=sum(orders)
        if not sigma:return self.identity(m)
        A=[[self.trim(a[:orders[j]]) for j,a in enumerate(row)] for row in A]
        count(self.stats,'jnv_general_calls')
        if sigma<=m:return self.small_order_canonical(A,orders,shifts)
        if n>=m:
            permutation=sorted(range(n),key=lambda j:(-orders[j],j))
            sorted_A=[[row[j] for j in permutation] for row in A]
            sorted_orders=[orders[j] for j in permutation]
            residual,new_orders,new_shift,P1=self.reduce_columns(sorted_A,sorted_orders,shifts)
            P2=self.canonical_general(residual,new_orders,new_shift)
        else:
            budget=sigma//2; first=[]; second=[]
            for d in orders:
                take=min(d,budget);first.append(take);second.append(d-take);budget-=take
            left=[j for j,d in enumerate(first) if d]
            P1=self.canonical_general([[row[j] for j in left] for row in A],
                                      [first[j] for j in left],shifts)
            right=[j for j,d in enumerate(second) if d]

            product=self.residual_product(P1,[[row[j] for j in right] for row in A],
                                          [orders[j] for j in right])
            residual=[[self.trim(a[first[j]:orders[j]]) for a,j in zip(row,right)] for row in product]
            P2=self.canonical_general(residual,[second[j] for j in right],self.shifted_degrees(P1,shifts))
        degrees=[len(P1[i][i])+len(P2[i][i])-2 for i in range(m)]
        return self.known_general(A,orders,degrees)
