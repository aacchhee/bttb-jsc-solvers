# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""KN supplied-generator algebra.

Role: Trial/shared. Canonical/known-degree approximant, vector approximation and simultaneous Hermite-Pade routines for the supplied-generator construction.

Entry points: Approximant.

References: Khichane and Neiger (2026); see REFERENCES.md for the precise
approximant/kernel sources and preprint scope.
Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from heapq import heappush, heappop
from bttb_solvers._internal.kn_polynomial import count
from bttb_solvers._internal.kn_kernel import Kernel


class Approximant(Kernel):
    def known_degree_basis(self, A, orders, degrees):
        m,n = len(A),len(orders)
        if not m or any(len(row)!=n for row in A) or any(type(o) is not int or o<0 for o in orders):
            raise ValueError('matching shape and nonnegative integer orders required')
        sigma=sum(orders)
        if not n:
            return self.identity(m)
        if any(type(x) is not int or x<0 for x in degrees) or len(degrees)!=m:
            raise ValueError('nonnegative pivot degrees required')
        if n>=m:
            target=max(orders)
            shifted=[[self.shift(a[:orders[j]],target-orders[j])
                      for j,a in enumerate(row)] for row in A]
            R=self.pm_basis(shifted,target,[-x for x in degrees])
            leading=[[row[j][degrees[j]] if len(row[j])>degrees[j] else self.F.zero()
                      for j in range(m)] for row in R]
            inv=self.inverse_constant(leading)
            return self.degree_budget_product([[[a] for a in row] for row in inv],R,[0]*m)
        block=max(1,(sigma+m-1)//m)
        expanded=[]; shifts=[]; ends=[]; compression=[]
        for i,di in enumerate(degrees):
            copies=max(1,(di+block-1)//block)
            for k in range(copies):
                expanded.append([self.shift(A[i][j],k*block)[:orders[j]] for j in range(n)])
                shifts.append(-block if k<copies-1 else -(di-(copies-1)*block))
                compression.append((i,k*block))
            ends.append(len(expanded)-1)
        mb=len(expanded); columns=[]; linear_orders=[]; extra=[]
        for j,order in enumerate(orders):
            alpha=(order-1)//block
            if alpha<=1:
                columns.append([expanded[i][j] for i in range(mb)])
                linear_orders.append(order)
            else:
                start=len(columns)
                for k in range(alpha):
                    columns.append([expanded[i][j][k*block:(k+2)*block] for i in range(mb)])
                    linear_orders.append(2*block if k<alpha-1 else order-(alpha-1)*block)
                extra.extend(start+k+1 for k in range(alpha-1))
        linear=self.transpose(columns)
        linear.extend([[self.F.one()] if j==column else [] for j in range(len(columns))]
                      for column in extra)
        target=max(linear_orders)
        linear=[[self.shift(a,target-linear_orders[j]) for j,a in enumerate(row)] for row in linear]
        W=self.pm_basis(linear,target,shifts+[-block]*len(extra))

        
        R=[[[self.F.zero() for _ in range(degrees[j]+target+1)]
            for j in range(m)] for _ in range(m)]
        for i,index in enumerate(ends):
            for k,(j,power) in enumerate(compression):
                for q,value in enumerate(W[index][k]):
                    R[i][j][power+q]=self.F.add(R[i][j][power+q],value)
        count(self.stats,'kn_canonical_reassembly_cells',
              m*sum(degree+target+1 for degree in degrees))
        R=[[self.trim(a) for a in row] for row in R]
        leading=[[row[j][degrees[j]] if len(row[j])>degrees[j] else self.F.zero()
                  for j in range(m)] for row in R]
        inv=self.inverse_constant(leading)
        return self.degree_budget_product([[[a] for a in row] for row in inv],R,[0]*m)

    def small_order_canonical(self, A, orders, shifts):
        from bttb_solvers._internal.kn_krylov import canonical_leaf
        return canonical_leaf(self,A,orders,shifts)

    def reference_small_order_canonical(self, A, orders, shifts):
        """Existing sorted first-dependency algorithm; classical cost exposed."""
        F=self.F; m=len(A); sigma=sum(orders)
        heap=[]; selected=[]; echelon=[]; B=[None]*m
        count(self.stats,'kn_classical_krylov_calls')
        count(self.stats,'kn_classical_krylov_scalar_bound',m*sigma*sigma+sigma**3)
        for i in range(m):
            heappush(heap,(shifts[i],i,0))
        while heap:
            _,i,power=heappop(heap)
            v=[]
            for j,d in enumerate(orders):
                a=self.shift(A[i][j],power)[:d]
                v.extend(a+[F.zero() for _ in range(d-len(a))])
            expr={}
            for col,row,coeff in echelon:
                fac=v[col]
                if not F.is_zero(fac):
                    v=[F.sub(x,F.mul(fac,y)) for x,y in zip(v,row)]
                    for k,c in coeff.items():
                        expr[k]=F.add(expr.get(k,F.zero()),F.mul(fac,c))
            pivot=next((j for j in range(sigma) if not F.is_zero(v[j])),None)
            if pivot is None:
                row=[[] for _ in range(m)]; row[i]=self.shift([F.one()],power)
                for k,c in expr.items():
                    if not F.is_zero(c):
                        coord,degree=selected[k]
                        row[coord]=self.add(row[coord],self.shift([F.neg(c)],degree))
                B[i]=row
            else:
                inv=self.inverse(v[pivot]); k=len(selected); selected.append((i,power))
                coeff={k:inv}
                for j,c in expr.items():
                    coeff[j]=F.sub(coeff.get(j,F.zero()),F.mul(inv,c))
                echelon.append((pivot,[F.mul(x,inv) for x in v],coeff))
                heappush(heap,(shifts[i]+power+1,i,power+1))
        return B

    def canonical(self, A, orders, shifts):
        m=len(A)
        if len(orders)>4:
            raise ValueError('KN constant-column service supports at most four columns')
        if not m or len(shifts)!=m or any(type(s) is not int for s in shifts) or any(len(row)!=len(orders) for row in A):
            raise ValueError('approximant dimensions mismatch')
        if any(type(o) is not int or o<0 for o in orders):
            raise ValueError('nonnegative integer orders required')
        sigma=sum(orders)
        count(self.stats,'kn_canonical_calls')
        if not sigma:
            return self.identity(m)
        A=[[self.trim(a[:orders[j]]) for j,a in enumerate(row)] for row in A]
        if sigma<=m:
            return self.small_order_canonical(A,orders,shifts)
        budget=sigma//2; first=[]; second=[]
        for d in orders:
            take=min(d,budget); first.append(take); second.append(d-take); budget-=take
        left=[j for j,d in enumerate(first) if d]
        P1=self.canonical([[row[j] for j in left] for row in A],[first[j] for j in left],shifts)
        right=[j for j,d in enumerate(second) if d]
        product=self.degree_budget_product(P1,[[row[j] for j in right] for row in A])
        residual=[[a[first[j]:orders[j]] for a,j in zip(row,right)] for row in product]
        P2=self.canonical(residual,[second[j] for j in right],self.shifted_degrees(P1,shifts))
        degrees=[len(P1[i][i])+len(P2[i][i])-2 for i in range(m)]
        return self.known_degree_basis(A,orders,degrees)

    def vector_mpade(self, polynomials, order, rhs=None):
        a=len(polynomials)
        if type(order) is not int or order<1 or a<1:
            raise ValueError('positive order and width required')
        v=self.trim([] if rhs is None else rhs)
        f=[self.trim(q[:order]) for q in polynomials]; shifts=[0]*a
        augmented=bool(v)
        if augmented:
            f.append(self.scale(v,self.F.neg(self.F.one()))); shifts.append(order)
        A=[[q] for q in f]+[[self.shift([self.F.one()],order)]]
        tau=max(shifts)-min(shifts)+2*order
        Q=self.transpose(self.canonical(A,[tau],shifts+[min(shifts)]))
        size=len(f); Q=[row[:size] for row in Q[:size]]
        if not augmented:
            return Q,[self.F.one()],[[] for _ in range(a)]
        return [row[:a] for row in Q[:a]],Q[a][a],[Q[i][a] for i in range(a)]

    def negative_basis(self, A, order, shifts):
        W=self.canonical(A,[order]*len(A[0]),shifts)
        degrees=self.shifted_degrees(W,shifts)
        indices=[i for i,x in enumerate(degrees) if x<0]
        return [W[i] for i in indices],[degrees[i] for i in indices]

    def recursive_shpade(self, S, order, T, N):
        t,n=len(T),len(N)
        if not n:
            return self.identity(t),[-x for x in T]
        if n<=t:
            B=[[self.scale(a,self.F.neg(self.F.one())) for a in row] for row in S]
            B+=self.identity(n)
            W,degrees=self.negative_basis(B,order,[-x for x in T+N])
            return [row[:t] for row in W],degrees
        mid=n//2
        L1,d1=self.recursive_shpade([row[:mid] for row in S],order,T,N[:mid])
        L2,d2=self.recursive_shpade([row[mid:] for row in S],order,T,N[mid:])
        R=[row+list(row) for row in self.identity(t)]
        R += [[self.scale(a,self.F.neg(self.F.one())) for a in row]+[[] for _ in range(t)] for row in L1]
        R += [[[] for _ in range(t)]+[self.scale(a,self.F.neg(self.F.one())) for a in row] for row in L2]
        W,degrees=self.negative_basis(R,max(T),[-x for x in T]+d1+d2)
        return [row[:t] for row in W],degrees

    def simultaneous_mpade(self, vector, order, bounds, rhs=None):
        rhs=[[] for _ in vector] if rhs is None else rhs
        if type(order) is not int or order<1 or any(type(b) is not int or not 0<=b<=order for b in bounds):
            raise ValueError('positive integer order and integer bounds within order required')
        if any(len(a)>order for a in vector+rhs):
            raise ValueError('simultaneous inputs must have degree below order')
        if len(rhs)!=len(vector) or len(bounds)!=len(vector):
            raise ValueError('simultaneous dimensions mismatch')
        S=[[list(a) for a in vector],[self.scale(a,self.F.neg(self.F.one())) for a in rhs]]
        rows,degrees=self.recursive_shpade(S,order,[order,1],list(bounds))
        L=self.transpose(rows) if rows else [[],[]]
        caps=[-x for x in degrees]
        support=[j for j,a in enumerate(L[1]) if a]
        if not support:
            return L[0],caps,None
        pivot=min(support,key=lambda j:(-caps[j],j))
        if len(L[1][pivot])!=1:
            raise AssertionError('inhomogenizing component must be constant')
        particular=self.scale(L[0][pivot],self.inverse(L[1][pivot][0]))
        denominators=[]; outcaps=[]
        for j in range(len(caps)):
            if j!=pivot:
                c=L[1][j][0] if L[1][j] else self.F.zero()
                denominators.append(self.add(L[0][j],self.scale(particular,self.F.neg(c))))
                outcaps.append(caps[j])
        return denominators,outcaps,particular
