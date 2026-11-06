# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.halfstrip_core.dense_backend import terminal_method, selection
from bttb_solvers._internal.kn_approximant import Approximant
from bttb_solvers._internal.kn_polynomial import count
import time

METHOD='KN_NATIVE_THREE_PHASE_STRASSEN_SUPPLIED_GENERATORS'


def phase_call(stats, name, operation, *args, **kwargs):
    """Completed-call telemetry only; this cannot localize a killed process."""
    if stats is None:
        return operation(*args, **kwargs)
    wall, cpu = time.perf_counter(), time.process_time()
    result = operation(*args, **kwargs)
    phases = stats.setdefault('completed_phase_seconds', {})
    record = phases.setdefault(name, {'calls':0,'wall_seconds':0.0,'cpu_seconds':0.0})
    record['calls'] += 1
    record['wall_seconds'] += time.perf_counter()-wall
    record['cpu_seconds'] += time.process_time()-cpu
    return result


class _PreparedClassicalDense:
    """One explicit paid-base preparation, owned by one physical attempt."""
    def __init__(self, service, g, h):
        self.service, self.g, self.h = service, g, h
        self.data = service._prepare_paid_dense(g, h)

    def solve_stein(self, g, h, rhs=None, *, homogeneous=False):
        if g is not self.g or h is not self.h:
            raise ValueError('prepared terminal requires its original generator objects')
        if selection()[0] != 'classical':
            raise ValueError('prepared classical terminal used outside its backend scope')
        rhs = [self.service.F.zero() for _ in g] if rhs is None else list(rhs)
        if len(rhs) != len(g):
            raise ValueError('RHS dimension mismatch')
        answer = self.service._apply_paid_dense(self.data, rhs, homogeneous)
        answer.update(method=terminal_method(), paid_dense_base=True)
        return answer

    def invalidate(self):
        self.__dict__.clear()
        self.invalidated = True


class Services(Approximant):
    def prepare_classical_base(self, g, h):
        """Explicit lifetime, no generator-value cache and no structured fallback.

        Caller relinquishes generator mutation until this attempt ends.
        """
        if selection()[0] == 'classical' and len(g) <= len(g[0]):
            return _PreparedClassicalDense(self, g, h)
        return self

    def invalidate(self):

        pass

    def _prepare_paid_dense(self, g, h):
        d,beta=len(g),len(g[0])
        if d>beta:
            raise ValueError('physical dense reconstruction requires d<=beta')
        count(self.stats,'kn_paid_dense_preparations')
        A=uniform.matrix_product(self.F,g,self.transpose(h),stats=self.stats)
        for i in range(1,d):
            for j in range(1,d):
                A[i][j]=self.F.add(A[i][j],A[i-1][j-1])
        result=uniform.gauss_jordan(self.F,A,stats=self.stats)
        rank=result['rank']; scale=result['d']; permutation=result['permutation']
        U=[[self.F.zero() for _ in range(d)] for _ in range(d)]
        for i in range(d):
            U[i][i]=scale
            for j in range(rank):
                U[i][j]=result['columns'][i][j]
        PA=[A[permutation[i] if permutation is not None else i] for i in range(d)]
        transformed=uniform.matrix_product(self.F,U,PA,stats=self.stats)
        inverse=self.inverse(scale)
        transformed=[[self.F.mul(inverse,x) for x in row] for row in transformed]
        pivots=[]; start=0
        for row in transformed[:rank]:
            pivot=next(j for j in range(start,d) if not self.F.is_zero(row[j]))
            pivots.append(pivot); start=pivot+1
        return d,rank,permutation,U,inverse,transformed,pivots

    def _apply_paid_dense(self, prepared, rhs, homogeneous):
        d,rank,permutation,U,inverse,transformed,pivots = prepared
        count(self.stats,'kn_paid_dense_applications')
        if homogeneous:
            compressed=[]
            for j in range(d):
                if j not in pivots:
                    u=[self.F.zero() for _ in range(d)]; u[j]=self.F.one()
                    for i,k in enumerate(pivots):
                        u[k]=self.F.neg(transformed[i][j])
                    compressed.append({'u':u,'cap':1})
            return {'status':'NULLSPACE','nullity':d-rank,'compressed':compressed,'dimension':d}
        Pb=[[rhs[permutation[i] if permutation is not None else i]] for i in range(d)]
        y=uniform.matrix_product(self.F,U,Pb,stats=self.stats)
        y=[self.F.mul(inverse,row[0]) for row in y]
        if any(not self.F.is_zero(x) for x in y[rank:]):
            return {'status':'INCONSISTENT'}
        u=[self.F.zero() for _ in range(d)]
        for i,j in enumerate(pivots):
            u[j]=y[i]
        return {'status':'CONSISTENT','solution':u}

    def paid_dense(self, g, h, rhs, homogeneous):
        return self._apply_paid_dense(self._prepare_paid_dense(g,h),rhs,homogeneous)

    def solve_stein(self, g, h, rhs=None, *, homogeneous=False):
        d=len(g)
        if not d or len(h)!=d:
            raise ValueError('matching positive dimensions required')
        beta=len(g[0])
        if any(len(row)!=beta for row in g+h):
            raise ValueError('generator width mismatch')
        rhs=[self.F.zero() for _ in range(d)] if rhs is None else list(rhs)
        if len(rhs)!=d:
            raise ValueError('RHS dimension mismatch')
        if d<=beta:
            answer=self.paid_dense(g,h,rhs,homogeneous)
            answer.update({'method':terminal_method(),'paid_dense_base':True})
            return answer
        if beta==0:
            if homogeneous:
                return {'status':'NULLSPACE','nullity':d,'dimension':d,
                        'compressed':[{'u':[self.F.one()]+[self.F.zero() for _ in range(d-1)],'cap':d}],
                        'method':terminal_method(),'paid_dense_base':False}
            answer=({'status':'INCONSISTENT'} if any(not self.F.is_zero(x) for x in rhs)
                    else {'status':'CONSISTENT','solution':[self.F.zero() for _ in range(d)]})
            answer.update({'method':terminal_method(),'paid_dense_base':False})
            return answer
        f=[[row[j] for row in g] for j in range(beta)]
        hbar=[[row[j] for row in h] for j in range(beta)]
        P,mu,pv=phase_call(self.stats,'kn_phase1_vector_mpade',self.vector_mpade,f,d,None if homogeneous else rhs)
        delta=[len(P[i][i])-1 for i in range(beta)]
        if any(x<0 for x in delta) or sum(delta)>d:
            raise AssertionError('KN phase1 linear degree budget lost')
        Qbar=[[self.trim(list(reversed(a+[self.F.zero() for _ in range(delta[j]+1-len(a))])))
               for j,a in enumerate(row)] for row in P]
        Fvector=phase_call(self.stats,'kn_phase2_generator_vector',self.rational_vector_xd,Qbar,hbar,d,delta)
        admissible=(len(mu)==1 and self.F.is_zero(self.F.sub(mu[0],self.F.one()))
                    and all(len(a)<=d for a in pv))
        if admissible and not homogeneous:
            reversed_pv=[list(reversed(a+[self.F.zero() for _ in range(d-len(a))])) for a in pv]
            w=phase_call(self.stats,'kn_phase2_rhs_vector',self.rational_vector_xd,Qbar,reversed_pv,d,delta)
        else:
            w=[[] for _ in range(beta)]
        pbar,caps,particular=phase_call(self.stats,'kn_phase3_simultaneous_mpade',self.simultaneous_mpade,
                                     Fvector,d,[max(0,d-x) for x in delta],w)
        metadata={'method':terminal_method(),'phase_degree_budget':sum(delta),'paid_dense_base':False}
        if homogeneous:
            compressed=[]
            for a,cap in zip(pbar,caps):
                degree=len(a)-1
                if degree<0 or cap<=0 or degree+cap>d:
                    raise AssertionError('invalid KN compressed solution basis')
                u=self.shift(list(reversed(a)),d-degree-cap)
                compressed.append({'u':u+[self.F.zero() for _ in range(d-len(u))],'cap':cap})
            return dict(metadata,status='NULLSPACE',nullity=sum(caps),compressed=compressed,dimension=d)
        if not admissible or particular is None:
            return dict(metadata,status='INCONSISTENT')
        u=list(reversed(particular+[self.F.zero() for _ in range(d-len(particular))]))
        return dict(metadata,status='CONSISTENT',solution=u)

    def scalar_to_stein(self, left, right_transpose):
        """algorithm c=1 conversion with actual first-row acquisition."""
        d=len(left); alpha=len(left[0]) if d else 0
        if not d or any(len(row)!=alpha for row in left) or len(right_transpose)!=alpha or any(len(row)!=d for row in right_transpose):
            raise ValueError('scalar generator dimensions mismatch')
        h=self.transpose(right_transpose) if alpha else [[] for _ in range(d)]
        rotated=left[1:]+left[:1]
        v=[self.F.zero() for _ in range(d)]
        for j in range(alpha):
            product=self.mul([row[j] for row in reversed(rotated)],[row[j] for row in h])
            for i in range(d):
                if d-1+i<len(product):
                    v[i]=self.F.add(v[i],product[d-1+i])
        stein_g=[list(row)+[self.F.one() if i==0 else self.F.zero()]
                 for i,row in enumerate(reversed(rotated))]
        rv=[self.F.zero()]+list(reversed(v))[:-1]
        stein_h=[list(row)+[rv[i]] for i,row in enumerate(reversed(h))]
        count(self.stats,'kn_scalar_adapter_copy_cells',2*d*(alpha+1))
        return stein_g,stein_h

    def solve_scalar(self, left, right_transpose, rhs=None, *, transposed=False, homogeneous=False):
        g,h=self.scalar_to_stein(left,right_transpose)
        if transposed:
            g,h=h,g
        answer=self.solve_stein(g,h,None if rhs is None else list(reversed(rhs)),homogeneous=homogeneous)
        if answer['status']=='CONSISTENT':
            answer['solution'].reverse()
        if answer['status']=='NULLSPACE':
            answer['coordinate_reversal']=True
        return answer
