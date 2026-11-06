# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.banded_bttb.symbol import BandedSymbol, det_symbol_poly, find_mu
from bttb_solvers._internal.banded_bttb import repair
from bttb_solvers._internal.banded_bttb.circulant import frequency_symbol_lags, accumulate_G, corners, apply_C_inv, _dft, transform_route
from bttb_solvers._internal.banded_bttb.woodbury import merged_correction, inner_matrix, factor
from bttb_solvers._internal.banded_bttb.toeplitz import toeplitz_solve
from bttb_solvers._internal.banded_bttb.errors import SingularInnerMatrix
from bttb_solvers._internal.banded_bttb.field import int_element
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier


class ToeplitzInverseFactors:
    """cyclic-displacement C(g1)U(h1)+C(g2)U(h2), every nonsingular Toeplitz input."""
    def __init__(self,F,lag,n):
        if n<1 or len(lag)!=2*n-1:raise ValueError('Toeplitz dimensions')
        self.F,self.n=F,n
        r=[lag[n-1+n-1-j] for j in range(n)]
        c=[F.sub(lag[n-1+i-(n-1)],lag[n-1+i+1]) for i in range(n-1)]+[F.zero()]
        en=[F.zero()]*n;en[-1]=F.one()
        e0=[F.zero()]*n;e0[0]=F.one()
        shift=lambda v:[v[-1]]+v[:-1]
        alpha=toeplitz_solve(F,lag,n,en)
        gamma=toeplitz_solve(F,lag,n,c)
        beta=toeplitz_solve(F,lag,n,shift(r),transpose=True)
        delta=toeplitz_solve(F,lag,n,e0,transpose=True)
        self.factors=[(shift(alpha),beta),(shift(gamma),delta)]
        self.multiply=make_multiplier(F)
        self.actions={'forward':0,'transpose':0,'scalar_convolutions':0}

    def apply(self,v,*,transpose=False):
        F,n=self.F,self.n
        if len(v)!=n:raise ValueError('Toeplitz RHS dimension')
        out=[F.zero()]*n
        for g,h in self.factors:
            if transpose:
                a=self.multiply(v,list(reversed(g)))
                w=[F.add(a[n-1+i],a[i-1] if i else F.zero()) for i in range(n)]
                value=self.multiply(h,w)[:n]
            else:
                w=self.multiply(v,list(reversed(h)))[n-1:2*n-1]
                a=self.multiply(g,w)
                value=[F.add(a[i],a[i+n] if i+n<len(a) else F.zero()) for i in range(n)]
            out=[F.add(a,b) for a,b in zip(out,value)]
        self.actions['transpose' if transpose else 'forward']+=1
        self.actions['scalar_convolutions']+=4
        return out


class CirculantCorrectionPrepared:
    def __init__(self,F,n,k,raw,omega,*,arm='retained',dft='auto'):
        if arm not in ('original','reuse_inner','retained'):raise ValueError('classical HB arm')
        if type(n) is not int or type(k) is not int or n<3 or not 1<=2*k<n:
            raise ValueError('narrow HB dimensions')
        if (F.q-1)%n or (F.q-1)//n<=2*k*n:raise ValueError('HB root/coset capacity')
        if any(F.eq(F.pow(omega,j),F.one())!=(j==n) for j in range(1,n+1)):
            raise ValueError('supplied omega exact order required')
        self.F,self.n,self.k,self.omega,self.arm=F,n,k,omega,arm
        self.sym=BandedSymbol(F,n,k,tuple(tuple(raw[(a,b)] for b in range(1-n,n)) for a in range(-k,k+1)))
        self.apply_route=transform_route(n,dft,phase='apply')
        setup_route=transform_route(n,dft,k=k,phase='setup')
        s=0;X=Y=None
        f=det_symbol_poly(F,self.sym)
        if not f:
            rank,witness=repair.normal_rank(F,self.sym,return_point=True)
            s=n-rank
            E,X,Y=repair.find_E0(F,self.sym,s,_normal_rank_witness=(rank,witness))
            self.sym=self.sym.with_lag0_shift(E)
            f=det_symbol_poly(F,self.sym)
        search={}
        self.mu=find_mu(F,self.sym,n,omega,route='gcd',f=f,stats=search)
        self.frequency_lags=frequency_symbol_lags(F,self.sym,n,self.mu,omega)
        gstats={}
        G=accumulate_G(F,self.sym,n,k,self.mu,omega,stats=gstats,X=X,Y=Y,
                       dft=setup_route,frequency_lags=self.frequency_lags)
        self.corr=merged_correction(F,self.sym,n,X,Y,self.mu)
        M=inner_matrix(F,G,n,n,k,self.corr,corners(F,G,n,n,k),gstats.get('projections'))
        self.fact=factor(F,M)
        self.symbol_factors=None
        if arm=='retained':
            self.symbol_factors=[ToeplitzInverseFactors(F,lag,n) for lag in self.frequency_lags]
        self.metadata={'method':'CC_'+arm.upper(),'hb_arm':arm,
            'fully_kn_free':True,'supplied_root':True,'field_construction':False,
            'inner_preparations':1,'physical_order':n*n,'repair_rank':s,
            'represented_order':self.corr.r,'query_transform':self.apply_route,
            'inner_factor_reuse':True,'same_preparation':True,
            'structured_symbol_factors':arm=='retained',
            'duplicate_factor_acquisition_paid':arm=='retained',
            'forward_calls':0,'transpose_calls':0}
        self.statistics={'carrier':gstats,'search':search,'symbol_factor_count':n if self.symbol_factors else 0}

    def _scale(self,v,inverse=False):
        F,n=self.F,self.n;step=F.inv(self.mu) if inverse else self.mu
        value=F.one();out=[]
        for i in range(n):
            out.extend(F.mul(value,x) for x in v[i*n:(i+1)*n]);value=F.mul(value,step)
        return out

    def _action(self,v,transpose):
        F,n=self.F,self.n
        if self.symbol_factors is None:
            return apply_C_inv(F,self.sym,n,self.mu,self.omega,v,transpose=transpose,
                              dft=self.apply_route,frequency_lags=self.frequency_lags)
        inverse_n=F.inv(int_element(F,n));inverse_omega=F.inv(self.omega)
        frequencies=[[F.zero()]*n for _ in range(n)]
        for a in range(n):
            transformed=_dft(F,[v[t*n+a] for t in range(n)],inverse_omega,self.apply_route)
            for j in range(n):frequencies[j][a]=F.mul(inverse_n,transformed[j])
        solved=[self.symbol_factors[(-j)%n if transpose else j].apply(frequencies[j],transpose=transpose) for j in range(n)]
        result=[F.zero()]*(n*n)
        for a in range(n):
            transformed=_dft(F,[solved[j][a] for j in range(n)],self.omega,self.apply_route)
            for t in range(n):result[t*n+a]=transformed[t]
        return result

    def query(self,rhs,*,transpose=False):
        if len(rhs)!=self.n*self.n:raise ValueError('physical RHS dimension')
        F=self.F
        y=self._action(self._scale(rhs,inverse=transpose),transpose)
        if transpose:
            z=self.fact.solve_transpose(self.corr.apply_UT(y));lift=self.corr.apply_V(z)
        else:
            z=self.fact.solve(self.corr.applyT_V(y));lift=self.corr.apply_U(z)
        value=self._action(lift,transpose)
        result=self._scale([F.add(a,b) for a,b in zip(y,value)],inverse=not transpose)
        self.metadata['transpose_calls' if transpose else 'forward_calls']+=1
        return result

    def solve(self,rhs):return self.query(rhs)
    def query_many(self,rhs,*,transpose=False):return [self.query(v,transpose=transpose) for v in rhs]

    def invalidate(self):
        self.__dict__.clear()
        self.invalidated=True


def prepare_cc(F,n,k,raw,omega,*,arm='retained',dft='auto'):
    return CirculantCorrectionPrepared(F,n,k,raw,omega,arm=arm,dft=dft)
