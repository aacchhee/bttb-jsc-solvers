# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from contextlib import contextmanager
import time
from bttb_solvers._internal.halfstrip_core.dense_backend import dense_scope
from bttb_solvers._internal.halfstrip_core.mosaic import run_mosaic, solve_mosaic
from bttb_solvers._internal.halfstrip.convolve import next_pow2


def causal_solve(F,sym,leading,backend,cnt,forcing,*,transpose=False,stats=None):
    """Solve lower block Toeplitz horizon h; forcing[t][response][spatial].

    transpose means coefficients F_d^T, not temporal transposition. Callers
    reverse temporal rows separately to solve an upper block Toeplitz system.
    """
    h=len(forcing);n=sym.n
    if not h:return []
    q=len(forcing[0]);C=[[list(v) for v in row] for row in forcing]
    if any(len(row)!=q or any(len(v)!=n for v in row) for row in C):
        raise ValueError('ragged finite-horizon forcing')
    X=[[None for _ in range(q)] for _ in range(h)]
    kernel=[None]+[tuple(reversed(sym.lag(d-sym.w))) if transpose else sym.lag(d-sym.w)
                   for d in range(1,h)]
    st={} if stats is None else stats
    st['calls']=st.get('calls',0)+1;st['responses']=st.get('responses',0)+q
    st['leaf_applications']=st.get('leaf_applications',0)+q*h
    apply=leading.apply_transpose if transpose else leading.apply
    def cdq(lo,hi):
        if hi-lo==1:
            with cnt.stage('apply_inv'):
                for c in range(q):X[lo][c]=apply(C[lo][c])
            return
        mid=(lo+hi)//2;cdq(lo,mid);length=hi-lo;left=mid-lo;fresh=False
        with cnt.stage('merge'):
            for c in range(q):
                out,now=backend.merge_window([X[lo+a][c] for a in range(left)],
                                             kernel[:length],n,left,length,left)
                fresh=fresh or now
                for tau,vec in out.items():
                    C[lo+tau][c]=[F.sub(a,b) for a,b in zip(C[lo+tau][c],vec)]
            nt=next_pow2(left+length-1);ns=next_pow2(3*n-2)
            cnt.charge_conv2d(q,nt,ns,kernel_cells=nt*ns if fresh else 0,
                              raw=(left+length-1,3*n-2),engine=backend.kind,kernel_fresh=fresh)
        st['merges']=st.get('merges',0)+1;cdq(mid,hi)
    cdq(0,h)
    return X


class StructuredTerminal:
    def __init__(self,F,sym,leading,backend,cnt,Xh,*,inverse_responses=None):
        self.F,self.sym,self.leading,self.backend,self.cnt=F,sym,leading,backend,cnt
        self.source_layout = 'n_impulse' if inverse_responses is not None else 'wn_homogeneous'
        n,w=sym.n,sym.w;self.n=n*w
        self.stats={'phase_seconds':{},'causal':{},'dense_scope_ledger':{},'query_count':0}
        with dense_scope('classical',1,polynomial_entry_policy='off') as ledger:
            with self._phase('generator_extraction'):
                if inverse_responses is not None:
                    selected = {d:[[inverse_responses[d][j][i] for j in range(n)]
                                   for i in range(n)] for d in range(n-w+1,n+w)}
                else:

                    
                    forcing=[[[Xh[n+a][(w-1-t)*n+i][s] for i in range(n)]
                               for a in (0,w-1) for s in range(n)] for t in range(w)]
                    solved=causal_solve(F,sym,leading,backend,cnt,forcing,transpose=True,stats=self.stats['causal'])
                    selected={}
                    for edge,a in enumerate((0,w-1)):
                        for b in range(w):
                            degree=n+a-b
                            block=[[solved[w-1-b][edge*n+s][i] for i in range(n)] for s in range(n)]
                            if degree in selected:
                                if any(not F.is_zero(F.sub(x,y)) for r,t in zip(selected[degree],block) for x,y in zip(r,t)):
                                    raise ArithmeticError('first/last K block-row overlap disagrees')
                            else:selected[degree]=block
                self.selected=selected
                self.dictionary=selected
            with self._phase('mosaic_prepare'):
                self.mosaic=run_mosaic(F,[w]*n,lambda i,j,k:selected[n+k][i][j])
                self.det=F.mul(self.mosaic['det'],F.pow(leading.det,w))
                self.rank=self.mosaic['rank'];self.singular=self.rank<self.n
            self._record_ledger(ledger)

    @contextmanager
    def _phase(self,name):
        wall,cpu=time.perf_counter(),time.process_time()
        try:yield
        finally:
            rec=self.stats['phase_seconds'].setdefault(name,{'calls':0,'wall_seconds':0.,'cpu_seconds':0.})
            rec['calls']+=1;rec['wall_seconds']+=time.perf_counter()-wall;rec['cpu_seconds']+=time.process_time()-cpu

    def _record_ledger(self,ledger):
        for key,value in ledger.items():
            self.stats['dense_scope_ledger'][key]=self.stats['dense_scope_ledger'].get(key,0)+value

    def nullity(self):return self.n-self.rank

    def solve(self,rhs):
        if len(rhs)!=self.n:raise ValueError('terminal RHS dimension mismatch')
        if self.singular:return None
        F,n,w=self.F,self.sym.n,self.sym.w
        with dense_scope('classical',1,polynomial_entry_policy='off') as ledger:
            with self._phase('mosaic_solve'):
                shuffled=[rhs[t*n+i] for i in range(n) for t in range(w)]
                affine=solve_mosaic(F,self.mosaic,shuffled)
                u=affine['solution']
                if u is None:raise ArithmeticError('nonsingular K lacks affine solution')
            with self._phase('basis_conversion'):
                forcing=[[[u[i*w+t] for i in range(n)]] for t in range(w)]
                alpha=causal_solve(F,self.sym,self.leading,self.backend,self.cnt,forcing,stats=self.stats['causal'])
                result=[alpha[t][0][i] for t in range(w) for i in range(n)]
            self.stats['query_count']+=1;self._record_ledger(ledger)
            return result

    def summary(self):
        return {'terminal_backend':'structured_classical','terminal_engine':'uniform_fast',
                'dense_arithmetic':'classical','terminal_matrix_materialized':False,
                'identity':'Bt=K*Cw; K[a,b]=R[n+a-b]; det(Cw)=det(F0)^w',
                'source':'n length-L inverse impulses; direct selected K blocks' if self.source_layout=='n_impulse' else 'existing recurrence homogeneous responses; first/last K block rows',
                'response_layout':self.source_layout,
                'selected_blocks':len(self.selected),'r':self.sym.n,'m':self.n,'w':self.sym.w,
                'chunk_bound':self.mosaic['chunk_bound'],'approximant_order':self.mosaic['approximant_order'],
                'cost_scope':'soft O(w*n^3) with achieved recurrence NTT; oracle fields correctness only',
                'acquisition_cost':'O(w*n^2) selected-entry copies from prepared inverse responses' if self.source_layout=='n_impulse' else 'soft O(w*n^2) via 2n horizon-w transposed causal solves',
                'statistics':self.stats}


class DenseImpulseTerminal:
    """Dense K comparison arm with the identical Bt=K*Cw basis conversion."""
    def __init__(self,F,sym,leading,backend,cnt,impulse):
        from bttb_solvers._internal.halfstrip.dense import LU
        self.F,self.sym,self.leading,self.backend,self.cnt=F,sym,leading,backend,cnt
        n,w=sym.n,sym.w
        self.matrix=[[impulse[n+a-b][j][i] for b in range(w) for j in range(n)]
                     for a in range(w) for i in range(n)]
        self.factor=LU(F,self.matrix)
        self.det=F.mul(self.factor.det,F.pow(leading.det,w))
        self.singular=self.factor.singular

    def nullity(self): return self.factor.nullity()

    def solve(self,rhs):
        F,n,w=self.F,self.sym.n,self.sym.w
        u=self.factor.solve(rhs)
        forcing=[[[u[t*n+i] for i in range(n)]] for t in range(w)]
        alpha=causal_solve(F,self.sym,self.leading,self.backend,self.cnt,forcing)
        return [alpha[t][0][i] for t in range(w) for i in range(n)]
