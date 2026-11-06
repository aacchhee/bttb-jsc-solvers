# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small JSON input and execution helpers for the practitioner examples."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bttb_solvers as s
from bttb_solvers.matrices import action,banded_to_halfstrip

ALGORITHMS=('CC','CC-R','CC-E','CC-E-R','BFM','GBE','HIR','HPR','HSC','HSC-q0','Dense-LU')
DETERMINANTS={'HIR','HPR','HSC','HSC-q0','Dense-LU'}


def load(path):
    problem=json.loads(Path(path).read_text())
    if problem['kind'] not in ('banded','halfstrip'):raise ValueError('kind must be banded or halfstrip')
    p=problem.setdefault('p',s.PRIME)
    for name in ('n','width','p'):
        if type(problem[name]) is not int:raise ValueError(f'{name} must be an integer')
    from bttb_solvers._internal.banded_bttb.field import is_prime
    if not is_prime(p):raise ValueError('p must be prime')
    for name in ('lags','rhs','transpose_rhs'):
        if name not in problem:continue
        if not isinstance(problem[name],list) or any(not isinstance(row,list) for row in problem[name]):
            raise ValueError(f'{name} must be a list of rows/vectors')
        if any(type(v) is not int for row in problem[name] for v in row):
            raise ValueError('these exact-field examples require integer residues, not floating-point entries')
        problem[name]=[[v%p for v in row] for row in problem[name]]
    if not problem['rhs']:raise ValueError('supply at least one RHS vector')
    problem.setdefault('transpose_rhs',problem['rhs'])
    if any(len(v)!=problem['n']**2 for v in problem['rhs']+problem['transpose_rhs']):
        raise ValueError('each RHS must have n*n entries')
    return problem


def prepare(problem,algorithm):
    n,w,p=problem['n'],problem['width'],problem['p'];rows=problem['lags']
    if algorithm.startswith('CC') or algorithm in ('BFM','GBE'):
        if problem['kind']!='banded':raise ValueError(f'{algorithm} requires banded lag input')
    if algorithm in ('CC','CC-R'):
        return (s.prepare_cc if algorithm=='CC' else s.prepare_cc_r)(rows,n,w,p=p)
    if algorithm in ('CC-E','CC-E-R'):
        if n%p==0:raise ValueError('extension transform length must be coprime to p')
        degree=problem.get('extension_degree',1)
        if 'extension_degree' not in problem:
            while pow(p,degree,n)!=1%n:degree+=1
        F=s.extension_field(p,degree);root=s.root_of_unity(F,n)
        return (s.prepare_cc_e if algorithm=='CC-E' else s.prepare_cc_e_r)(rows,n,w,field=F,root=root)
    if algorithm=='BFM':return s.prepare_bfm(rows,n,w,p=p)
    if algorithm=='GBE':return s.prepare_gbe(rows,n,w,p=p)
    if algorithm=='Dense-LU':return s.prepare_dense_lu(rows,n,w,kind=problem['kind'],p=p)
    hs=banded_to_halfstrip(rows,n,w) if problem['kind']=='banded' else rows
    if algorithm=='HIR':return s.prepare_hir(hs,n,w,p=p)
    if algorithm=='HPR':return s.prepare_hpr(hs,n,w,p=p)
    raise ValueError('one-shot route has no prepared object')


def run(problem,algorithm):
    n,w,p=problem['n'],problem['width'],problem['p'];rows=problem['lags']
    if problem.get('determinant') and algorithm not in DETERMINANTS:
        raise ValueError(f'{algorithm} exposes solves only; choose a determinant-capable route')
    started=time.perf_counter();setup=None;det=None
    if algorithm in ('HSC','HSC-q0'):
        hs=banded_to_halfstrip(rows,n,w) if problem['kind']=='banded' else rows
        fn=s.solve_hsc if algorithm=='HSC' else s.solve_hsc_q0
        xs=[];ys=[]
        for trans,rhses,out in ((False,problem['rhs'],xs),(True,problem['transpose_rhs'],ys)):
            for b in rhses:
                result=fn(hs,n,w,b[::-1] if trans else b,p=p)
                if result['singular']:raise ValueError('physical matrix is singular: determinant zero, no nonsingular-solve service')
                x=result['solution'];out.append(x[::-1] if trans else x);det=result['determinant']
    else:
        with prepare(problem,algorithm) as q:
            setup=time.perf_counter()-started
            xs=q.solve_many(problem['rhs']);ys=q.solve_many(problem['transpose_rhs'],transpose=True)
            if problem.get('determinant'):det=q.determinant()
    elapsed=time.perf_counter()-started
    for trans,vectors,rhses in ((False,xs,problem['rhs']),(True,ys,problem['transpose_rhs'])):
        for x,b in zip(vectors,rhses):
            if action(rows,n,w,x,p=p,kind=problem['kind'],transpose=trans)!=b:
                raise ArithmeticError('exact residual failed')
    return {'algorithm':algorithm,'solutions':xs,'transpose_solutions':ys,'determinant':det,
            'setup_seconds':setup,'solver_seconds':elapsed,'exact_residual_checks':len(xs)+len(ys)}
