#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Sequential execution of the paper's fixed cases, with exact residual checks."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bttb_solvers as solvers
from bttb_solvers.api import _kernels
from bttb_solvers.matrices import dense_matrix,action,banded_to_halfstrip
from fixtures import fixture


class Phases:
    def __init__(self):self.rows=[];self.metadata={}
    def run(self,name,fn):
        wall,cpu=time.perf_counter(),time.process_time()
        try:return fn()
        finally:self.rows.append({'phase':name,'cpu_seconds':time.process_time()-cpu,
                                  'wall_seconds':time.perf_counter()-wall})


def _columns(vectors,n,p):
    from flint import nmod_mat
    return nmod_mat(n,len(vectors),[int(vectors[j][i]) for i in range(n) for j in range(len(vectors))],p)


def _vectors(matrix):
    return [[int(matrix[i,j]) for i in range(matrix.nrows())] for j in range(matrix.ncols())]


def _control(method,rows,n,w,forward,transpose,kind,ph,backend):
    """Bounded construction controls from the exponent discussion."""
    from bttb_solvers._internal.halfstrip_core.exact import PrimeField
    from bttb_solvers._internal.halfstrip_core.dense_backend import dense_scope
    hs=banded_to_halfstrip(rows,n,w) if kind=='banded' else rows
    raw={(a,b):rows[a+w][n-1+b] for a in range(-w,w+1) for b in range(1-n,n)}
    xs,ys,det=[],[],None
    with _kernels(backend):
        if method=='Structured-halfstrip':
            from bttb_solvers._internal.full_halfstrip import solve_full
            for rhs in forward:
                result=ph.run('combined-forward',lambda rhs=rhs:solve_full(PrimeField(solvers.PRIME),hs,n,w,rhs))
                xs.append(result['physical_solution']);det=result['det']
            for rhs in transpose:
                result=ph.run('combined-transpose',lambda rhs=rhs:solve_full(PrimeField(solvers.PRIME),hs,n,w,rhs[::-1]))
                ys.append(result['physical_solution'][::-1])
        elif method.startswith('Dedicated-'):
            from bttb_solvers._internal.dedicated_banded_arithmetic import solve_dedicated_banded_many,solve_dedicated_banded
            if method=='Dedicated-classical':
                result=ph.run('combined-queries',lambda:solve_dedicated_banded_many(PrimeField(solvers.PRIME),n,w,raw,forward,transpose,dense_backend='classical'))
                xs,ys=result['solutions'],result['transpose_solutions']
            else:
                for b,c in zip(forward,transpose):
                    result=ph.run('combined-pair',lambda b=b,c=c:solve_dedicated_banded(PrimeField(solvers.PRIME),n,w,raw,b,c,dense_backend='strassen',strassen_leaf=8))
                    xs.append(result['solution']);ys.append(result['transpose_solution'])
                    ph.metadata['strassen_ledger']=result['statistics'].get('dense_scope_ledger',{})
        elif method=='Structured-CC':
            from bttb_solvers._internal.hb_physical_solver import solve_hb
            from bttb_solvers._internal.banded_bttb.field import PrimeField as CCField
            root=pow(3,(solvers.PRIME-1)//n,solvers.PRIME)
            with dense_scope('classical',1):
                for b,c in zip(forward,transpose):
                    result=ph.run('combined-pair',lambda b=b,c=c:solve_hb(CCField(solvers.PRIME,count_ops=False),n,w,raw,root,b,c))
                    xs.append(result['solution']);ys.append(result['transpose_solution'])
        else:
            from bttb_solvers._internal.banded_bttb.field import PrimeField as CCField
            from bttb_solvers._internal.halfstrip.counter import HalfStripCounter
            from bttb_solvers._internal.halfstrip.symbol import HalfStripSymbol
            from bttb_solvers._internal.halfstrip.allfield_solver import AllFieldHalfstripSolver
            modes={'Uniform-halfstrip':('allfield','allfield'),'Uniform-halfstrip-CK-terminal':('allfield','ck'),
                   'CK-halfstrip-uniform-terminal':('ck','allfield'),'CK-halfstrip':('ck','ck')}
            fm,tm=modes[method]
            counter=HalfStripCounter(count_ops=False)
            q=AllFieldHalfstripSolver(HalfStripSymbol(CCField(solvers.PRIME,counter,count_ops=False),n,w,
                       {d-w:tuple(row) for d,row in enumerate(hs)}),counter,frontend_multiplier=fm,terminal_multiplier=tm)
            ph.run('prepare',q.prepare);det=ph.run('determinant',q.det)
            xs=[ph.run('forward',lambda b=b:q.solve(b)) for b in forward]
            ys=[ph.run('transpose',lambda b=b:q.solve(b[::-1]))[::-1] for b in transpose]
    return xs,ys,det


def observe(case,method):
    import flint
    flint.ctx.threads=1
    from bttb_solvers._internal.halfstrip import allfield_poly
    allfield_poly.NATIVE_ACCELERATOR_MAX=method['native_product_cap']
    rows,forward,transpose,kind=fixture(case)
    n,w=case['n'],case['width'];algorithm=method['algorithm'];backend=method['backend']
    ph=Phases();det=None;xs=[];ys=[];metadata={}
    encoded=json.dumps([rows,forward,transpose],separators=(',',':')).encode()
    input_digest=hashlib.sha256(encoded).hexdigest()
    if algorithm=='Dense':
        A=ph.run('prepare',lambda:dense_matrix(rows,n,w,kind=kind))
        xs=ph.run('forward-batch',lambda:_vectors(A.solve(_columns(forward,n*n,solvers.PRIME))))
        At=ph.run('transpose-prepare',A.transpose)
        del A
        ys=ph.run('transpose-batch',lambda:_vectors(At.solve(_columns(transpose,n*n,solvers.PRIME))))
        det=ph.run('determinant',lambda:int(At.det()))
    elif algorithm in ('HSC','HSC-q0'):
        hs=banded_to_halfstrip(rows,n,w) if kind=='banded' else rows
        fn=solvers.solve_hsc_q0 if algorithm=='HSC-q0' else solvers.solve_hsc
        for b in forward:
            result=ph.run('combined-forward',lambda b=b:fn(hs,n,w,b,backend=backend))
            xs.append(result['solution']);det=result['determinant']
        for c in transpose:
            result=ph.run('combined-transpose',lambda c=c:fn(hs,n,w,c[::-1],backend=backend))
            ys.append(result['solution'][::-1])
    elif algorithm.startswith('CC-E'):
        # Supply field, root and embeddings before starting solver timers.
        degree=1
        while pow(solvers.PRIME,degree,n)!=1:degree+=1
        F=solvers.extension_field(solvers.PRIME,degree)
        root=solvers.root_of_unity(F,n)
        raw={(a,b):F.ctx(rows[a+w][n-1+b]) for a in range(-w,w+1) for b in range(1-n,n)}
        bs=[[F.ctx(v) for v in b] for b in forward];cs=[[F.ctx(v) for v in c] for c in transpose]
        from bttb_solvers._internal.cc_prepared import prepare_cc
        from bttb_solvers._internal.halfstrip_core.dense_backend import dense_scope
        with _kernels('flint',circulant=True,extension=True),dense_scope('classical',1,polynomial_entry_policy='off'):
            q=ph.run('prepare',lambda:prepare_cc(F,n,w,raw,root,arm='retained' if algorithm=='CC-E-R' else 'original'))
            extxs=[ph.run('forward',lambda b=b:q.query(b)) for b in bs]
            extys=[ph.run('transpose',lambda c=c:q.query(c,transpose=True)) for c in cs]
        def descend(vector):
            out=[]
            for v in vector:
                coefficients=[int(x) for x in v.polynomial()]
                if any(coefficients[1:]):raise ArithmeticError('non-base-field solution coordinate')
                out.append(coefficients[0] if coefficients else 0)
            return out
        xs=[descend(v) for v in extxs];ys=[descend(v) for v in extys]
        metadata['extension_degree']=degree
    elif algorithm in ('CC','CC-R','BFM','BFM-Strassen','GBE','HIR','HPR','HPR-division','HPR-appended','Dense-LU'):
        if algorithm in ('CC','CC-R'):
            root=pow(3,(solvers.PRIME-1)//n,solvers.PRIME)
            fn=solvers.prepare_cc if algorithm=='CC' else solvers.prepare_cc_r
            setup=lambda:fn(rows,n,w,root=root,backend=backend)
        elif algorithm.startswith('BFM'):
            setup=lambda:solvers.prepare_bfm(rows,n,w,backend=backend,product='strassen' if algorithm=='BFM-Strassen' else 'classical')
        elif algorithm=='GBE':setup=lambda:solvers.prepare_gbe(rows,n,w,backend=backend)
        elif algorithm=='HIR':
            hs=banded_to_halfstrip(rows,n,w) if kind=='banded' else rows
            setup=lambda:solvers.prepare_hir(hs,n,w,backend=backend)
        elif algorithm.startswith('HPR'):
            hs=banded_to_halfstrip(rows,n,w) if kind=='banded' else rows
            mode=algorithm.split('-',1)[1] if '-' in algorithm else 'lu'
            setup=lambda:solvers.prepare_hpr(hs,n,w,backend=backend,terminal_query=mode)
        else:
            from bttb_solvers._internal.dense_lu import get_api
            get_api()  # ABI/library checks are outside the public service timers.
            setup=lambda:solvers.prepare_dense_lu(rows,n,w,kind=kind)
        with ph.run('prepare',setup) as q:
            xs=ph.run('forward-batch',lambda:q.solve_many(forward))
            ys=ph.run('transpose-batch',lambda:q.solve_many(transpose,transpose=True))
            if algorithm in ('HIR','HPR','HPR-division','HPR-appended','Dense-LU'):
                det=ph.run('determinant',q.determinant)
            if algorithm=='Dense-LU':metadata['factorization_calls']=q.state.factorization_calls
            if algorithm=='BFM-Strassen':metadata['strassen_ledger']=q.state.statistics()['strassen_ledger']
    else:xs,ys,det=_control(algorithm,rows,n,w,forward,transpose,kind,ph,backend)
    if len(xs)!=len(forward) or len(ys)!=len(transpose):raise ArithmeticError('missing solve results')
    for trans,vectors,rhses in ((False,xs,forward),(True,ys,transpose)):
        for x,b in zip(vectors,rhses):
            if action(rows,n,w,x,kind=kind,transpose=trans)!=b:
                raise ArithmeticError('exact residual failed')
    metadata.update(ph.metadata)
    phases=ph.rows
    total=sum(p['cpu_seconds'] for p in phases)
    det_cpu=sum(p['cpu_seconds'] for p in phases if p['phase']=='determinant')
    with_determinant=algorithm in ('HSC','HSC-q0','HIR','HPR','HPR-division','HPR-appended','Structured-halfstrip') or algorithm.startswith(('Uniform-','CK-'))
    memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'status':'ok','input_sha256':input_digest,
            'output_sha256':hashlib.sha256(json.dumps([xs,ys],separators=(',',':')).encode()).hexdigest(),
            'determinant':None if det is None else str(det),'service':'det+solve' if with_determinant else 'solve-only',
            'cpu_seconds':total if with_determinant else total-det_cpu,
            'cpu_solve_seconds':total-det_cpu,'cpu_det_solve_seconds':total,
            'rss_gib':memory*(1 if sys.platform=='darwin' else 1024)/2**30,
            'exact_residual_checks':len(xs)+len(ys),'phases':phases,'metadata':metadata}


def summarize(records,destination):
    groups={}
    by_case={}
    for r in records:
        if r['status']=='ok':by_case.setdefault(r['case'],[]).append(r)
    for label,rs in by_case.items():
        if len({r['input_sha256'] for r in rs})!=1 or len({r['output_sha256'] for r in rs})!=1:
            raise ArithmeticError(f'cross-algorithm input/output mismatch: {label}')
        if len({r['determinant'] for r in rs if r.get('determinant') is not None})>1:
            raise ArithmeticError(f'cross-algorithm determinant mismatch: {label}')
    for r in records:
        key=(r['case'],r['algorithm'],r['backend'],r['native_product_cap'])
        groups.setdefault(key,[]).append(r)
    medians={}
    for key,rows in groups.items():
        ok=[x for x in rows if x['status']=='ok']
        if len(ok)!=2:continue
        hashes={x['input_sha256'] for x in ok};outputs={x['output_sha256'] for x in ok}
        if len(hashes)!=1 or len(outputs)!=1:raise ArithmeticError('mirrored visit input/output disagreement')
        medians[key]={f:statistics.median(x[f] for x in ok) for f in ('cpu_seconds','cpu_solve_seconds','cpu_det_solve_seconds','rss_gib')}
    fields=['case','n','width','pairs','algorithm','backend','native_product_cap','visits','status','service','cpu_seconds','rss_gib','speedup_dense_lu']
    with (destination/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key,rows in groups.items():
            base=rows[0];out={k:base[k] for k in fields if k in base};out['visits']=sum(r['status']=='ok' for r in rows)
            out['status']='ok' if key in medians else 'incomplete';m=medians.get(key,{})
            out.update({k:m.get(k,'') for k in ('cpu_seconds','rss_gib')})
            dense=medians.get((key[0],'Dense-LU','flint',4096))
            if dense and m:
                service=base.get('service','solve-only');ref=dense['cpu_det_solve_seconds' if service=='det+solve' else 'cpu_solve_seconds']
                out['speedup_dense_lu']=ref/m['cpu_seconds']
            writer.writerow(out)
    derived=[]
    for pairs in (1,4):
        points=[]
        for key,value in medians.items():
            row=groups[key][0]
            if row['family']=='banded' and row['n']==128 and row['pairs']==pairs and key[1:]==('CC-R','flint',4096):
                reference=medians.get((key[0],'Dense-LU','flint',4096))
                if reference:points.append((row['width'],value['cpu_seconds']-reference['cpu_solve_seconds']))
        for (a,da),(b,db) in zip(sorted(points),sorted(points)[1:]):
            if da<=0<db:
                derived.append({'calculation':'bandwidth_linear_crossover','pairs':pairs,'value':a+(b-a)*(-da)/(db-da),'low':a,'high':b})
                break
    with (destination/'derived.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['calculation','pairs','value','low','high']);writer.writeheader();writer.writerows(derived)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--small',action='store_true',help='exactly the three smallest cases')
    mode.add_argument('--large',action='store_true',help='all cases, no wall-time limit')
    parser.add_argument('--output',type=Path,help='output directory (existing measurements are replaced)')
    parser.add_argument('--list',action='store_true',help='show cases without running them')
    args=parser.parse_args()
    cases=json.loads((Path(__file__).parent/'cases.json').read_text())
    if args.small:cases=cases[:3]
    total=sum(2*len(c['methods']) for c in cases)
    print(f'{len(cases)} cases; {total} observations; sequential execution',flush=True)
    if args.list:
        for c in cases:print(c['case'],len(c['methods']),'methods')
        return 0
    destination=(args.output or Path(__file__).parent/'output'/('small' if args.small else 'large')).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    import flint
    (destination/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
        'python_flint':flint.__version__,'flint':str(flint.__FLINT_VERSION__),'cases':len(cases),'observations':total,
        'threads':1,'parallel':False,'time_limit':None},indent=2)+'\n')
    records=[];number=0
    with (destination/'measurements.jsonl').open('w') as output:
        for case in cases:
            methods=case['methods']+list(reversed(case['methods']))
            for visit,method in enumerate(methods):
                number+=1
                label={k:v for k,v in case.items() if k!='methods'};label.update(method,visit=visit)
                print(f"[{number}/{total}] {case['case']} {method['algorithm']}/{method['backend']} cap={method['native_product_cap']}",flush=True)
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],input=json.dumps([case,method]),
                                      text=True,capture_output=True,env=os.environ.copy())
                try:r=json.loads(result.stdout)
                except (ValueError,TypeError):r={'status':'error','error':result.stderr[-4000:] or 'worker ended without a result'}
                r.update(label);records.append(r);output.write(json.dumps(r,sort_keys=True)+'\n');output.flush()
                if r['status']!='ok':print(r.get('error',r['status']),flush=True)
    summarize(records,destination)
    from tables import fresh_summary
    fresh_summary(destination,destination)
    failures=sum(r['status']!='ok' for r in records)
    print(f'Wrote {destination}; {len(records)-failures} successful, {failures} failures/refusals.',flush=True)
    return int(bool(failures))


if __name__=='__main__':
    if sys.argv[1:]==['--worker']:
        try:
            case,method=json.loads(sys.stdin.read());print(json.dumps(observe(case,method)))
        except Exception as exc:
            print(json.dumps({'status':'error','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}))
            sys.exit(1)
    else:sys.exit(main())
