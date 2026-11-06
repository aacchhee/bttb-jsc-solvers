#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect actual recursion in the two explicit Strassen comparisons."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bttb_solvers as s
from bttb_solvers.matrices import action
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--route',choices=('BFM','Dedicated'),default='BFM')
p.add_argument('--n',type=int,default=16)
p.add_argument('--width',type=int,default=2)
p.add_argument('--leaf',type=int,choices=(1,2,4,8,16))
a=p.parse_args();n,k=a.n,a.width;rng=random.Random(17)
rows=[[rng.randrange(s.PRIME) for _ in range(2*n-1)] for _ in range(2*k+1)]
rows[0][:n-1]=[0]*(n-1);rows[0][n-1]=1;rows[0][n]=1
b=list(range(1,n*n+1))
if a.route=='BFM':
    with s.prepare_bfm(rows,n,k,product='strassen',strassen_leaf=a.leaf or 16) as q:
        x=q.solve(b);y=q.solve_transpose(b);ledger=q.state.statistics()['strassen_ledger']
else:
    r=s.solve_dedicated(rows,n,k,b,b,product='strassen',strassen_leaf=a.leaf or 8)
    x,y=r['solution'],r['transpose_solution'];ledger=r['statistics']['dense_scope_ledger']
assert action(rows,n,k,x)==b
assert action(rows,n,k,y,transpose=True)==b
print('Exact forward/transpose checks passed.')
print(json.dumps(ledger,indent=2))
print('Small dimensions can stay in classical leaves; inspect recursive-node counts when varying n, width and leaf.')
