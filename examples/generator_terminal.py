#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Solve a supplied Stein-generator system, separately from physical acquisition."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bttb_solvers._internal.halfstrip_core.exact import PrimeField
from bttb_solvers._internal.kn_services import Services
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--size',type=int,default=6)
a=p.parse_args()
if a.size<2:p.error('size must be at least two')
d=a.size;F=PrimeField(101)
# M = 2 I + Z. Its Stein displacement M-Z M Z^T has one generator column.
g=[[2],[1]]+[[0] for _ in range(d-2)];h=[[1]]+[[0] for _ in range(d-1)]
b=list(range(1,d+1));answer=Services(F).solve_stein(g,h,b)
assert answer['status']=='CONSISTENT'
x=answer['solution']
assert [(2*x[i]+(x[i-1] if i else 0))%101 for i in range(d)]==[v%101 for v in b]
print('Supplied-generator terminal: exact check passed; solution =',x)
print('This example supplies g,h directly. Dedicated physical routes must acquire their generators from raw lags first.')
