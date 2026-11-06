#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare uniform and Cantor-Kaltofen polynomial products with schoolbook."""
import argparse,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bttb_solvers._internal.halfstrip_core.exact import PrimeField
from bttb_solvers._internal.halfstrip import allfield_poly,ck_poly
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--length',type=int,default=80)
p.add_argument('--prime',type=int,default=101)
p.add_argument('--synthetic',action='store_true',
               help='disable the bounded native accelerator in the Uniform multiplier')
a=p.parse_args()
if a.length<1:p.error('length must be positive')
if a.synthetic:allfield_poly.NATIVE_ACCELERATOR_MAX=0
F=PrimeField(a.prime);rng=random.Random(1)
x=[rng.randrange(a.prime) for _ in range(a.length)];y=[rng.randrange(a.prime) for _ in range(a.length)]
reference=[0]*(2*a.length-1)
for i,u in enumerate(x):
    for j,v in enumerate(y):reference[i+j]=(reference[i+j]+u*v)%a.prime
for label,module in (('Uniform',allfield_poly),('Cantor-Kaltofen',ck_poly)):
    multiply=module.make_multiplier(F);result=multiply(x,y)
    result+= [0]*(len(reference)-len(result))
    assert result==reference
    print(label,'exact product PASS;',multiply.stats)
