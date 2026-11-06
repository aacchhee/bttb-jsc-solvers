#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare explicit solver candidates on your own supplied problem."""
import argparse,sys
from problem import load,run,ALGORITHMS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('problem')
p.add_argument('--algorithms',nargs='+',choices=ALGORITHMS,
               help='explicit candidates; defaults to the main practical candidates for this matrix class')
a=p.parse_args();problem=load(a.problem)
methods=a.algorithms or (['Dense-LU','CC','CC-R','BFM','HIR'] if problem['kind']=='banded' else ['Dense-LU','HIR','HPR'])
if problem.get('determinant') and not a.algorithms:
    methods=['Dense-LU','HIR','HPR','HSC-q0','HSC']
answers=[]
for name in methods:
    print(f'Trying {name}...',flush=True)
    try:answers.append(run(problem,name))
    except Exception as exc:print(f'  unavailable/failed: {type(exc).__name__}: {exc}',flush=True)
if not answers:sys.exit('No candidate completed; inspect the reported input/domain errors.')
expected=answers[0]
for result in answers[1:]:
    if result['solutions']!=expected['solutions'] or result['transpose_solutions']!=expected['transpose_solutions']:
        sys.exit('Cross-algorithm output disagreement')
    if problem.get('determinant') and result['determinant']!=expected['determinant']:
        sys.exit('Cross-algorithm determinant disagreement')
print('\nAlgorithm       Setup (s)       Total solver (s)       Exact checks')
for r in sorted(answers,key=lambda r:r['solver_seconds']):
    setup='repeated' if r['setup_seconds'] is None else f"{r['setup_seconds']:.6g}"
    print(f"{r['algorithm']:<15} {setup:<15} {r['solver_seconds']:<22.6g} {r['exact_residual_checks']}")
print('One local observation per candidate, including setup; use reproduce/ for the controlled paper protocol.')
