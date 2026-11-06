#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Solve a supplied exact-field problem with a named main interface."""
import argparse,json,sys
from problem import load,run,ALGORITHMS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('problem',help='JSON problem file; see examples/problems/')
p.add_argument('--algorithm',choices=ALGORITHMS,required=True)
a=p.parse_args()
try:print(json.dumps(run(load(a.problem),a.algorithm),indent=2))
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(1)
