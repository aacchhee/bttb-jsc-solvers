#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Render extended recorded tables, or display the summary of a fresh run."""
import argparse
import csv
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def reference_tables(destination):
    records=list(csv.DictReader((HERE/'reference/paper_numbers.csv').open()))
    values={r['key']:r['value'] for r in records}
    if len(values)!=len(records):raise ValueError('duplicate paper number')
    destination.mkdir(parents=True,exist_ok=True)
    fragments=[]
    for name in json.loads((HERE/'reference/tables.json').read_text()):
        template=(HERE/'reference'/name).read_text()
        text=re.sub(r'\\papernum\{([^}]+)\}',lambda m:values[m[1]],template)
        if '\\papernum{' in text:raise ValueError('unfilled table value')
        (destination/name).write_text(text);fragments.append(text)
    header=r'''\documentclass{article}
\usepackage[a4paper,margin=20mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx}
\newcommand{\F}{\mathbb F}
\newcommand{\softO}{\widetilde O}
\begin{document}
'''
    (destination/'recorded_tables.tex').write_text(header+'\n\\clearpage\n'.join(fragments)+'\n\\clearpage\n\\end{document}\n')
    with (destination/'paper_numbers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['key','value','unit','kind']);w.writeheader();w.writerows(records)
    print(f'Rendered {len(fragments)} extended tables and {len(values)} recorded keys in {destination}.')
    print('These are the recorded paper values, not new timing measurements.')
    return values


def fresh_summary(source,destination):
    rows=list(csv.DictReader((source/'summary.csv').open()))
    destination.mkdir(parents=True,exist_ok=True)
    lines=['# Fresh measurements','',
           '| Case | Algorithm | Backend | Cap | CPU (s) | RSS (GiB) | Dense-LU speedup | Status |',
           '|---|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        vals=[row[k] for k in ('case','algorithm','backend','native_product_cap','cpu_seconds','rss_gib','speedup_dense_lu','status')]
        for i in (4,5,6):
            if vals[i]:vals[i]=f'{float(vals[i]):.5g}'
        lines.append('| '+' | '.join(vals)+' |')
    (destination/'measurements.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {destination/"measurements.md"} from {source/"summary.csv"}.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference',action='store_true',help='reconstruct the recorded paper tables and all printed values')
    mode.add_argument('--results',type=Path,help='directory produced by run.py')
    p.add_argument('--output',type=Path,default=HERE/'output/tables')
    a=p.parse_args()
    if a.reference:reference_tables(a.output)
    else:fresh_summary(a.results,a.output)


if __name__=='__main__':main()
