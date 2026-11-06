# Exact two-level Toeplitz solvers

Code accompanying **Exact banded and half-strip two-level Toeplitz solvers over
finite fields: practical routes and exponent bounds**, by Andrey Chesnokov.
The algorithm names below match the paper.

## Install

Python 3.10 or later is required. For the paper's native dense-LU baseline, use
Linux x86-64 with the pinned python-flint 0.9.0 wheel (FLINT 3.6.0).

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

No research repository, private data, host configuration, downloaded solver
snapshot or compiler is needed. FLINT is installed as a dependency; its code
and binaries are not vendored here.

## Algorithms

| Name | Construction | Outputs and requirements |
|---|---|---|
| CC | Circulant correction | Prepared forward and literal transpose solves in the paper's supplied-root HB domain. |
| CC-R | CC with retained frequency factors | Same domain and outputs; retains four-vector Toeplitz inverse factors. |
| CC-E | CC over a supplied extension field | Prime-field inputs; solutions descend to the prime subfield. |
| CC-E-R | CC-R over a supplied extension field | Retained factors over that extension. |
| BFM | Boundary-front merge | Nonsingular banded input; prepared forward and transpose solves. Classical and explicit Strassen products. |
| HSC | Half-strip classical determinant | Determinant at every rank and one nonsingular solve. Construction repeats per right-hand side. |
| HSC-q0 | HSC with guarded zero-corank precision reduction | Refuses a singular outermost block. |
| HIR | Half-strip impulse response | Requires an invertible outermost block and suitable NTT roots; determinant and prepared solves. |
| HPR | Half-strip prepared reconstruction | Allows a singular outermost block. The timed implementation retains a dense terminal factorization. |
| GBE | Guarded boundary elimination | Banded forward/transpose comparator, with no determinant service. |
| Dense-LU | Reusable native FLINT LU | One paid factorization, batched forward and transpose queries, determinant. |

`backend="flint"` selects compiled kernels; `backend="python"` selects the
reference arithmetic. Substituting compiled dense kernels does **not** implement
the paper's fast-exponent constructions. HIR determinant/transpose correspondence
and HPR's general prepared-query theorem remain unproved; their successful finite
checks are not general stability or complexity guarantees. CC-E/CC-E-R likewise
retain the paper's measured-adapter scope. There is no implementation of the
paper's general fast-exponent equality-test trees.

## Matrix convention and example

A matrix has `n` block rows and `n`-by-`n` Toeplitz blocks, so its physical order
is `n*n`. Flatten a coordinate `(t, a)` as `t*n+a`.

For a banded matrix of half-bandwidth `k`, provide `2*k+1` lag rows, each of length
`2*n-1`, ordered from outer lag `-k` to `k`. Then

```text
T[t*n+a, u*n+b] = lags[t-u+k][n-1+a-b]
```

An out-of-range outer lag means zero. For a half-strip of width `w`, provide
`n+w` rows, ordered from outer lag `-w` to `n-1`, and replace `k` by `w` in
this formula. Entries and right-hand sides are residues modulo `p` (default
`998244353`). No floating-point arithmetic is used.

```python
from bttb_solvers import prepare_cc_r
from bttb_solvers.matrices import action

n, k = 4, 1
lags = [[0] * (2*n-1) for _ in range(2*k+1)]
lags[k][n-1] = 2             # diagonal blocks: 2*identity
lags[0][n-1] = 1             # one block superdiagonal
b = list(range(n*n))
with prepare_cc_r(lags, n, k) as solver:
    x = solver.solve(b)
    y = solver.solve_transpose(b)  # literal transpose, not adjoint
    assert action(lags, n, k, x) == b
    assert action(lags, n, k, y, transpose=True) == b
```

The other prepared APIs are `prepare_cc`, `prepare_cc_e`, `prepare_cc_e_r`,
`prepare_bfm`, `prepare_hir`, `prepare_hpr`, `prepare_gbe` and
`prepare_dense_lu`. Extension routes take explicit `field=` and `root=`;
`extension_field(p, degree)` and `root_of_unity(field, n)` supply them separately.
`solve_hsc(lags, n, w, rhs)` and `solve_hsc_q0(...)` return a dictionary with
`determinant`, `solution` and `singular`. A transpose HSC solve reverses the
right-hand side before solving and reverses the result afterwards, using the
matrix's persymmetry.

Use one process at a time for solver/kernel contexts. The reproduction driver
is sequential; it uses fresh child processes to measure each observation and
release native memory afterwards.

## Reproduce the paper

The `reproduce/` directory contains the fixed problem suite, runner, recorded
paper values and table generator. It documents exactly what a rerun measures.

```bash
python reproduce/run.py --small    # exactly the three smallest listed cases
python reproduce/run.py --large    # the complete suite; can take a very long time
python reproduce/tables.py --reference  # reconstruct printed values/tables from supplied records
```

New timings depend on hardware and software; they need not equal the recorded
paper timings. Every completed solve is checked by an exact residual outside
its solver timer. The large suite is intentionally expensive and is not run
as part of the quick tests.

## Layout and license

- `bttb_solvers/api.py`: public interfaces and input conventions.
- `bttb_solvers/_internal/`: exact arithmetic, construction and query kernels.
- `tests/`: small independent dense-reference, refusal and reuse checks.
- `reproduce/`: inputs, measurements, tables and usage instructions.

Copyright (C) 2026 Andrey Chesnokov. Licensed under **AGPL-3.0-or-later**;
see [LICENSE](LICENSE). The circulant-correction construction is credited to
Chesnokov and Van Barel as in the paper. Dependencies retain their own licenses.
