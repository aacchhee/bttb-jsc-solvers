# Exact two-level Toeplitz solvers

Code accompanying **Exact banded and half-strip two-level Toeplitz solvers over
finite fields: practical routes and exponent bounds**, by Andrey Chesnokov.
The algorithm names match the paper. Inputs are exact finite-field lag
coefficients; the physical matrix order is `N=n*n`.

## Start with your goal

**I have a banded or half-strip problem and want to solve it quickly.**
Start with the [practical guide](docs/PRACTICAL.md): choose candidates for your
matrix, supply lag coefficients, reuse setup and compare the service you need.
The [runnable examples](examples/README.md) accept your own JSON input and check
every answer exactly. Include Dense-LU in the comparison; the fastest choice
depends on dimensions, bandwidth, roots, RHS reuse and hardware.

**I want to study this method and neighbouring approaches.**
Read the [solver guide](docs/SOLVERS.md) for each construction's setup, retained
state, queries and restrictions. The [code guide](docs/CODE_GUIDE.md) maps the
matrix representations, basic operations and source files. A separate
[file reference](docs/FILE_REFERENCE.md) lists every internal module.
[Trials and outcomes](REJECTED_ROUTES.md) separates alternative constructions
and nondefault backends from the main solver interfaces, with
[paper references](REFERENCES.md) explaining CK, KN and related kernels.
The examples include polynomial, supplied-generator and actual Strassen routes.

## Install and try

Python 3.10 or later is required. The native dense-LU baseline requires Linux
x86-64 and the pinned python-flint 0.9.0 wheel (FLINT 3.6.0).

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python examples/compare.py examples/problems/banded.json
python -m unittest discover -s tests -v
```

FLINT is installed as a dependency; its sources and binaries are not vendored.
The package runs independently of the original research workspace.

## Main interfaces

| Paper name | Public entry point | Useful for / key restriction |
|---|---|---|
| CC, CC-R | `prepare_cc`, `prepare_cc_r` | Banded prepared solves in the supplied-root domain; CC-R retains inverse factors |
| CC-E, CC-E-R | `prepare_cc_e`, `prepare_cc_e_r` | Extension-field CC adapters, with checked descent to the prime subfield |
| BFM | `prepare_bfm` | Root-free boundary-front prepared solves; classical or explicit Strassen products |
| GBE | `prepare_gbe` | Root-free guarded boundary comparator |
| HIR | `prepare_hir` | Half-strip prepared solves and determinant; requires invertible outermost block and suitable roots |
| HSC | `solve_hsc` | Half-strip all-rank determinant and nonsingular one-shot solve |
| HSC-q0 | `solve_hsc_q0` | Reduced-precision HSC, guarded by invertibility of the outermost block |
| HPR | `prepare_hpr` | Half-strip retained reconstruction, including singular outermost blocks; measured dense-terminal implementation |
| Dense-LU | `prepare_dense_lu` | One reusable native physical LU, batched solves and determinant |

Prepared objects expose `solve`, `solve_transpose`, `solve_many`, and
`determinant` where documented. Use a context manager to release retained state.
Use kernel contexts serially within one process. See the
[input convention and Python example](docs/PRACTICAL.md#supply-the-matrix-without-expanding-it).

`backend="flint"` selects compiled kernels where supported. These replacements
do not by themselves implement fast-exponent constructions. HIR's general
determinant/transpose correspondence and HPR's general prepared-query theorem
remain unproved; extension adapters retain their measured scope. The general
fast-exponent equality-test trees are not implemented. The
[solver guide](docs/SOLVERS.md) distinguishes executable behavior from guarantees.

BFM-Strassen is `prepare_bfm(..., product="strassen", strassen_leaf=16)`.
Dedicated-Strassen is `solve_dedicated(..., product="strassen", strassen_leaf=8)`.
Both contain explicit recursion, with inspectable ledgers; see
[examples](examples/README.md#explore-neighbouring-approaches).

## Regenerate measurements and tables

```bash
python reproduce/run.py --small       # exactly the three smallest cases
python reproduce/run.py --large       # all cases; intentionally expensive
python reproduce/tables.py --reference
```

The [reproduction guide](reproduce/README.md) explains the fixed inputs,
sequential fresh-process runner, exact checks and generated outputs. Fresh
measurements depend on hardware and software. `--reference` renders the
supplied recorded paper values; it does not rerun those timings.
The [validation record](VALIDATION.md) describes completed checks and scope.

## Repository map

| Location | Contents |
|---|---|
| [bttb_solvers/api.py](bttb_solvers/api.py) | Public preparation and one-shot interfaces |
| [bttb_solvers/matrices.py](bttb_solvers/matrices.py) | Raw matrix action, small dense representations and banded-to-half-strip conversion |
| [bttb_solvers/_internal/](bttb_solvers/_internal/README.md) | Algorithm setup/query code and shared arithmetic; module roles documented in the code guide |
| [examples/](examples/README.md) | User-input solving/comparison and research-kernel examples |
| [reproduce/](reproduce/README.md) | Paper grids, measurements, recorded values and table generation |
| [tests/](tests/test_solvers.py) | Independent dense-reference, domain, transpose and reuse checks |

Copyright (C) 2026 Andrey Chesnokov. Licensed under **AGPL-3.0-or-later**;
see [LICENSE](LICENSE). Dependencies retain their own licenses.
Construction credits and implementation scope are in [REFERENCES](REFERENCES.md).
Software citation metadata is supplied in [CITATION.cff](CITATION.cff).
