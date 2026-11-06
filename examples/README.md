# Runnable examples

Run these commands from the repository root after `python -m pip install -e .`.
The first three scripts address practical use; the remaining examples expose
research kernels. None is an automatic universal solver selector.

| File | Purpose | Interface level |
|---|---|---|
| [solve.py](solve.py) | Solve a JSON input with one named method; print answers, optional determinant and timing | Public API |
| [compare.py](compare.py) | Try selected methods on the same input; report failures and compare checked outputs | Public API |
| [problem.py](problem.py) | Shared JSON loading, method dispatch, timing and independent residual checking | Example support, not a second solver implementation |
| [strassen.py](strassen.py) | Run BFM-Strassen or Dedicated-Strassen and display actual recursion ledgers | Public comparison API, internal statistics |
| [polynomial_backends.py](polynomial_backends.py) | Compare Uniform and Cantor–Kaltofen products against schoolbook multiplication | Internal research kernels |
| [generator_terminal.py](generator_terminal.py) | Supply explicit Stein generators and solve a small structured terminal | Internal research kernel; not generator acquisition from a physical problem |
| [problems/](problems/) | Small banded, H0 half-strip and singular-leading-block half-strip inputs | Editable JSON data |

## Solve and compare

```bash
python examples/solve.py examples/problems/banded.json --algorithm CC-R
python examples/compare.py examples/problems/banded.json
python examples/compare.py examples/problems/halfstrip.json --algorithms Dense-LU HIR HPR
python examples/solve.py examples/problems/halfstrip_deficient.json --algorithm HPR
```

Every returned forward and transpose solution is checked using the raw lag
matrix action, outside the solver timer. `compare.py` also requires agreement
between completed methods. Its default candidate list depends on the matrix
class; request `--algorithms` to control it. Domain failures are printed and
other candidates continue. The reported total solver time includes setup.
HSC/HSC-q0 repeat the whole construction per RHS rather than retaining state.

The deficient example has a singular outermost block and a nonsingular full
matrix. HPR and HSC handle it; HIR and HSC-q0 refuse its H0 violation.
For determinant zero on a singular full matrix, use `solve_hsc` directly: these
CLI examples request nonsingular solutions and report singularity as an error.

## Input format

Each JSON object contains:

| Key | Meaning |
|---|---|
| `kind` | `"banded"` or `"halfstrip"` |
| `n` | Block count and inner block order; RHS length is `n*n` |
| `width` | Outer half-bandwidth `k`, or half-strip upper width `w` |
| `p` | Prime modulus; default `998244353` |
| `lags` | Integer coefficient rows, with the ordering in [PRACTICAL](../docs/PRACTICAL.md#supply-the-matrix-without-expanding-it) |
| `rhs` | Nonempty list of forward RHS vectors, even for one vector |
| `transpose_rhs` | List of transpose RHS vectors; omitted means use `rhs` again |
| `determinant` | Optional boolean; request only determinant-capable methods |
| `extension_degree` | Optional explicit degree for CC-E/CC-E-R; otherwise search the smallest degree admitting an `n`th root |

Integer entries are reduced modulo `p`. Floating-point values are rejected.
The full numerical files in `problems/` are a more convenient starting point
than a dense `n*n` by `n*n` matrix. Extension construction can be expensive;
the comparison's total timer includes it. A supplied degree may still fail
the CC field-size/domain guards.

## Explore neighbouring approaches

```bash
python examples/strassen.py --route BFM --n 16 --width 2
python examples/strassen.py --route Dedicated --n 8 --width 1
python examples/polynomial_backends.py --length 80 --prime 101
python examples/polynomial_backends.py --length 80 --prime 2
python examples/polynomial_backends.py --length 80 --prime 101 --synthetic
python examples/generator_terminal.py --size 6
```

The Strassen examples verify both solve directions and print recursion counts.
Small dimensions can stay entirely in classical leaves; vary `--leaf`, `--n`
and `--width` to examine the actual recursion. The dedicated example acquires
its generators from the input inside the measured construction.

The polynomial example exercises different arbitrary-field multiplication
paths, including characteristic two. `--synthetic` disables Uniform's bounded
native accelerator so its synthetic transform recursion is visible in the
statistics; without it, small products can use FLINT directly.
These are arithmetic controls, not new
physical solvers. CK abbreviates **Cantor–Kaltofen**. The generator example
uses the supplied-generator services labelled KN for **Khichane–Neiger**;
its `g,h` are already supplied and do not demonstrate the cost of acquiring
them from raw lags. Read the [source references](../REFERENCES.md) and
[trial scope](../REJECTED_ROUTES.md) before interpreting these experiments.

For all source files and their roles, see [CODE_GUIDE](../docs/CODE_GUIDE.md).
For fixed paper inputs and the serial measurement protocol, see
[reproduce/](../reproduce/README.md).
