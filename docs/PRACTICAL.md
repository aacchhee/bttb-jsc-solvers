# Solving your own problem

This guide is for choosing a solver, supplying data and measuring the service
you actually need. The package works over exact finite fields. Its inputs are
integer residues, with no floating-point tolerance or numerical pivoting model.
The detailed [solver descriptions](SOLVERS.md) state each method's domain.

## Choose candidates, then measure

There is no hardware-independent fastest method. Physical order is `N=n*n`,
not `n`; bandwidth, field roots, number of right-hand sides and memory all
change the comparison.

| Your problem | Useful first candidates | What decides the choice |
|---|---|---|
| Narrow banded, suitable roots in the base field | CC, CC-R, Dense-LU | CC-R retains more state for repeated queries. Include its setup cost. |
| Banded, base-field CC root/domain unavailable | BFM, Dense-LU; HIR if H0 and its transform requirements hold | BFM does not need transform roots. An extension-field CC comparison is also available, with field acquisition paid. |
| Broad banded or modest physical order | Dense-LU alongside a structured candidate | Dense compiled arithmetic can win despite its storage and asymptotic costs. |
| Half-strip, invertible outermost block H0 | HIR, Dense-LU | Include impulse construction and terminal factorization, then the actual RHS batch. |
| Half-strip, singular outermost block | Dense-LU, HPR | Dense-LU is the CPU baseline; HPR is useful for studying the measured storage tradeoff. Its general prepared-query theorem remains open. |
| Half-strip determinant, including singular physical matrices | HSC | Returns determinant zero at singularity. HSC-q0 requires H0; HIR/HPR provide determinant services in their documented scopes. |

The paper's finite bandwidth crossover is evidence for its tested inputs and
machine. It is not a rule that a particular width always chooses a solver.
GBE and the explicit Strassen routes are available for comparisons; their
measured outcomes and other trials are in [REJECTED_ROUTES](../REJECTED_ROUTES.md).

## Supply the matrix without expanding it

Flatten `(block_row, inner_row)` as `t*n+a`. Each lag row has `2*n-1` entries,
ordered by inner lag `-(n-1), ..., n-1`.

- Banded width `k`: supply `2*k+1` rows, outer lags `-k, ..., k`.
- Half-strip width `w`: supply `n+w` rows, outer lags `-w, ..., n-1`.

For either class, with `width=k` or `w`,

```text
T[t*n+a, u*n+b] = lags[t-u+width][n-1+a-b]
```

An outer lag outside the supplied range is zero. The transpose is the literal
matrix transpose over the same field. A singular outermost block is different
from a singular physical matrix: the former can still give a solvable system.

The [JSON examples](../examples/README.md#input-format) accept one or more
forward and transpose RHS vectors. Copy a file from `examples/problems/`,
replace its lag coefficients and RHS vectors, then run:

```bash
python examples/solve.py my_problem.json --algorithm CC-R
python examples/compare.py my_problem.json --algorithms Dense-LU CC CC-R BFM
```

`compare.py` reports each candidate's domain errors, checks every successful
answer by the raw matrix action, and compares the answers between methods.
It measures one local observation per candidate, including setup and both
RHS batches. For repeatable paper measurements use [reproduce/](../reproduce/README.md).

## Reuse preparation in Python

```python
from bttb_solvers import prepare_cc_r
from bttb_solvers.matrices import action

n, k = 4, 1
lags = [[0] * (2*n-1) for _ in range(2*k+1)]
lags[k][n-1] = 2                 # diagonal blocks: 2 I
lags[0][n-1] = 1                 # first block superdiagonal: I
rhs = [list(range(n*n)), [1] * (n*n)]
with prepare_cc_r(lags, n, k) as solver:
    solutions = solver.solve_many(rhs)
    transposed = solver.solve_many(rhs, transpose=True)
    for b, x in zip(rhs, solutions):
        assert action(lags, n, k, x) == b
```

Prepare once for a fixed matrix and reuse that object for additional RHS
vectors. Closing the object releases retained native state. The prepared
interfaces support `solve`, `solve_transpose` and `solve_many`; determinant
availability differs by method. HSC and HSC-q0 are one-shot calls and repeat
construction for every RHS. See [SOLVERS](SOLVERS.md) for their return values.
Use the kernel contexts serially within a process, since backend rebinding is
process-global.

`banded_to_halfstrip(lags, n, k)` in `bttb_solvers.matrices` supplies the zero
outer-lag rows needed to try a half-strip method on banded input. The
comparison example performs this conversion automatically.

## Interpret time, storage and failures

Measure the workload you will run: one solve, many forward RHS vectors, a
forward/transpose pair, or a determinant plus solves. Include preparation,
field/root acquisition and required conversions in an end-to-end comparison.
The examples' independent residual checking is extra verification work outside
the solver timer. Internal solver guards, including BFM/GBE residual guards,
remain inside it. Import and first native-library load can affect a short
single-process example; the paper driver has explicit measurement boundaries.

`backend="flint"` is the usual compiled implementation; `backend="python"`
selects the reference arithmetic where exposed. They need not use identical
arithmetic primitives. Dense-LU's native binding requires the documented Linux
x86-64, python-flint 0.9.0 / FLINT 3.6.0 combination. Refusals are reported rather
than silently switching algorithms. A failed domain check may suggest another
method; a failed exact residual is an implementation error to investigate.

If you need to change the implementation, continue with the
[code and representation guide](CODE_GUIDE.md). For neighbouring constructions,
read the [trial map](../REJECTED_ROUTES.md) and its linked references.
