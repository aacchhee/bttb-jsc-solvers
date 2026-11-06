# Solving your own problem

Suppose you have a two-level Toeplitz matrix T and want to solve Tx=b,
perhaps for several vectors b. This page explains how to supply T, try a
method, and decide which method is worth using for your problem.

All calculations here are exact, over a finite field. In the usual case the
entries are integers modulo a prime p. The examples use p=998244353. This
software does not solve systems over the real or complex numbers.

## Which method should I try?

If your matrix is banded, start by comparing **CC-R**, **BFM**, and **Dense-LU**.
CC-R needs suitable roots of unity in the field; BFM does not. Dense-LU is
ordinary LU factorization of the full matrix, using a fast exact-arithmetic
library. It is often competitive for smaller problems or wider bands.

For a half-strip matrix, the first question is whether its outermost permitted
block A[-w] is invertible. This is the paper's assumption **H0**.

| Your half-strip problem | Start with |
|---|---|
| A[-w] is invertible | HIR and Dense-LU |
| A[-w] is singular, but T is nonsingular | HPR and Dense-LU |
| You need det(T), including when T is singular | HSC |

HPR is worth considering when memory matters. It did not beat reusable dense
LU in CPU time in the paper's comparisons, and its general theorem for repeated
solves remains open. A singular A[-w] does **not** imply that T is singular.

The [solver guide](SOLVERS.md) explains these methods and their assumptions.
There is no single fastest choice for all sizes, bandwidths, fields, and
numbers of right-hand sides. The comparison below lets you try your own case.

## Try a supplied example

After following the [installation instructions](../README.md#install-and-try),
run these commands from the repository's main directory:

```bash
python examples/solve.py examples/problems/banded.json --algorithm CC-R
python examples/compare.py examples/problems/banded.json
```

The first command solves one example and prints the answer. The second solves
the same problem by several methods and lists their running times. Every answer
is checked by multiplying it by the original matrix. If a method's assumptions
are not satisfied, the comparison explains the failure and tries the others.

For half-strips, try:

```bash
python examples/compare.py examples/problems/halfstrip.json --algorithms Dense-LU HIR HPR
```

To use your own data, copy one of the files in `examples/problems/`, replace its
entries, and give the new filename to the same commands. These are ordinary
text files in JSON format; the [example instructions](../examples/README.md#input-format)
explain their fields.

## Supply the matrix without expanding it

Here n is both the number of block rows and the order of each block. Thus T
has n² rows and columns. You supply the Toeplitz coefficients rather than all
n⁴ matrix entries.

Write T's blocks as T[t,u]=A[t-u]. Each A[d] is itself Toeplitz. Store it as a
row of 2n−1 coefficients, in order from diagonal index −(n−1) to n−1. The
middle coefficient is on the main diagonal. Our sign convention is
A[d][a,b]=a[d,a−b], so negative indices describe upper diagonals.

For a banded matrix, supply the coefficient rows for A[-k], …, A[k], in that
order. Blocks farther from the main block diagonal are zero. For a half-strip,
supply A[-w], …, A[n−1]. There are at most w block superdiagonals, while all
block subdiagonals are allowed.

For example, this banded block matrix has width k=1:

```text
       [ 2I   I   0   0 ]
T  =   [  0  2I   I   0 ]
       [  0   0  2I   I ]
       [  0   0   0  2I ]
```

Take n=4, so I is the 4-by-4 identity and T is 16-by-16. Its three coefficient
rows are:

```text
A[-1]:  [0, 0, 0, 1, 0, 0, 0]
A[ 0]:  [0, 0, 0, 2, 0, 0, 0]
A[ 1]:  [0, 0, 0, 0, 0, 0, 0]
```

These three rows form `lags` in the input file or Python call. A vector b is
listed block by block: first its n entries for block row 0, then those for
block row 1, and so on. Each vector has n² entries. Entries are reduced modulo p.

## Solve several systems with the same matrix

Most methods first prepare data depending only on T, much as LU factorization
does. You can then solve for many vectors b without repeating this work.
Here is the preceding example in Python:

```python
from bttb_solvers import prepare_cc_r

lags = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
b = list(range(16))
c = [1] * 16

with prepare_cc_r(lags, n=4, k=1) as solver:
    x = solver.solve(b)              # T x = b
    z = solver.solve(c)              # T z = c, using the same preparation
    y = solver.solve_transpose(b)    # Tᵀ y = b
```

The `with` statement releases the stored data when you finish. You can also
use `solve_many([b, c])` to solve a list of systems. HSC and HSC-q0 differ:
each call repeats the construction. Their purpose and determinant output are
explained in the [solver guide](SOLVERS.md).

Run these examples one at a time. Some arithmetic settings are shared within
a Python process, so simultaneous calls from different threads are unsupported.

## Decide what “fastest” means for your use

For a single system, compare the total time including preparation. For a fixed
matrix and many right-hand sides, compare preparation plus all the solves you
expect to perform. Include transpose solves and determinant computation if you
need them. A method with more preparation can become worthwhile after enough
solves.

The comparison example includes preparation and the requested solves in its
time. It checks answers separately afterwards; checks performed inside the
algorithms remain part of their time. A single run on a tiny example is only
a first indication, since loading the arithmetic library can take appreciable
time. For the paper's fixed experiments and repeated measurements, see
[reproduce/](../reproduce/README.md).

The examples normally use FLINT, an exact-arithmetic library installed with
the package. The Dense-LU comparison requires Linux x86-64 and the versions
listed in the installation instructions. If a method reports an unmet
assumption, try another applicable method. If a computed answer fails the
exact multiplication check, that is an error to investigate.

To understand how a method is implemented, continue with the
[code guide](CODE_GUIDE.md). To explore alternatives studied in the paper,
see [trials and outcomes](../REJECTED_ROUTES.md).
