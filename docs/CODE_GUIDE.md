# Finding your way around the code

The code follows the mathematics: it stores Toeplitz coefficients, constructs
data for a chosen method, and uses those data to solve systems. This guide
explains where those steps live. You can use the package without reading all
of them; start with [solving your own problem](PRACTICAL.md) if that is your goal.

## Start with one method

The functions intended for users are in [api.py](../bttb_solvers/api.py).
For example, `prepare_cc_r` prepares CC-R for a particular matrix and returns
an object on which `solve(b)` solves Tx=b. The word **prepare** means work
that depends on T but not on b, just as computing LU factors does.

The algorithm implementations are under `bttb_solvers/_internal/`. The name
`_internal` means their individual functions may change as the code develops;
the functions in `api.py` provide the common way to use them.

| If you want to read… | Start here | What happens there |
|---|---|---|
| CC and CC-R | [cc_prepared.py](../bttb_solvers/_internal/cc_prepared.py) | Build the circulant approximation and its correction; solve using the saved data |
| BFM | [boundary_front/api.py](../bttb_solvers/_internal/boundary_front/api.py) | Prepare the subdivision of the matrix and connect it to the solve routines |
| GBE | [guarded_boundary/unified.py](../bttb_solvers/_internal/guarded_boundary/unified.py) | Choose the appropriate boundary-elimination calculation for the bandwidth |
| HIR | [hir_prepared.py](../bttb_solvers/_internal/hir_prepared.py), then [halfstrip/solver.py](../bttb_solvers/_internal/halfstrip/solver.py) | Compute responses of the recurrence and solve the remaining boundary equations |
| HSC | [halfstrip_core/solver.py](../bttb_solvers/_internal/halfstrip_core/solver.py) | Assemble the polynomial and boundary calculations for determinant and solution |
| HPR | [halfstrip_core/reuse.py](../bttb_solvers/_internal/halfstrip_core/reuse.py) | Save the matrix-dependent part of that calculation for later right-hand sides |
| Ordinary dense LU | [dense_lu.py](../bttb_solvers/_internal/dense_lu.py) | Form T explicitly, factor it once, and reuse its factors |

For HSC, the public function reaches `solver.py` through `oneshot_solve` in
`reuse.py`; despite that filename, this particular call repeats the construction.
HSC-q0 starts in [hsc_solver.py](../bttb_solvers/_internal/hsc_solver.py).
The [solver guide](SOLVERS.md) explains the mathematical differences.

## How is a matrix represented?

The original input is a list of Toeplitz coefficients. The function `action`
in [matrices.py](../bttb_solvers/matrices.py) multiplies this matrix by a vector
without first writing out all its entries. This is also how the examples check
an answer: they compute Tx and compare it with b.

Different methods then store different information about the matrix:

- **Dense-LU** writes out the full n²-by-n² matrix and stores its LU factors.
- **CC** works with n smaller Toeplitz matrices obtained by a Fourier transform,
  together with a correction for the entries changed by the circulant embedding.
  **CC-R** stores each smaller inverse through four vectors, applied using
  polynomial multiplication.
- **BFM and GBE** eliminate unknowns in groups and keep equations relating the
  remaining boundary unknowns. Their dense matrices describe these smaller
  problems, rather than an explicitly stored inverse of T.
- **HIR** uses a recurrence. It stores responses to elementary inputs and a
  factorization of the matrix imposing the final boundary conditions.
- **HSC and HPR** use polynomial matrices and truncated power series to reduce
  the original problem to boundary equations. HPR saves the data needed to
  repeat this reduction for new right-hand sides.

Thus a “dense matrix” mentioned inside an algorithm may be only a smaller
boundary system. It does not necessarily mean that the full T has been formed.
Those smaller systems can nevertheless account for substantial memory use.

Some research methods instead use **displacement generators**. The idea is
that a matrix M can be described by the low-rank difference between M and a
shifted copy of itself. In the Stein form used here,

```text
M − Z M Zᵀ = g hᵀ,
```

where Z shifts coordinates down by one position. If g and h have few columns,
they give a compact description of M. These generators still have to be
computed from the original problem. An experiment that starts with g and h
already supplied measures a different task from solving directly from the
Toeplitz coefficients. The [generator example](../examples/generator_terminal.py)
illustrates the former.

## Where are the basic calculations?

The algorithms share routines for finite-field arithmetic, polynomial
multiplication, matrix multiplication, and elimination. There is no need to
read their programming conventions before understanding the algorithms.

| Calculation | Main files |
|---|---|
| Arithmetic in prime and extension fields | [banded_bttb/field.py](../bttb_solvers/_internal/banded_bttb/field.py), [extension_field.py](../bttb_solvers/_internal/extension_field.py), [halfstrip_core/exact.py](../bttb_solvers/_internal/halfstrip_core/exact.py) |
| Toeplitz and circulant operations for CC | [banded_bttb/](../bttb_solvers/_internal/banded_bttb/) |
| Polynomial products and truncated series for half-strips | [halfstrip_core/poly.py](../bttb_solvers/_internal/halfstrip_core/poly.py), [halfstrip/allfield_poly.py](../bttb_solvers/_internal/halfstrip/allfield_poly.py) |
| Elimination and merging of BFM boundary equations | [boundary_front/homogeneous.py](../bttb_solvers/_internal/boundary_front/homogeneous.py); applying the saved equations is in [query.py](../bttb_solvers/_internal/boundary_front/query.py) |
| Recurrence calculations for HIR | [halfstrip/recurrence.py](../bttb_solvers/_internal/halfstrip/recurrence.py) |
| Structured boundary equations for HSC/HPR | [halfstrip_core/mosaic.py](../bttb_solvers/_internal/halfstrip_core/mosaic.py) |

Many operations have both a straightforward Python implementation and an
implementation using **FLINT**, the exact-arithmetic library installed with
this package. The argument `backend="python"` or `backend="flint"` selects
between them where supported. Files such as `cc_kernels.py` and
`halfstrip_core/flint_backend.py` connect the algorithms to FLINT's routines.
Here “kernel” simply means a routine for a basic calculation.

A faster library operation may use a different algorithm from the one used
in the paper's complexity argument. Its measured speed and that argument's
operation count therefore need to be distinguished.

## Where is Strassen multiplication?

Strassen's method replaces eight half-size matrix products by seven, at the
cost of additional additions. The code stops subdividing at a chosen size and
uses ordinary multiplication there. Small problems may never reach a
subdivision at all.

The repository has two implementations because the kinds of matrices differ:

- [boundary_front/kernel.py](../bttb_solvers/_internal/boundary_front/kernel.py)
  contains the products used by **BFM-Strassen**.
- [halfstrip_core/uniform_terminal.py](../bttb_solvers/_internal/halfstrip_core/uniform_terminal.py)
  contains products of constant and polynomial matrices used by
  **Dedicated-Strassen**.

Run [examples/strassen.py](../examples/README.md#explore-neighbouring-approaches)
to try either. It prints counts showing whether recursive subdivision actually
occurred. The paper uses stopping sizes 16 and 8 respectively. The file
`halfstrip_core/dense_backend.py` selects ordinary or Strassen multiplication
and collects these counts; the multiplication itself is in the files above.

## CK and KN: what these prefixes mean

**CK means Cantor–Kaltofen.** Their construction multiplies polynomials over
arbitrary fields without requiring roots of unity in the original field.
[halfstrip/ck_poly.py](../bttb_solvers/_internal/halfstrip/ck_poly.py) implements
one such multiplication method. It is an ingredient that can be used within
a solver. The file `halfstrip/allfield_poly.py` provides another arbitrary-field
multiplier. The [polynomial example](../examples/README.md#explore-neighbouring-approaches)
compares their answers and lets you inspect their calculations.

**KN means Khichane–Neiger.** The `kn_*` files implement ingredients of their
approach to solving matrices given by displacement generators. Begin with
[kn_services.py](../bttb_solvers/_internal/kn_services.py) for how these ingredients
are combined. The other files handle polynomial arithmetic, polynomial kernel
bases, approximation, and related intermediate problems. The
[detailed file reference](FILE_REFERENCE.md#kn-supplied-generator-algebra)
locates each part; [REFERENCES](../REFERENCES.md) gives the papers and the precise
preprint version used.

These author abbreviations identify mathematical sources. They do not identify
recommended solvers or imply that every algorithm and bound in those sources
has been implemented.

## Which files are useful experiments, and which are trials?

For solving a new problem, use the methods listed in the first table through
`api.py`. For exploring alternatives, begin with
[trials and outcomes](../REJECTED_ROUTES.md). It explains why each comparison was
retained and what the measurements do and do not establish.

Three useful starting points are:

| Question | Files to explore |
|---|---|
| What happens if the boundary system is solved using displacement generators? | `dedicated_banded_arithmetic.py`, then `kn_services.py` |
| What does the alternative structured half-strip construction do? | `full_halfstrip.py` |
| How does the generator-based circulant comparison differ from CC? | `hb_physical_solver.py`, rather than the main `cc_prepared.py` |

The `hb_*` filenames refer to the paper's **HB class**, the root and field-size
conditions stated under [CC](SOLVERS.md#cc-and-cc-r-wrap-diagonalize-correct).
These files belong to the Structured-CC comparison. A trial may use valuable
shared arithmetic even when its complete solver was slower in the experiments.

To change an input, edit an example data file. To compare ordinary and Strassen
products, use the Strassen example. To regenerate the paper's experiments, use
[reproduce/](../reproduce/README.md). These are convenient places to experiment
before changing the algorithm implementations.

For a particular file not listed here, consult the
[complete file reference](FILE_REFERENCE.md). Each internal source file also
begins with a short description of its purpose.
