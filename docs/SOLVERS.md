# Choosing and understanding a solver

Each method solves a system Tx=b over an exact finite field. Here T has n
block rows, with n-by-n Toeplitz blocks, so its order is N=n². A banded matrix
has block half-bandwidth k; a half-strip has at most w block superdiagonals.
The assumption **H0** means that the outermost block A[-w] is invertible.

Most methods separate the work into **preparation**, depending only on T, and
**solves**, depending on b. Think of the familiar separation between computing
LU factors and using them for successive right-hand sides. The name “query”
in the code means one of these subsequent solves.

For input instructions and a first calculation, see the
[practical guide](PRACTICAL.md). For the source files implementing these ideas,
see the [code guide](CODE_GUIDE.md).

## A first choice

| Problem or interest | Methods to consider |
|---|---|
| Banded systems | CC, CC-R, BFM, and Dense-LU |
| Half-strips satisfying H0 | HIR and Dense-LU; HSC-q0 for a determinant or a single solve |
| Half-strips without H0 | HSC for determinant and single solves; HPR and Dense-LU for repeated solves |
| Determinant of a possibly singular half-strip | HSC |
| Alternative boundary elimination | GBE |
| Effect of Strassen multiplication | BFM-Strassen and Dedicated-Strassen |
| Neighbouring structured constructions | The methods described in [trials and outcomes](../REJECTED_ROUTES.md) |

This table suggests comparisons, rather than a universal fastest method.
Some implementations have limitations beyond their input assumptions; these
are stated below, including the open results for HIR and HPR.

## CC and CC-R: wrap, diagonalize, correct

**Circulant correction (CC)** replaces the outer Toeplitz arrangement by a
block-circulant one: the block diagonals wrap around the ends. A Fourier
transform separates this circulant problem into n smaller Toeplitz systems.
A low-rank correction accounts for the entries changed by wrapping. The
resulting correction equations are solved using the Woodbury identity.
If the matrix-valued symbol is singular, preparation includes an additional
repair and accounts for it in the correction.

Preparation constructs these smaller problems and the correction. Subsequent
solves apply the circulant inverse and the saved correction to each b.
**CC-R** does more preparation: it represents each smaller Toeplitz inverse by
four vectors, so applying it reduces to polynomial multiplication. The extra
work can be worthwhile when many right-hand sides share the same T.

These methods require a nonsingular banded T and suitable field size and roots.
For a field with q elements, the conditions in this square-block implementation
are `2*k < n`, `n` dividing `q-1`, and `(q-1)/n > 2*k*n`, with a primitive nth
root of unity. These are the paper's **HB conditions**. The public functions
use a prime field and can find the root; a caller may also supply it.

Use `prepare_cc` or `prepare_cc_r`. Both solve Tx=b and Tᵀx=b using the same
preparation. Neither exposes a determinant calculation through this interface.
The construction follows [Chesnokov and Van Barel (2010)](../REFERENCES.md#chesnokov-and-van-barel-2010).

## CC-E and CC-E-R: work in an extension field

If the original field lacks the required root, an extension field may contain
it. CC-E and CC-E-R put the coefficients into that extension, run CC or CC-R,
and check that the answer belongs to the original prime field. The HB
conditions must hold in the extension. If the characteristic divides n,
an extension cannot supply the required primitive nth root.

Use `extension_field(p, d)` and `root_of_unity(field, n)`, then pass that field
and root to `prepare_cc_e` or `prepare_cc_e_r`. These are the extension-field
implementations measured in the paper. The measurements treat constructing
the field and root, converting the input, and returning to the original field
separately from solving. Include that work when comparing total time for your
own problem; a general cost theorem for the complete procedure is not claimed.

## BFM: eliminate interiors and merge boundary fronts

**Boundary-front merge (BFM)** divides the block rows into intervals. Within
each interval it eliminates interior unknowns, leaving equations for the
boundary unknowns. It then merges neighbouring intervals and repeats until
only the final boundary system remains. Preparation saves the equations
needed to recover the eliminated unknowns. A solve propagates the right-hand
side through this construction and recovers x.

BFM requires a nonsingular banded matrix. It needs neither roots of unity nor
invertibility of the outermost block. Use `prepare_bfm`. It solves both Tx=b
and Tᵀx=b, but does not expose a determinant calculation. For these Toeplitz
inputs, the identity Tᵀ=JTJ, where J reverses coordinates, lets the transpose
solve use the same preparation.

BFM normally uses ordinary matrix multiplication. Choose
`product="strassen"` to obtain **BFM-Strassen**. The
[Strassen example](../examples/README.md#explore-neighbouring-approaches) shows
how to check whether recursive Strassen products actually occur.

## GBE: another boundary-elimination comparison

**Guarded boundary elimination (GBE)** also eliminates unknowns while keeping
boundary equations. Its way of organizing the elimination and applying the
saved equations differs from BFM's. “Guarded” refers to explicit checks in the
calculation, including exact checks of the resulting solutions.

It solves nonsingular banded systems without requiring transform roots. Use
`prepare_gbe` for forward and transpose solves; there is no determinant
calculation in this interface. The default uses the Python arithmetic version.
Replacing its basic calculations by FLINT routines was slower on most of the
paper's tested cases, so that replacement remains an optional comparison.

## HIR: solve by recurrence and impose the final boundary conditions

**Half-strip impulse response (HIR)** uses invertibility of A[-w] to express
successive unknown blocks by a recurrence. Preparation computes responses to
elementary inputs, then forms and factors the smaller system imposing the
final boundary conditions. For a new b, it computes the forced response,
solves those boundary equations, and reconstructs x.

Use `prepare_hir`. Besides H0, this implementation requires roots of unity
for the polynomial products used in the recurrence. It provides forward and
transpose solves and a determinant. These outputs pass the finite exact checks
reported here. As stated in the paper, the general correspondence of its
determinant and transpose calculations to the written forward construction
remains unproved.

## HSC and HSC-q0: determinant and single solves for half-strips

**Half-strip classical (HSC)** allows the outermost block to be singular. It
makes a controlled low-rank correction to construct the required power series,
then accounts for that correction in a smaller structured boundary problem.
Polynomial approximation computes the determinant and, when T is nonsingular,
the solution.

Use `solve_hsc(lags, n, w, b)`. The result contains `determinant`, `solution`,
and `singular`. HSC returns determinant zero when T is singular; it does not
offer a solution of a singular system. To request only the determinant, omit b.
Every call repeats the construction, even when T is unchanged.

**HSC-q0** is the version for invertible A[-w]. This condition reduces the
number of power-series coefficients required, from `n+w+3*n*w` to `n+w`.
Use `solve_hsc_q0`; it rejects an input failing H0 and also repeats the
construction on every call. Here “q0” names the zero-corank case.

## HPR: save the half-strip preparation for repeated solves

**Half-strip prepared reconstruction (HPR)** saves the part of the HSC
construction depending only on T. New right-hand sides use those saved data
to form boundary equations, solve them, and recover x. It allows a singular
outermost block, provided the full matrix is nonsingular.

Use `prepare_hpr`. The default factors the boundary matrix by ordinary dense
LU and reuses those factors. This is the HPR implementation timed in the paper.
It offers forward and transpose solves and a determinant. It used less memory
than full dense LU on some reported cases, but did not win the CPU-time
comparisons. Its general theorem for prepared solves remains open.

Two experimental choices change how the boundary equations are solved:

| `terminal_query` | What is saved and reused |
|---|---|
| `"lu"` (default) | LU factors of the boundary matrix |
| `"division"` | A polynomial approximation basis, used to solve by polynomial division |
| `"appended"` | The matrix-dependent data for a new polynomial approximation problem with each b |

The latter two choices are research comparisons with their own checks and
unresolved theoretical questions. Their timings should be distinguished from
the default HPR timings. See [trials and outcomes](../REJECTED_ROUTES.md).

## Dense references: ordinary exact linear algebra

`prepare_dense_lu` forms the full N-by-N matrix, computes one LU factorization,
and reuses it for forward solves, transpose solves, and the determinant. This
**Dense-LU** comparison is especially useful when assessing repeated solves:
ordinary LU can reuse preparation too. Use the `with` statement shown in the
practical guide to release its stored matrices when finished.

The implementation requires a nonsingular matrix and the Linux x86-64 library
versions in the [installation instructions](../README.md#install-and-try).
The reproduction tables also include **Dense**, which performs separate
elimination calculations instead of reusing one LU factorization. These two
comparisons measure different amounts of work.

## Implementations and mathematical bounds

FLINT supplies fast exact arithmetic for many of these methods. Some FLINT
routines use ordinary dense calculations in places where a theoretical
construction uses structured operations. A good running time therefore does
not by itself demonstrate the paper's asymptotic bound. The paper's general
fast-exponent equality-test trees are not implemented here.

For alternatives involving displacement generators, other polynomial
multipliers, and different boundary constructions, see
[trials and outcomes](../REJECTED_ROUTES.md). The [code guide](CODE_GUIDE.md)
explains where the main methods and these experiments live; the
[references](../REFERENCES.md) give their mathematical sources.
