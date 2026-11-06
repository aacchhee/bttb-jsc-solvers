# Algorithmic references

These are the sources cited by the accompanying paper and relevant to this
implementation. A source reference identifies the mathematical construction;
it does not assert that every fast primitive or bound in that source is
implemented here. The implementation boundaries are in [SOLVERS](docs/SOLVERS.md),
[CODE_GUIDE](docs/CODE_GUIDE.md) and [REJECTED_ROUTES](REJECTED_ROUTES.md).

## Chesnokov and Van Barel (2010)

A. Chesnokov and M. Van Barel, **A direct method to solve block banded block
Toeplitz systems with non-banded Toeplitz blocks**, *Journal of Computational
and Applied Mathematics* 234(5), 1485–1491.
[DOI: 10.1016/j.cam.2010.02.029](https://doi.org/10.1016/j.cam.2010.02.029).
CC's block-circulant embedding and low-rank correction follow this construction.
This package realizes the operations exactly over its finite-field adapters.

## Strassen (1969)

V. Strassen, **Gaussian elimination is not optimal**, *Numerische Mathematik*
13, 354–356. [DOI: 10.1007/BF02165411](https://doi.org/10.1007/BF02165411).
Source of the seven-product matrix multiplication used explicitly in the
BFM-Strassen and dedicated Strassen comparisons. Classical leaves and padding
are implementation choices; FLINT's internal multiplication selection is opaque.

## Cantor and Kaltofen (1991)

D. G. Cantor and E. Kaltofen, **On fast multiplication of polynomials over
arbitrary algebras**, *Acta Informatica* 28(7), 693–701.
[DOI: 10.1007/BF01178683](https://doi.org/10.1007/BF01178683).
`halfstrip/ck_poly.py` implements a cyclotomic-quotient multiplication variant
from this line. `allfield_poly.py` supplies the separately specified uniform
synthetic-transform implementation used by other code paths. Neither name
means that a whole linear-system solver has been selected.

## Khichane and Neiger (2026)

S. Khichane and V. Neiger, **Matrices with displacement structure: a deterministic
approach for linear systems and nullspace bases**,
[arXiv:2603.02425v1](https://arxiv.org/abs/2603.02425v1), preprint.
The paper uses this precise preprint version and treats its stated results as
named premises. The supplied-generator three-phase solving construction is the
reference for `kn_services.py`, with implementation components in the other
`kn_*` modules. Raw physical generator acquisition is separate. The repository
contains bounded implementations/controls; its presence does not implement the
paper's general fast-exponent equality-test trees.

## Approximant and polynomial-matrix algebra

- P. Giorgi, C.-P. Jeannerod and G. Villard (2003), **On the complexity of
  polynomial matrix computations**, ISSAC, 135–142.
  [DOI: 10.1145/860854.860889](https://doi.org/10.1145/860854.860889).
  PM-Basis polynomial-matrix computation underpinning terminal algorithms.
- C.-P. Jeannerod, V. Neiger and G. Villard (2020), **Fast computation of
  approximant bases in canonical form**, *Journal of Symbolic Computation*
  98, 192–224. [DOI: 10.1016/j.jsc.2019.07.011](https://doi.org/10.1016/j.jsc.2019.07.011).
  Canonical and known-degree approximant-basis constructions.
- C.-P. Jeannerod, V. Neiger, É. Schost and G. Villard (2017), **Computing
  minimal interpolation bases**, *Journal of Symbolic Computation* 83, 272–314.
  [DOI: 10.1016/j.jsc.2016.11.015](https://doi.org/10.1016/j.jsc.2016.11.015).
  Interpolation/striped-Krylov ingredients for small-order leaves.
- J. Rosenkilde and A. Storjohann (2021), **Algorithms for simultaneous
  Hermite–Padé approximations**, *Journal of Symbolic Computation* 102, 279–303.
  [DOI: 10.1016/j.jsc.2019.07.026](https://doi.org/10.1016/j.jsc.2019.07.026).
  Simultaneous approximation components of the supplied-generator construction.
- W. Zhou, G. Labahn and A. Storjohann (2012), **Computing minimal nullspace
  bases**, ISSAC, 366–373. [DOI: 10.1145/2442829.2442881](https://doi.org/10.1145/2442829.2442881).
  Polynomial kernel-basis ingredients discussed in the paper's premises.
- G. Labahn, V. Neiger and W. Zhou (2017), **Fast, deterministic computation of
  the Hermite normal form and determinant of a polynomial matrix**, *Journal of
  Complexity* 42, 44–71.
  [DOI: 10.1016/j.jco.2017.03.003](https://doi.org/10.1016/j.jco.2017.03.003).
  Polynomial-matrix determinant background for the stated source constructions.

## Software dependency

[FLINT](https://flintlib.org) and [python-flint](https://python-flint.readthedocs.io)
provide exact finite-field polynomials and dense matrix kernels. Reproduction
pins python-flint 0.9.0 / FLINT 3.6.0. They are installed dependencies under their
own licenses; no dependency source or wheel is vendored here.
