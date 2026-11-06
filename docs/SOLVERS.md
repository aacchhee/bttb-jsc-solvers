# Choosing and understanding a solver

Start with a matrix class and the service you need. Here `n` is the block count
and block order, so the physical system has order `N=n*n`. A banded input has
outer half-bandwidth `k`. A half-strip input has `w` permitted block
superdiagonals. **H0** means that its outermost block `A[-w]` is invertible.

The public entry points live in [api.py](../bttb_solvers/api.py). The
[code guide](CODE_GUIDE.md) maps their implementations and representations.
Runnable [examples](../examples/README.md) show setup, reuse and exact checks.

## Which part of the repository is useful for what?

| Purpose | Start here | How to interpret it |
|---|---|---|
| Solve a banded system and reuse preparation | CC, CC-R or BFM | Main solver interfaces. Domain checks still apply. |
| Work with half-strips and invertible H0 | HIR; HSC-q0 for the determinant/one-shot route | HIR retains state; HSC-q0 repeats construction per call. |
| Work without H0 | HSC or HPR | HSC gives the all-rank determinant route. HPR is a reusable, measured implementation with open general correspondence/bounds. |
| Obtain a reusable dense reference | Dense-LU | Paid physical matrix construction and one native LU. |
| Compare root-free boundary elimination | GBE | A usable exact solver retained as the paper's comparator. |
| Study explicit Strassen products | BFM-Strassen and `solve_dedicated(..., product="strassen")` | Actual recursive code, with recursion ledgers. Performance conclusions are finite comparisons. |
| Study displacement/approximant constructions | `solve_dedicated`, Structured-CC, Structured-halfstrip, multiplier variants | Trial/comparison implementations; see [REJECTED_ROUTES.md](../REJECTED_ROUTES.md). |
| Reproduce measurements | [reproduce/](../reproduce/README.md) | Fixed inputs and serial measurements, independently checked by exact residuals. |

“Main interface” describes usability and documentation. It does not promote an
implementation's mathematical guarantee. In particular, HPR and the extension
adapters retain the limitations stated in the paper.

## CC and CC-R: wrap, diagonalize, correct

**Problem:** a nonsingular banded two-level Toeplitz matrix. The supplied field
has size `q`, and the implementation checks `2*k<n`, `n | q-1`,
`(q-1)/n > 2*k*n`, and the order of the supplied root. The convenience public
interfaces construct a prime field; callers may supply `root=` explicitly.

**Idea:** wrap the outer block index to obtain a block-circulant matrix. Its DFT
has `n` one-level Toeplitz frequency blocks. Correct the wrap-around entries by
a smaller Woodbury system. If the matrix-valued symbol is singular, the setup
constructs a repair and includes that repair in the correction.

**Setup and stored state:**

1. [`cc_prepared.py`](../bttb_solvers/_internal/cc_prepared.py),
   `CirculantCorrectionPrepared.__init__`, checks the domain, computes the symbol
   determinant, performs any repair and chooses a valid circulant parameter.
2. [`banded_bttb/circulant.py`](../bttb_solvers/_internal/banded_bttb/circulant.py)
   constructs frequency lag vectors and the required inverse-corner information.
3. [`banded_bttb/woodbury.py`](../bttb_solvers/_internal/banded_bttb/woodbury.py)
   constructs and factors the correction matrix.
4. CC-R additionally constructs `ToeplitzInverseFactors`: four vectors per
   frequency, used through scalar polynomial convolutions.

**Query:** `query(rhs, transpose=False)` applies the circulant inverse and the
stored correction. `transpose=True` follows the literal transposed operations.
CC uses one-level Toeplitz solves during queries; CC-R applies the retained
four-vector factors. CC-R has extra preparation work and storage, so distinguish
one-shot and repeated-query comparisons. Both public prepared arms expose solves
only; their determinant identity is not a timed determinant API here.

**Compiled implementation:** [`cc_kernels.py`](../bttb_solvers/_internal/cc_kernels.py)
selects FLINT implementations of the scalar/Toeplitz/correction operations.
Some are dense cubic kernels replacing structured reference kernels. The
arithmetic bound of the structured method does not automatically describe
these replacements. Construction credit: [Chesnokov–Van Barel 2010](../REFERENCES.md#chesnokov-and-van-barel-2010).

## CC-E and CC-E-R: the same construction over a supplied extension

These embed prime-field input coefficients into `GF(p**d)`, use CC or CC-R there,
and require every solution coordinate to descend to `GF(p)`. They are useful
when a transform root is unavailable in the base field but exists in the
extension. The HB inequalities must hold in the supplied extension.

Use `extension_field(p, d)` and `root_of_unity(field, n)` before
`prepare_cc_e(..., field=field, root=root)` or `prepare_cc_e_r(...)`.
[`extension_field.py`](../bttb_solvers/_internal/extension_field.py) owns field and
polynomial contexts; [`extension_kernels.py`](../bttb_solvers/_internal/extension_kernels.py)
implements extension arithmetic, including regular-representation matrix kernels.

Field/root construction, embedding and descent are separate from the timed
solver phases in the reproduction driver. This is the paper's measured adapter;
a general end-to-end extension acquisition/descent cost theorem is not claimed.

## BFM: eliminate interiors and merge boundary fronts

**Problem:** nonsingular banded input. No transform root or H0 assumption.
The public interface supplies a prime-field adapter.

**Setup:** [`boundary_front/api.py`](../bttb_solvers/_internal/boundary_front/api.py)
constructs `BoundaryFrontPrepared`. `_Core` builds a balanced tree of intervals.
[`homogeneous.py`](../bttb_solvers/_internal/boundary_front/homogeneous.py) eliminates
interiors and merges the remaining boundary equations. Equal interval types
share prepared data. [`selection.py`](../bttb_solvers/_internal/boundary_front/selection.py)
performs guarded row selection; the root boundary system is inverted once.
The retained objects are local dense fronts and maps, not a dense `N`-by-`N`
physical inverse.

**Query:** [`query.py`](../bttb_solvers/_internal/boundary_front/query.py) batches
upward and downward propagation by interval type. For raw two-level Toeplitz
lags, transpose queries reverse the RHS and result, reusing the same core through
`T^t = J T J`. The internal arbitrary-dense-block interface instead prepares a
second transposed core lazily; it cannot assume Toeplitz persymmetry. Solves are
checked by the selected residual observer. There is no determinant API.

**Strassen:** `product="strassen", strassen_leaf=16` selects explicit recursion
in [`boundary_front/kernel.py`](../bttb_solvers/_internal/boundary_front/kernel.py).
The default product is classical. Inspect `prepared.state.statistics()` for
setup/query counters and `strassen_ledger`. Changing this switch changes local
products; it does not turn a front into a displacement generator.

## GBE: guarded scalar boundary elimination

GBE solves the same nonsingular banded class without transform roots. It is the
retained comparison for the boundary-front approach. It uses guarded boundary
normalization and individual query propagation rather than BFM's batched query
schedule. At width one, a dedicated three-block recurrence is used.

`prepare_gbe` enters [`guarded_boundary/unified.py`](../bttb_solvers/_internal/guarded_boundary/unified.py)
with `strategy="rootfree"`. Setup selects `operations.prepare` for width one or
`banded.prepare_banded` otherwise. Prepared forward/transpose queries include
exact physical residual guards. It exposes no determinant service. `backend="python"`
is the retained comparator; the compiled replacement was slower on most of the
paper's sampled cases.

## HIR: invert H0, generate impulses, factor a terminal system

**Problem:** half-strip input with invertible `F0=A[-w]`, and a prime supporting
the required NTT lengths. The public wrapper selects `n_impulse` response layout
and a dense terminal.

**Setup:** [`hir_prepared.py`](../bttb_solvers/_internal/hir_prepared.py) constructs
[`halfstrip/solver.py`](../bttb_solvers/_internal/halfstrip/solver.py)'s `HalfStripSolver`.
`leading.py` builds the compact Toeplitz inverse of `F0`. `recurrence.py` and
`convolve.py` compute causal responses using CDQ convolution. The impulse tails
form a terminal matrix, which `dense.py` factors and retains.

**Query:** run the forced causal recurrence, solve the terminal system, and
reconstruct physical coordinates. A transpose query reverses both coordinate
axes around the forward solve, using persymmetry. The prepared object exposes
its determinant. The executed determinant/transpose services have finite exact
checks; their full correspondence to the written forward construction remains
unproved, as stated in the paper.

## HSC and HSC-q0: carrier, defect and mosaic terminal

**Problem:** half-strip matrices, allowing singular `F0`. HSC supplies a
determinant at every rank, and a solution when the physical matrix is nonsingular.
The public wrapper uses the full pipeline for each right-hand side.

**Construction:**

1. `halfstrip_core/field_setup.py` supplies a working algebra and labels. Guards
   may restrict an added algebra and restart from the original input.
2. `carrier.py` adds a controlled low-rank correction so that an inverse-series
   carrier can be computed even when `F0` is singular.
3. `joint.py` compiles the small defect data and the truncated dictionaries
   needed for the terminal system and physical reconstruction.
4. `mosaic.py` builds a mosaic Toeplitz terminal and computes its determinant and
   affine solve through approximant bases.
5. `solver.py` applies the determinant scaling and reconstructs/descends the
   physical solution.

The public `solve_hsc` calls `halfstrip_core/reuse.py::oneshot_solve`: despite the
module's name, that function repeats the entire construction. It returns a
result dictionary, not a reusable preparation.

HSC-q0 uses [`hsc_solver.py`](../bttb_solvers/_internal/hsc_solver.py) with
`precision_mode="q0_L"`. It checks that the outermost-block corank is zero and
reduces source precision to `n+w`; otherwise it refuses. It remains a one-shot
construction. In full precision the source window is `n+w+3*n*w`.

The linear/polynomial algebra behind the terminal is described by the
[PM-Basis and approximant-basis sources](../REFERENCES.md#approximant-and-polynomial-matrix-algebra).

## HPR: retain preparation, choose an explicit terminal-query method

`prepare_hpr` calls `halfstrip_core/reuse.py::prepare_failed_h0_halfstrip`.
`FailedH0Reuse` owns the working algebra and restart lifetime. Its `_Core` first
runs the same right-hand-side-independent carrier/defect/terminal construction
as HSC and retains the dictionaries `W`, `B`, `G` and reconstruction shuffle.

The `terminal_query` argument has three distinct meanings:

| Value | Retained terminal data | Work per query | Role |
|---|---|---|---|
| `"lu"` | Dense terminal matrix and guarded LU factors | Forward/back substitution, then physical reconstruction | The paper's timed HPR implementation and the public default. |
| `"division"` | Root approximant basis and truncated inverse of its reversal | Polynomial division/remainder, then reconstruction | Experimental correspondence/cost route with runtime certificate checks. |
| `"appended"` | Terminal coefficient window and degree/profile data | A new affine approximant problem with an appended RHS | A comparison for separating reuse of the front end from terminal-query work. |

The constructor is shared, but these query implementations must be reported
separately. HPR's general prepared-query theorem remains open. In the recorded
one-LU comparisons, the default implementation has a memory advantage on named
cases and no tested CPU win. That is a finite performance finding, not a
rejection of the correctness of every HPR query construction.

## Dense references

`prepare_dense_lu` uses [`dense_lu.py`](../bttb_solvers/_internal/dense_lu.py).
`NativeLU.from_problem` builds the **physical** `N`-by-`N` matrix; `factorize`
calls native FLINT LU once. Forward and literal transpose batches reuse those
factors. The determinant is permutation sign times the diagonal product. Its
native allocations must be closed; the public context manager does this.
This baseline refuses singular physical matrices and unsupported native ABIs.

The reproduction label `Dense` instead calls public python-flint elimination
separately for forward, transpose and determinant services. Keep these reference
costs separate when reading or generating speedups.

## Trials and references

The dedicated displacement methods, structured half-strip prototype, structured
CC reference and scalar-backend controls have different inputs, stored data and
costs from the main routes above. Their purpose and current disposition are in
[REJECTED_ROUTES.md](../REJECTED_ROUTES.md). CK and KN are author abbreviations,
explained in the [code guide](CODE_GUIDE.md#ck-and-kn-what-these-prefixes-mean)
and [references](../REFERENCES.md), not unlabelled recommendations.
