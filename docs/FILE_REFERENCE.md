# Detailed file reference

This catalogue is for locating a particular implementation detail. For an
overview of how the mathematics is divided among the files, first read
[the code guide](CODE_GUIDE.md). You do not need this catalogue to use a solver.

## Complete module index

Roles: **Main** is a main solver implementation; **Shared** is an algebra/representation
component; **Trial** is a retained comparison or alternate construction. **Mixed** files
also expose helpers used by main routes. The role is about this package, not a proof status.
Empty package `__init__.py` files provide namespaces only.

### Main CC setup and queries

| File | Role | Responsibility / reading entry |
|---|---|---|
| [cc_kernels.py](../bttb_solvers/_internal/cc_kernels.py) | Shared | Scoped FLINT replacements for CC Toeplitz, transform, repair and correction kernels; restores the original functions on scope exit. Start at `kernels`. |
| [cc_prepared.py](../bttb_solvers/_internal/cc_prepared.py) | Main | Construct the repaired block-circulant/Woodbury preparation; query it directly or with retained four-vector Toeplitz inverse factors. Start at `CirculantCorrectionPrepared; ToeplitzInverseFactors`. |
| [extension_field.py](../bttb_solvers/_internal/extension_field.py) | Shared | Native GF(p**d) field and polynomial contexts, coercions and guarded inverses for the supplied-extension CC adapters. Start at `FqField; FqFieldNative`. |
| [extension_kernels.py](../bttb_solvers/_internal/extension_kernels.py) | Shared | Scoped extension-field matrix and polynomial kernels, including regular-representation matrix arithmetic. Start at `kernels`. |

### Physical dense baseline

| File | Role | Responsibility / reading entry |
|---|---|---|
| [dense_lu.py](../bttb_solvers/_internal/dense_lu.py) | Main | Owned native FLINT matrices, checked ABI binding, one physical LU, batched literal-transpose/forward solves and determinant extraction. Start at `NativeAPI; OwnedMat; NativeLU`. |

### Main HIR construction

| File | Role | Responsibility / reading entry |
|---|---|---|
| [hir_prepared.py](../bttb_solvers/_internal/hir_prepared.py) | Main | Configure the HIR leading-block recurrence, response layout and terminal solver; the public wrapper selects n impulses and a dense terminal. Start at `prepare_hir; solve_hir`. |

### Alternative HIR terminals

| File | Role | Responsibility / reading entry |
|---|---|---|
| [halfstrip_structured_terminal.py](../bttb_solvers/_internal/halfstrip_structured_terminal.py) | Trial/shared | Alternative terminal representations and causal solves for recurrence-based half-strips; these are separate from the public HIR dense-terminal default. Start at `StructuredTerminal; DenseImpulseTerminal`. |

### Main HSC and HPR construction

| File | Role | Responsibility / reading entry |
|---|---|---|
| [hsc_solver.py](../bttb_solvers/_internal/hsc_solver.py) | Main | Raw-lag classical source adapter with precision selection and guarded algebra restart; used by the public HSC-q0 route. Public HSC uses oneshot_solve in halfstrip_core/reuse.py. Start at `solve_hsc; _attempt`. |

### Field setup and common representation tools

| File | Role | Responsibility / reading entry |
|---|---|---|
| [constant_profile.py](../bttb_solvers/_internal/constant_profile.py) | Shared | Exact constant-matrix row/rank-profile selection and kernel data; used by structured acquisition code. Start at `rank_profile_kernel; row_rank_profile`. |
| [fast_scalar.py](../bttb_solvers/_internal/fast_scalar.py) | Shared | Fast scalar-polynomial arithmetic used by label acquisition and physical front ends; reports nonunits to the owning restart loop. Start at `FastScalar; PolynomialNonUnit`. |
| [label_algebra.py](../bttb_solvers/_internal/label_algebra.py) | Shared | Acquire nonzero evaluation labels and an owning coefficient algebra; guard restrictions and descent remain attached to that construction. Start at `setup_label_algebra; LabelAlgebraContext`. |
| [native_field_view.py](../bttb_solvers/_internal/native_field_view.py) | Shared | Adapt explicit coordinate fields to the guarded native-operation interface used by source/generator acquisition. Start at `NativeBaseFieldView`. |
| [owned_label_algebra.py](../bttb_solvers/_internal/owned_label_algebra.py) | Shared | Own scalar embeddings, guarded operations, algebra restrictions and descent for label-based source constructions. Start at `OwnedLabelAlgebra`. |

### Displacement construction trials

| File | Role | Responsibility / reading entry |
|---|---|---|
| [banded_column_basis.py](../bttb_solvers/_internal/banded_column_basis.py) | Trial/shared | Polynomial column/image basis construction for the dedicated front end, with degree and reconstruction bookkeeping. Start at `ColumnBasis`. |
| [banded_frontend_rank.py](../bttb_solvers/_internal/banded_frontend_rank.py) | Trial/shared | Guarded maximum-minor and coordinate-completion operations for polynomial-symbol rank acquisition. Start at `certified_maximum_minor; narrow_raw_minor`. |
| [banded_general_approximant.py](../bttb_solvers/_internal/banded_general_approximant.py) | Trial/shared | General-order canonical approximant operations used by the raw polynomial front end. Start at `GeneralApproximant`. |
| [banded_physical_solver.py](../bttb_solvers/_internal/banded_physical_solver.py) | Trial | Dispatch the dedicated raw-lag solve to narrow or wide physical acquisition, paying front end, generator acquisition and both queries. Start at `solve_banded`. |
| [banded_polynomial_determinant.py](../bttb_solvers/_internal/banded_polynomial_determinant.py) | Trial/shared | Polynomial-symbol determinant acquisition from a certified maximum minor; signals the nonsingular determinant branch. Start at `determinant_from_maximum_minor; NonsingularDeterminant`. |
| [cauchy_transforms.py](../bttb_solvers/_internal/cauchy_transforms.py) | Trial/shared | Multipoint/Cauchy transform catalogs, reciprocal products and evaluation maps for generator acquisition. Start at `CauchyCatalog`. |
| [compact_cauchy.py](../bttb_solvers/_internal/compact_cauchy.py) | Trial/shared | Compact Cauchy transform operations used by the grouped source/chart constructions. Start at `CompactCauchy`. |
| [dedicated_banded_arithmetic.py](../bttb_solvers/_internal/dedicated_banded_arithmetic.py) | Trial | Select classical or explicit Strassen dense operations around the complete dedicated physical solve; the classical many-pair service has a separate reuse contract. Start at `solve_dedicated_banded; solve_dedicated_banded_many`. |
| [generator_compression.py](../bttb_solvers/_internal/generator_compression.py) | Trial/shared | Two-pass exact rank-factor compression of acquired displacement generators; preserves the represented matrix and records paid work. Start at `compress_generators; CompressedGenerators`. |
| [grouped_chart.py](../bttb_solvers/_internal/grouped_chart.py) | Trial/shared | Search and certify grouped/cyclic chart data consumed by narrow front-end acquisition. Start at `prepare_grouped_chart; select_cyclic_chart`. |
| [mixed_cross.py](../bttb_solvers/_internal/mixed_cross.py) | Trial/shared | Represent mixed upper/lower boundary crosses and physical displacement actions without an unconditional physical dense matrix. Start at `MixedUpperCross; MixedLowerCross; PhysicalDisplacement`. |
| [narrow_physical_solver.py](../bttb_solvers/_internal/narrow_physical_solver.py) | Trial | Narrow-band acquisition/restart, weighted carrier/correction construction, generator terminal and physical answer reconstruction. Start at `solve_narrow; solve_narrow_many`. |
| [narrow_raw_frontend.py](../bttb_solvers/_internal/narrow_raw_frontend.py) | Trial | Acquire a narrow raw polynomial symbol, chart and repaired front-end state; restrictions replay from the cached original lags. Start at `RawNarrowFrontend; prepare_narrow_frontend`. |
| [rectangular_cauchy.py](../bttb_solvers/_internal/rectangular_cauchy.py) | Trial/shared | Rectangular Cauchy subviews and products over selected row/column labels. Start at `RectangularCauchy`. |
| [spatial_transforms.py](../bttb_solvers/_internal/spatial_transforms.py) | Trial/shared | Spatial evaluation/correlation transforms used to move between raw Toeplitz data and carrier/generator coordinates. Start at `SpatialTransform; row_product; correlation`. |
| [toeplitz_repair.py](../bttb_solvers/_internal/toeplitz_repair.py) | Trial/shared | Repair a singular Toeplitz matrix-valued symbol within the acquired coefficient context; return the explicit low-rank correction. Start at `repair_toeplitz_symbol`. |
| [weighted_carrier.py](../bttb_solvers/_internal/weighted_carrier.py) | Trial/shared | Retain weighted carrier actions over the acquired chart; used before projected correction generators are constructed. Start at `WeightedRingCarrier; acquire_weighted_carrier`. |
| [weighted_correction.py](../bttb_solvers/_internal/weighted_correction.py) | Trial/shared | Assemble and expose the weighted wrap/repair correction and its generator-compatible maps. Start at `WeightedCorrection; acquire_weighted_correction`. |
| [weighted_projections.py](../bttb_solvers/_internal/weighted_projections.py) | Trial/shared | Build forward and transpose carrier projections and their physical actions for the correction construction. Start at `acquire_weighted_projections; ProjectedCarrier`. |
| [wide_physical_solver.py](../bttb_solvers/_internal/wide_physical_solver.py) | Trial | Acquire wide-band Stein generators, optionally compress them, and solve the resulting physical service through supplied-generator algebra. Start at `WidePhysicalSolver; solve_wide`. |
| [wide_raw_generator.py](../bttb_solvers/_internal/wide_raw_generator.py) | Trial | Construct wide-band physical displacement generators directly from original Toeplitz lags and their reversed orientation. Start at `WideGenerator; acquire`. |

### Structured CC comparison

| File | Role | Responsibility / reading entry |
|---|---|---|
| [hb_frontend.py](../bttb_solvers/_internal/hb_frontend.py) | Trial | Acquire the supplied-root structured CC front end, symbol rank/determinant, valid parameter and coordinate context. This is distinct from cc_prepared.py. Start at `prepare_hb_frontend`. |
| [hb_physical_solver.py](../bttb_solvers/_internal/hb_physical_solver.py) | Trial | Compose the structured supplied-root carrier, projected correction and terminal into a one-pair comparison, with an optional prepared classical inner base. Start at `solve_hb; prepare_hb`. |
| [hb_prepared_dense.py](../bttb_solvers/_internal/hb_prepared_dense.py) | Trial/shared | Retain the dense inner PLU base for the structured CC comparison. This is not CC-R four-vector frequency preparation. Start at `HBPreparedDense`. |

### Structured half-strip comparison

| File | Role | Responsibility / reading entry |
|---|---|---|
| [full_halfstrip.py](../bttb_solvers/_internal/full_halfstrip.py) | Mixed | Trial structured half-strip solve with explicit source histories and whole-attempt restart/descent; its input/restriction helpers are also shared by HSC-q0. Start at `solve_full; _input; _retained_factor`. |
| [full_source_readout.py](../bttb_solvers/_internal/full_source_readout.py) | Trial | Read source coefficients, form the terminal problem and recover a physical solution in the structured half-strip comparison. Start at `FullSource; solve_prepared`. |
| [initial_carrier.py](../bttb_solvers/_internal/initial_carrier.py) | Trial | Acquire the initial structured carrier for the explicit-history half-strip comparison. Main HSC/HPR use halfstrip_core/carrier.py instead. Start at `InitialCarrier`. |
| [initial_joint.py](../bttb_solvers/_internal/initial_joint.py) | Trial | Bind a structured carrier to the explicit joint state and its coefficient/window operations. Start at `InitialJoint`. |
| [joint_arrays.py](../bttb_solvers/_internal/joint_arrays.py) | Trial/shared | Fixed-window scalar series operations, polynomial factor trimming and permutation signs for the array-based joint representation. Start at `SeriesOps; trim_factor`. |
| [joint_blocks.py](../bttb_solvers/_internal/joint_blocks.py) | Trial/shared | Materialize requested physical/principal blocks from the explicit joint state when the construction pays for them. Start at `materialize_physical; whole_principal_block`. |
| [joint_compiler.py](../bttb_solvers/_internal/joint_compiler.py) | Trial/shared | Compile the explicit-history joint-defect representation. Main HSC/HPR use halfstrip_core/joint.py instead. Start at `compile_joint`. |
| [joint_state.py](../bttb_solvers/_internal/joint_state.py) | Trial/shared | Own the evolving array-based joint-defect state used by the structured half-strip construction. Start at `JointState`. |
| [joint_valuation.py](../bttb_solvers/_internal/joint_valuation.py) | Trial/shared | Select active valuations and extract common factors/column quotients during joint compilation. Start at `minimum_active; column_quotient; extract_common`. |
| [q0_source.py](../bttb_solvers/_internal/q0_source.py) | Trial/shared | Raw Toeplitz coefficient and spatial-convolution access for explicit source acquisition. The public HSC-q0 precision adapter lives in hsc_solver.py. Start at `SpatialConvolution; RawToeplitzLags`. |
| [source_histories.py](../bttb_solvers/_internal/source_histories.py) | Trial/shared | Retain principal and paired source-history windows used by explicit structured readout. Start at `PrincipalHistory; PairedHistories`. |

### KN supplied-generator algebra

| File | Role | Responsibility / reading entry |
|---|---|---|
| [kn_approximant.py](../bttb_solvers/_internal/kn_approximant.py) | Trial/shared | Canonical/known-degree approximant, vector approximation and simultaneous Hermite-Pade routines for the supplied-generator construction. Start at `Approximant`. |
| [kn_kernel.py](../bttb_solvers/_internal/kn_kernel.py) | Trial/shared | Polynomial kernel and degree-budget constructions, including rational_vector_xd for generator/RHS reconstruction. Start at `Kernel`. |
| [kn_krylov.py](../bttb_solvers/_internal/kn_krylov.py) | Trial/shared | Small-order canonical approximant leaf based on exact nilpotent/striped Krylov actions and row-rank profiles. Start at `canonical_leaf; row_rank_profile`. |
| [kn_modular_vector.py](../bttb_solvers/_internal/kn_modular_vector.py) | Trial/shared | Solve a polynomial-matrix/vector problem modulo a supplied polynomial or z**n-c for carrier and projection operations. Start at `ModularVector.solve_mod`. |
| [kn_polynomial.py](../bttb_solvers/_internal/kn_polynomial.py) | Shared | Khichane-Neiger-related scalar polynomial, matrix and degree-budget utilities, reused by both front-end and terminal components. Start at `Polynomial`. |
| [kn_services.py](../bttb_solvers/_internal/kn_services.py) | Trial | Khichane-Neiger-style supplied-generator solve/nullspace service: paid dense base and three approximation/reconstruction phases. Raw physical acquisition is external. Start at `Services.solve_stein; Services.solve_scalar`. |

### CC Toeplitz, symbol and correction primitives

| File | Role | Responsibility / reading entry |
|---|---|---|
| [banded_bttb/circulant.py](../bttb_solvers/_internal/banded_bttb/circulant.py) | Shared | Frequency symbols, direct/radix-two transforms, inverse-corner accumulation and circulant inverse application. |
| [banded_bttb/counter.py](../bttb_solvers/_internal/banded_bttb/counter.py) | Shared | Reference-operation and stage counters; compiled kernels do not invent equivalent scalar counts. |
| [banded_bttb/errors.py](../bttb_solvers/_internal/banded_bttb/errors.py) | Shared | Typed CC field, root, parameter, repair and singularity refusals. |
| [banded_bttb/field.py](../bttb_solvers/_internal/banded_bttb/field.py) | Shared | Explicit prime/extension coordinate fields, guarded arithmetic, size/order and supplied-root helpers. |
| [banded_bttb/polynomial.py](../bttb_solvers/_internal/banded_bttb/polynomial.py) | Shared | Exact scalar-polynomial utilities, Euclidean operations, interpolation and product trees. |
| [banded_bttb/repair.py](../bttb_solvers/_internal/banded_bttb/repair.py) | Shared | Normal-rank tests and low-rank Toeplitz symbol repair for CC. |
| [banded_bttb/symbol.py](../bttb_solvers/_internal/banded_bttb/symbol.py) | Shared | Raw banded symbol representation, frequency evaluation, determinant polynomial and valid-parameter search. |
| [banded_bttb/toeplitz.py](../bttb_solvers/_internal/banded_bttb/toeplitz.py) | Shared | One-level Toeplitz solve, inverse, determinant and rank primitives plus elementary matrix helpers. |
| [banded_bttb/woodbury.py](../bttb_solvers/_internal/banded_bttb/woodbury.py) | Shared | Wrap/repair maps, reduced correction matrix construction, factorization and forward/transpose correction. |

### BFM local-front implementation

| File | Role | Responsibility / reading entry |
|---|---|---|
| [boundary_front/api.py](../bttb_solvers/_internal/boundary_front/api.py) | Main/shared | BFM preparation, prepared queries and statistics. Raw Toeplitz transpose queries reuse the core; arbitrary dense blocks use a lazy second core. |
| [boundary_front/fields.py](../bttb_solvers/_internal/boundary_front/fields.py) | Main/shared | Native eight-operation fields and adapters, including LeanAttrField for the GBE comparator. |
| [boundary_front/homogeneous.py](../bttb_solvers/_internal/boundary_front/homogeneous.py) | Main/shared | Eliminate interiors and merge homogeneous boundary fronts by shared interval type. |
| [boundary_front/incidence.py](../bttb_solvers/_internal/boundary_front/incidence.py) | Main/shared | Describe how retained/seam/exterior rows enter a front merge. |
| [boundary_front/kernel.py](../bttb_solvers/_internal/boundary_front/kernel.py) | Main/shared | Local matrix products, classical/Strassen selection, rectangular blocking and native recursive seven-product squares. |
| [boundary_front/query.py](../bttb_solvers/_internal/boundary_front/query.py) | Main/shared | Batched upward and downward RHS propagation over the prepared interval schedule. |
| [boundary_front/residual.py](../bttb_solvers/_internal/boundary_front/residual.py) | Main/shared | Independent physical residual actions selected by the prepared BFM interface. |
| [boundary_front/schedule.py](../bttb_solvers/_internal/boundary_front/schedule.py) | Main/shared | Balanced occurrence tree and batches of equal interval types. |
| [boundary_front/selection.py](../bttb_solvers/_internal/boundary_front/selection.py) | Main/shared | Guarded row-profile selection and inverse maps for normalized boundary equations. |

### GBE boundary comparator

| File | Role | Responsibility / reading entry |
|---|---|---|
| [guarded_boundary/banded.py](../bttb_solvers/_internal/guarded_boundary/banded.py) | Main/shared | Retained-boundary elimination and queries for general outer half-bandwidth. |
| [guarded_boundary/boundary.py](../bttb_solvers/_internal/guarded_boundary/boundary.py) | Main/shared | Small exact dense/front operations and guarded normalization. |
| [guarded_boundary/fields.py](../bttb_solvers/_internal/guarded_boundary/fields.py) | Main/shared | Attribute-style exact fields expected by the boundary recurrence. |
| [guarded_boundary/operations.py](../bttb_solvers/_internal/guarded_boundary/operations.py) | Main/shared | Width-one repeated-block boundary preparation and query recurrence. |
| [guarded_boundary/unified.py](../bttb_solvers/_internal/guarded_boundary/unified.py) | Main/shared | Validate raw inputs, select width-one/general boundary preparation, retain transpose state and check query residuals. |

### HIR, shared multiplication and alternate terminal components

| File | Role | Responsibility / reading entry |
|---|---|---|
| [halfstrip/allfield_convolve.py](../bttb_solvers/_internal/halfstrip/allfield_convolve.py) | Shared | Select the uniform or Cantor-Kaltofen scalar multiplier for inverse-series and terminal controls. |
| [halfstrip/allfield_poly.py](../bttb_solvers/_internal/halfstrip/allfield_poly.py) | Shared | Uniform synthetic-transform polynomial multiplication with fixed leaves and a bounded native accelerator. |
| [halfstrip/allfield_solver.py](../bttb_solvers/_internal/halfstrip/allfield_solver.py) | Trial/shared | Uniform/CK inverse-series and approximant-terminal comparison; a separate route from public HSC and HIR. |
| [halfstrip/approximant.py](../bttb_solvers/_internal/halfstrip/approximant.py) | Shared | Reference PM-Basis, shifted-degree and polynomial-matrix operations used by mosaic terminals. |
| [halfstrip/block_approximant.py](../bttb_solvers/_internal/halfstrip/block_approximant.py) | Trial/shared | Block approximant terminal alternative for the recurrence construction. |
| [halfstrip/block_cauchy.py](../bttb_solvers/_internal/halfstrip/block_cauchy.py) | Trial/shared | Block-Cauchy representation supporting alternate structured recurrence terminals. |
| [halfstrip/block_levinson.py](../bttb_solvers/_internal/halfstrip/block_levinson.py) | Trial/shared | Block-Levinson terminal alternative with its own checked domain. |
| [halfstrip/ck_poly.py](../bttb_solvers/_internal/halfstrip/ck_poly.py) | Shared | Cantor-Kaltofen cyclotomic-quotient scalar multiplication alternative; fixed leaves and explicit recursion statistics. |
| [halfstrip/convolve.py](../bttb_solvers/_internal/halfstrip/convolve.py) | Shared | Schoolbook/NTT exact convolution selection for HIR recurrence and leading-block applications. |
| [halfstrip/counter.py](../bttb_solvers/_internal/halfstrip/counter.py) | Shared | HIR recurrence, convolution, terminal and structural-operation counters. |
| [halfstrip/dense.py](../bttb_solvers/_internal/halfstrip/dense.py) | Shared | Exact dense LU of the small recurrence terminal, separate from the full physical dense reference. |
| [halfstrip/errors.py](../bttb_solvers/_internal/halfstrip/errors.py) | Shared | Typed leading-block, transform, width and terminal failures. |
| [halfstrip/jacobi_series.py](../bttb_solvers/_internal/halfstrip/jacobi_series.py) | Shared | Jacobi/inverse-series source construction for the uniform/CK comparison routes. |
| [halfstrip/jacobi_solver.py](../bttb_solvers/_internal/halfstrip/jacobi_solver.py) | Trial/shared | Alternate Jacobi-series half-strip solver with a checked terminal-chart contract. |
| [halfstrip/leading.py](../bttb_solvers/_internal/halfstrip/leading.py) | Shared | Compact one-level Toeplitz inverse of the leading block and its convolution applications. |
| [halfstrip/recurrence.py](../bttb_solvers/_internal/halfstrip/recurrence.py) | Shared | Causal response generation and forcing propagation, including CDQ convolution. |
| [halfstrip/solver.py](../bttb_solvers/_internal/halfstrip/solver.py) | Shared | HIR response preparation, terminal construction, determinant and prepared physical solves. |
| [halfstrip/symbol.py](../bttb_solvers/_internal/halfstrip/symbol.py) | Shared | Raw half-strip lag storage and index/field conventions. |

### Main HSC and HPR series/mosaic implementation

| File | Role | Responsibility / reading entry |
|---|---|---|
| [halfstrip_core/carrier.py](../bttb_solvers/_internal/halfstrip_core/carrier.py) | Main/shared | Correct singular F0 by a controlled low-rank carrier and construct the required inverse-series data. |
| [halfstrip_core/dense_backend.py](../bttb_solvers/_internal/halfstrip_core/dense_backend.py) | Shared | Context-local classical/Strassen selection and ledgers; actual multiplication recursion is in uniform_terminal.py. |
| [halfstrip_core/exact.py](../bttb_solvers/_internal/halfstrip_core/exact.py) | Shared | Native guarded field operations, elementary elimination, scalar polynomials and sparse Laurent-polynomial matrix helpers. |
| [halfstrip_core/fast_linear.py](../bttb_solvers/_internal/halfstrip_core/fast_linear.py) | Shared | Fast local linear-algebra helpers used by carrier/defect calculations. |
| [halfstrip_core/field_setup.py](../bttb_solvers/_internal/halfstrip_core/field_setup.py) | Shared | Acquire/restrict an effective working algebra, embed original coefficients, supply labels and descend results. |
| [halfstrip_core/flint_backend.py](../bttb_solvers/_internal/halfstrip_core/flint_backend.py) | Shared | Context-local exact native polynomial, dot-product and update kernels; no whole-solver selection. |
| [halfstrip_core/joint.py](../bttb_solvers/_internal/halfstrip_core/joint.py) | Main/shared | Compile the small defect, degree shifts and W/B/G reconstruction dictionaries for main HSC/HPR. |
| [halfstrip_core/mosaic.py](../bttb_solvers/_internal/halfstrip_core/mosaic.py) | Main/shared | Build/check mosaic Toeplitz terminal approximant problems and expose determinant/affine reconstruction data. |
| [halfstrip_core/poly.py](../bttb_solvers/_internal/halfstrip_core/poly.py) | Shared | Sparse/dense polynomial-matrix products, coefficient windows and series inversion with exact scalar backends. |
| [halfstrip_core/reuse.py](../bttb_solvers/_internal/halfstrip_core/reuse.py) | Main/shared | HPR preparation, restart ownership and LU/division/appended queries; also HSC oneshot_solve and deterministic fixture helpers. |
| [halfstrip_core/solver.py](../bttb_solvers/_internal/halfstrip_core/solver.py) | Main/shared | Main HSC carrier-to-terminal orchestration, determinant scaling, physical reconstruction and working-algebra descent. |
| [halfstrip_core/uniform_terminal.py](../bttb_solvers/_internal/halfstrip_core/uniform_terminal.py) | Shared | Constant/polynomial matrix multiplication, explicit Strassen recursion, elimination leaves and recursive PM-Basis primitives. |

