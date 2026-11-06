# Trials, non-default routes and outcomes

This file answers **why code exists without being a recommended default**.
“Not selected” is scoped to a particular implementation, workload or experiment.
It does not reject a mathematical algorithm family. Shared primitives are listed
separately because they remain useful even when a solver using them loses a
finite timing comparison.

The [solver guide](docs/SOLVERS.md) describes main interfaces. The
[code guide](docs/CODE_GUIDE.md) locates every implementation component. Recorded
values below are available in [the paper-value CSV](reproduce/reference/paper_numbers.csv);
new runs produce their own measurements.

## Usable solvers retained as comparators

| Route/configuration | What was tried | Outcome in the paper | Why it remains |
|---|---|---|---|
| GBE with compiled replacement kernels | Swap the interpreted boundary comparator's elementary operations for FLINT kernels. | Most sampled pairs were slower; the compiled/interpreted median ratio was 3.996 over 33 successful pairs, with separate timeouts. This compiled swap was not selected as the comparator default. | Exact comparison and a check of the effect of kernel overhead. `prepare_gbe` defaults to `backend="python"`. |
| BFM-Strassen with FLINT products | Replace BFM's local classical products by explicit Strassen recursion. | The recorded Strassen/classical ratio had median 3.042 over 39 matched cells, within the tested orders up to 64. It was not the compiled default there. | Actual Strassen code and a useful multiplication experiment. The Python-kernel comparison had median ratio 0.879, so the outcome depends on the backend. |
| HPR with dense terminal LU | Retain the half-strip preparation and terminal factors for repeated queries, including failed-H0 inputs. | No measured CPU win against a reusable dense LU in the recorded range; a memory advantage remains on named cases. | Reuse, memory experiments and a reference for terminal-query alternatives. This is the public HPR default, not a claimed general fast prepared-query theorem. |

These routes still return exact answers on their supported inputs. Re-run the
fixed cases to test a new machine or workload; the recorded rankings are not an
automatic dispatch rule.

## Construction and terminal-query trials

| Reproduction label | Implementation entry | What it investigates | Current interpretation |
|---|---|---|---|
| `Structured-halfstrip` | `full_halfstrip.py::solve_full` | A source/terminal reconstruction with explicit carrier, joint histories and structured readout. | A bounded prototype connected to the exponent discussion, much slower than the dense reference on the recorded small controls. It does not implement every fast primitive in the written subcubic construction. |
| `Dedicated-classical` | `dedicated_banded_arithmetic.py::solve_dedicated_banded_many` | Paid acquisition of displacement generators, supplied-generator services and shared classical preparation for multiple pairs. | A construction control. Finite timings establish no practical superiority. |
| `Dedicated-Strassen` | `dedicated_banded_arithmetic.py::solve_dedicated_banded` | The same dedicated physical construction with explicit Strassen products for one pair; construction repeats per pair. | The dedicated Strassen column in the paper, not BFM-Strassen and not a reusable many-pair API. The test suite checks actual recursion. |
| `Structured-CC` | `hb_physical_solver.py::solve_hb` | A generator-based supplied-root front end and terminal solve for the banded problem. | A slower bounded comparison to the concrete CC implementation; it is not an alias for CC or CC-R. |
| `HPR-division` | `halfstrip_core/reuse.py::_Core._prepare_division` and division query | Retain a root approximant basis and a truncated reversed-basis inverse, then answer a RHS by polynomial division. | A separate query-method trial with runtime certificate checks. Its general correspondence/cost obligations remain open. A certificate failure is reported, not replaced silently by LU. |
| `HPR-appended` | `halfstrip_core/reuse.py`, `terminal_query="appended"` | Reuse front-end dictionaries but solve a new affine approximant problem for each RHS. | A comparison to isolate front-end reuse from the division/LU terminal-query choices. |
| `Uniform-halfstrip`, `CK-halfstrip` and the two mixed variants | `halfstrip/allfield_solver.py`, with explicit front-end/terminal multiplier arguments | Change scalar polynomial multiplication while retaining the surrounding inverse-series/approximant construction. | Backend controls, rather than new recommended solver names. A scalar backend's correctness or bound does not establish a complete physical solver's fast-exponent claim. |

These trials are executed only where listed in `reproduce/cases.json`. The three
smallest cases deliberately include several small controls; `--large` contains
the full retained grid. No separate scheduler or historical run directory is
needed to play with them.

## CK and KN are useful building blocks, not rejected route IDs

**CK** means **Cantor–Kaltofen**. `halfstrip/ck_poly.py` implements a
cyclotomic-quotient polynomial multiplier. Its job is to multiply coefficient
vectors over an exact field without searching that field for transform roots.
It is retained as an explicit alternative in the multiplier controls. The
uniform multiplier in `allfield_poly.py` is also used by main-path polynomial
operations. See [Cantor–Kaltofen 1991](REFERENCES.md#cantor-and-kaltofen-1991).

**KN** means **Khichane–Neiger**. The `kn_*` files contain polynomial kernel,
approximant, modular-vector and supplied-displacement-generator services used
by the dedicated construction controls. They also expose lower-level algebra
that can be reused independently. The prefix identifies the source construction,
not a discarded solver number. Read [Khichane–Neiger 2026](REFERENCES.md#khichane-and-neiger-2026)
and the [module-by-module explanation](docs/CODE_GUIDE.md#ck-and-kn-what-these-prefixes-mean).
The preprint version and its premises remain explicit; the package does not
claim a blanket implementation of all its theorem-level algorithms.

## Written routes without an executable implementation

The paper's general fast-exponent equality-test trees have no executable route
here. The fast-exponent half-strip theorem is also distinct from the bounded
compiled `Structured-halfstrip` control: timing that control does not measure
an implementation of all the theorem's fast primitives.

Integer lifting and unrelated exploratory methods are outside this paper and
are not included in this repository. This file records the roles and limits of
the retained code; it does not attempt to preserve an unrelated research archive.
