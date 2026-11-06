# Implementation navigation

Start with [the code guide](../../docs/CODE_GUIDE.md) for the mathematical role
of each part. The [complete file reference](../../docs/FILE_REFERENCE.md) locates
individual files. Every substantive module also begins with its role and
responsibility.

- `cc_prepared.py` and `banded_bttb/`: CC/CC-R and their Toeplitz/symbol/correction primitives.
- `extension_field.py`, `extension_kernels.py`: supplied-extension CC variants.
- `boundary_front/`: BFM setup, local fronts, query batching and explicit Strassen products.
- `guarded_boundary/`: GBE, the boundary-elimination comparator.
- `hir_prepared.py`, `halfstrip/solver.py`: HIR leading inverse, causal recurrence and terminal solve.
- `halfstrip_core/`: main HSC/HPR carrier, defect, mosaic terminal, query reuse and shared exact algebra.
- `dense_lu.py`: the physical dense-LU reference and native ownership.
- Remaining physical/source/generator modules: [construction trials](../../REJECTED_ROUTES.md),
  with individually documented roles in the code guide.
- `ck_poly.py` and `kn_*` identify published constructions; their full sources and
  implementation scope are in [REFERENCES](../../REFERENCES.md).

Public users should call [api.py](../api.py). Researchers can inspect these modules
without needing an external research repository or historical route identifiers.
