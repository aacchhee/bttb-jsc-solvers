# Validation

Checked on Linux x86-64 with python-flint 0.9.0 / FLINT 3.6.0:

- A fresh virtual environment installs the package with `pip install -e .`.
- All nine solver tests pass, including independent dense-reference comparisons,
  singular-symbol repair, singular-leading-block handling, extension fields,
  literal transpose solves and single-factorization reuse.
- Both explicit Strassen routes produce correct answers and report nonzero
  recursive-node counts. This checks actual recursion, not just a leaf fallback.
- The three-case `--small` run completes all 76 observations successfully, with
  exact residual checks and input/output agreement across algorithms and visits.
- Additional small dispatcher checks exercise the remaining named routes,
  including both HPR terminal-query comparisons and the extension/Strassen routes.
- Recorded-table generation resolves all 565 printed keys; the ten-table standalone
  TeX document compiles successfully.
- The user-input solve/comparison examples and the Strassen, polynomial and
  supplied-generator examples pass their exact checks. All 107 substantive
  internal modules have documented roles and entries in the detailed file reference.
- The `--large --list` plan contains 158 cases and 2,420 sequential observations.
  The full large run has not been executed during this packaging check.

Run the checks yourself from the repository root:

```bash
python -m unittest discover -s tests -v
python reproduce/run.py --small
python reproduce/tables.py --reference
```

These finite checks do not extend the mathematical guarantees stated in the
paper. Fresh timing and memory values depend on the machine running them.
