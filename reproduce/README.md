# Reproduce the paper's experiments and figures

Run from the repository root after `python -m pip install -e .`.

```bash
python reproduce/run.py --small
python reproduce/run.py --large
```

The driver is sequential. Each observation runs in a fresh child process; no
workers, cluster configuration, job scheduler or network service is needed.
There is no wall-time limit or automatic dense fallback. `--large` is intended
for someone willing to supply substantial time and memory.

## What the switches mean

`cases.json` contains 158 distinct parameter cases in increasing order of
`(n, width, pairs, family, seed)`. A case includes its fixed list of algorithms,
backends and native polynomial-product caps. Each list runs in forward and
reverse order: two mirrored visits of every method configuration.

`--small` runs exactly the first three cases, for 76 observations:

1. Banded, n=4, width=1, one forward/transpose pair.
2. Banded, n=8, width=1, one pair.
3. Half-strip with invertible outermost block, n=8, width=1, one pair.

`--large` runs all 158 cases, for 2,420 observations. This includes the narrow
and wide band grids, extension-field cases, large scaling and memory cases,
full-rank and rank-deficient outermost blocks, repeated right-hand sides,
Strassen comparisons and bounded exponent-discussion controls. Integer lifting
is excluded, as it is outside this paper.

To inspect the suite without starting it:

```bash
python reproduce/run.py --small --list
python reproduce/run.py --large --list
```

The union of the paper's parameter grids is stored once: identical cases use
a combined method list rather than repeated historical host runs. Native
product caps 512 and 4096 are separate configurations where both were tested.
No host assignment or historical run identifier is needed to reproduce an input.

## Results

Default output is `reproduce/output/small/` or `reproduce/output/large/`.
`--output /some/directory` chooses another location. Existing files there are
replaced, so choose a new directory to preserve an earlier run.

- `measurements.jsonl`: every observation, disjoint timed phases, exact input
  and output hashes, determinant where exposed, residual-check count and errors.
- `summary.csv`: medians of two successful visits, whole-job peak RSS, matching
  service and same-case Dense-LU speedup where that reference was run.
- `derived.csv`: the n=128 CC-R bandwidth interpolation and bracketing sampled
  widths, when a complete crossing is present. It is arithmetic on measurements,
  not a statistical threshold or universal choice rule.
- `measurements.md`: readable table of the fresh summary.
- `environment.json`: Python, platform and FLINT versions, serial execution.

An incomplete pair has no median or speedup. Failures remain in the JSONL and
make the driver return a nonzero exit code. All successful algorithms for a case
must agree on their input and output hashes; exposed determinants must agree too.

Timings use process CPU and elapsed wall clocks around the paid solver phases.
Fixture generation and independent exact residual checks are outside the timers.
BFM and GBE additionally retain their internal classical residual guards inside
the query timers, matching the paper's guarded boundary comparison. These are
distinct from the independent raw-lag checks after every completed solve.
CC-E/CC-E-R additionally supply the extension field, root and embeddings before
timing, and descend results afterwards, matching the paper's stated scope.
Dense uses separate elimination per query batch; Dense-LU pays one native LU and
reuses it for both directions and the determinant. Det+solve routes use the
latter service as denominator; solve-only routes exclude its determinant phase.

RSS is the whole child-process peak, including fixture generation and exact
checks. Different machines, Python versions, process overhead and dependency
builds change measured times and memory. The new values should not be presented
as identical to the recorded paper values.

## Strassen routes and other controls

- `BFM-Strassen`: boundary-front merge with explicit recursive Strassen products,
  leaf size 16, for both reference Python and compiled FLINT kernels.
- `Dedicated-Strassen`: the dedicated one-pair construction with Strassen
  products, leaf size 8, for the dedicated-construction comparison.
- `Dedicated-classical`, `Structured-halfstrip` and `Structured-CC` are the
  other bounded construction controls from the exponent discussion.
- `Uniform-halfstrip` and the `CK-...` variants retain the scalar multiplication
  controls. They do not implement the paper's general equality-test trees.
- `HPR-division` and `HPR-appended` retain the separate terminal-query comparisons;
  `HPR` itself is the timed dense-terminal variant. They are not interchangeable
  algorithm labels.

For the explicit Strassen routes, `metadata.strassen_ledger` records actual
recursive-node counts. FLINT's own internal product selection is not controlled
or inferred from timings. The tests check that recursion actually occurs in both
Strassen constructions and that their answers satisfy exact residuals.

## Recorded figures

The three figures in Section 5 can be drawn without running the solvers:

```bash
python -m pip install matplotlib
python reproduce/plots.py
```

This writes PDF and SVG figures in `reproduce/output/plots/`, together with
`plot_points.json`, which identifies the recorded value behind each point.
The figures show the bandwidth crossing, large-order scaling, and half-strip
comparisons. Lines join observations; they are not fitted performance models.
Matplotlib is optional and is needed only for drawing figures.

## Extended numerical tables

```bash
python reproduce/tables.py --reference
```

This formats thirteen tables (ten extended comparisons and three compact
experimental tables used in the current article) from the supplied templates and the
recorded values in `reference/paper_numbers.csv`. The article presents a shorter
selection of these comparisons, together with the figures above. It writes each table's TeX,
`recorded_tables.tex` (a standalone compilable document) and the value CSV to
`reproduce/output/tables/`. The value records retain `measured` versus `derived`
labels; their keys use the paper's algorithm names. Some values are descriptive
metadata about the original measurements, not timings to rerun.

This mode formats the paper's **recorded** numbers; it does not claim to execute
an experiment or recalculate every historical statistic from raw job logs.
`run.py` executes the actual algorithms on the retained input grids and produces
new measurements and matching-service ratios. Neither mode needs private files.

Optionally compile the recorded tables with a normal LaTeX installation:

```bash
cd reproduce/output/tables
pdflatex recorded_tables.tex
```

To reformat an existing fresh run:

```bash
python reproduce/tables.py --results reproduce/output/small --output reproduce/output/small
```
