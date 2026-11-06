# Recorded numerical values and extended table templates

`paper_numbers.csv` contains the recorded values used by the figures and extended tables,
with their displayed values, units and measured/derived classification. Keys
use construction names consistently, including separate CC and CC-R entries.
They are not executable method selectors.

The ten extended templates retain the detailed comparisons behind the article:

1. CC setup/query cost ledger.
2. Algorithm domains and services.
3. Banded small-order timing grid.
4. Order-128 bandwidth, memory and speedups.
5. Large-order scaling.
6. Method comparisons on common inputs.
7. Extension-field comparisons.
8. Half-strip comparisons.
9. Prepared-query arithmetic bounds.
10. Explicit Strassen control comparisons.

Three additional templates, `current_memory.tex`, `current_extension.tex` and
`current_controls.tex`, give the compact experimental tables used in the current
article. The recorded collection contains 580 values; the plots use 67 of them.

The templates use `\papernum{key}`. `../tables.py --reference` substitutes the
CSV values and writes compilable TeX. The article condenses these comparisons into figures and smaller tables;
section and table numbers in these extended templates need not match the article. Generic host labels retain the
separation between measurement series; they are not configuration requirements.

Use `../run.py --small` or `--large` to execute solvers and obtain fresh timing
and memory numbers. Formatting these recorded values is a separate operation.
