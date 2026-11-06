# Recorded numerical values and article table templates

`paper_numbers.csv` contains all 565 numeric keys printed in the manuscript,
with their displayed values, units and measured/derived classification. Keys
use construction names consistently, including separate CC and CC-R entries.
They are not executable method selectors.

The ten templates, in article order, are:

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

The templates use `\papernum{key}`. `../tables.py --reference` substitutes the
CSV values and writes compilable TeX. References to theorem/section numbers
are those of the accompanying manuscript. Generic host labels retain the
separation between measurement series; they are not configuration requirements.

Use `../run.py --small` or `--large` to execute solvers and obtain fresh timing
and memory numbers. Formatting these recorded values is a separate operation.
