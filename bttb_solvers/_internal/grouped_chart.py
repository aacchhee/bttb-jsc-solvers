# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.banded_bttb.polynomial import ProductTree, trim
from bttb_solvers._internal.fast_scalar import FastScalar
from bttb_solvers._internal.label_algebra import setup_banded_label_algebra


class ChartSearchExhausted(ArithmeticError):
    """The supplied nonzero/field/catalog contract failed to yield a chart."""


class _GroupedChartSearch:
    def __init__(self, context):
        self.context = context
        self.field = context.field
        self.n = context.metadata['n']
        self.k = context.metadata['k']
        self.D = context.metadata['D']
        self.M = context.metadata['M']
        self.arithmetic = FastScalar(self.field)
        self._tree = None

    def select(self, polynomial):
        """Select first unit chart for f native in this attempt's field.

        The leading coefficient must be a guarded unit, not just nonzero in
        one component. Restart from any guard propagates unchanged.
        """
        F, A = self.field, self.arithmetic
        f = trim(F, list(polynomial))
        if not f:
            raise ValueError('certified nonzero polynomial required')
        if len(f)-1 > self.D:
            raise ValueError('supplied polynomial exceeds degree 2kn')
        F.inv(f[-1])
        groups = [f[r::self.n] for r in range(self.n)]
        if any(len(group) > 2*self.k+1 for group in groups):
            raise ArithmeticError('group-degree bound lost')
        if self._tree is None:
            self._tree = ProductTree(F, self.context.labels, arithmetic=A)

        
        values = [self._tree.evaluate(group) for group in groups]
        tried = []
        for index, label in enumerate(self.context.labels):
            remainder = trim(F, [values[r][index] for r in range(self.n)])
            modulus = [F.neg(label)] + [F.zero() for _ in range(self.n-1)] + [F.one()]
            gcd, u, v = A.xgcd(remainder, modulus)
            tried.append({'candidate_id': self.context.label_ids[index],
                          'remainder_degree': len(remainder)-1,
                          'modulus_degree': self.n, 'gcd_degree': len(gcd)-1})
            if len(gcd) == 1:

                
                if not F.is_zero(F.sub(gcd[0], F.one())):
                    raise ArithmeticError('scalar gcd did not normalize to one')
                F.inv(label)
                return {'status': 'UNIT_CHART', 'field': F, 'context': self.context,
                        'c': label, 'candidate_id': self.context.label_ids[index],
                        'remainder': remainder, 'modulus': modulus,
                        'gcd': gcd, 'bezout_remainder': u, 'bezout_modulus': v,
                        'grouped_values': values,
                        'metadata': {'n': self.n, 'k': self.k, 'D': self.D, 'M': self.M,
                                     'input_degree': len(f)-1,
                                     'group_lengths': [len(group) for group in groups],
                                     'group_input_coefficients': sum(map(len, groups)),
                                     'grouped_evaluation_calls': self.n,
                                     'grouped_output_coefficients': self.n*self.M,
                                     'gcd_calls': len(tried), 'tried': tried,
                                     'source_scope': 'supplied nonzero f; no determinant acquisition',
                                     'witness_scope': 'u*remainder+v*(z^n-c)=1; not an unreduced-f Bezout row'},
                        'scalar_stats': A.stats,
                        'hgcd_stats': A.hgcd_stats,
                        'multiplier_stats': A.uniform_multiply.stats}
        raise ChartSearchExhausted('all internally constructed chart candidates failed')


def prepare_grouped_chart(base, n, k, restriction=None):
    """Prepare catalog and same-field source seam before f is acquired."""
    context = setup_banded_label_algebra(base, [], n, k, restriction)
    return _GroupedChartSearch(context)


def select_cyclic_chart(base, n, k, polynomial, restriction=None):
    """Convenience: one-pass embed caller-native flat f, then search."""
    context = setup_banded_label_algebra(base, polynomial, n, k, restriction)
    return _GroupedChartSearch(context).select(context.coefficients)
