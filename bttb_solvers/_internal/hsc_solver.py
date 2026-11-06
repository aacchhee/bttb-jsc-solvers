# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.exact import Restart
from bttb_solvers._internal.halfstrip_core.solver import solve as _solve_original
from bttb_solvers._internal.label_algebra import setup_label_algebra
from bttb_solvers._internal.full_halfstrip import _input, _retained_factor


class _ClassicalField:
    """Attempt-local legacy compatibility, exposing no caller optional hooks.

    Deliberately not EtaleField: only the owning context removes its added
    algebra, after complete lower-source/readout execution. The mutable dict
    holds halfstrip_core.poly's cache without mutating a caller's slotted object.
    """
    def __init__(self, field):
        for name in ('zero', 'one', 'add', 'sub', 'neg', 'mul', 'inv', 'is_zero'):
            setattr(self, name, getattr(field, name))

    def eq(self, a, b):
        return self.is_zero(self.sub(a, b))

    def info(self, include_inversions=True):
        return {'kind': 'owned_classical_compatibility',
                'protocol': 'eight guarded bound operations; owned cache/eq/info'}


def _attempt(base, flat, row_count, n, w, has_rhs, restriction, precision_mode, precision_probe):
    context = setup_label_algebra(base, flat, n, restriction=restriction)
    width, lag_count = 2*n-1, row_count*(2*n-1)
    field = _ClassicalField(context.field)
    stage = 'original_classical_coefficient_blocks'
    try:

        rows = [context.coefficients[d*width:(d+1)*width] for d in range(row_count)]
        blocks = [[[rows[d][n-1+i-j] for j in range(n)] for i in range(n)]
                  for d in range(row_count)]
        rhs = list(context.coefficients[lag_count:]) if has_rhs else None
        stage = 'original_classical_source_terminal_readout'
        prepared = _solve_original(field, blocks, n, w, rhs, context.labels,
                                   fixture_cases=False, terminal_backend='classical',
                                   precision_mode=precision_mode,precision_probe=precision_probe)
        stage = 'classical_terminal_metadata'
        terminal = prepared.get('terminal')
        terminal_calls = {'determinant': 0, 'affine': 0}
        if terminal is not None:
            if terminal['backend']['terminal_backend'] != 'classical':
                raise ArithmeticError('classical determinant engine was not selected')
            terminal_calls['determinant'] = 1
            for record in prepared['rhs']:
                if record['affine']['terminal_backend'] != 'classical':
                    raise ArithmeticError('classical affine engine was not selected')
                terminal_calls['affine'] += 1
        stage = 'classical_physical_descent'
        determinant = context.descend(prepared['det'])
        solution = prepared['solution']
        physical = None if solution is None else [context.descend(value) for value in solution]
        joint = prepared['joint']
        return {'det': determinant, 'singular': prepared['singular'],
                'physical_solution': physical, 'n': n, 'w': w,
                'q': prepared['carrier']['q'], 'overcap': joint['overcap'],
                'delta': tuple(joint.get('delta', ())),
                'label_setup': context.metadata,
                'precision':prepared['precision'],'precision_probe':prepared['precision_probe'],
                'metadata': {'method': 'HSC',
                             'source': 'halfstrip_core original carrier/joint dense coefficient histories',
                             'source_precision': prepared['precision']['source_precision'],
                             'terminal_backend': 'classical', 'terminal_calls': terminal_calls,
                             'terminal_order': None if terminal is None else terminal['backend']['order'],
                             'dense_physical_matrix': False,
                             'cost_scope': 'soft O(w*n^3) original-field work; conservative same peak'}}, None
    except Restart as event:
        factor = _retained_factor(context, event)

        
        return None, {'old_degree': context.field.degree,
                      'new_degree': len(factor)-1, 'factor': factor,
                      'guard_index': event.guard_index, 'stage': stage,
                      'initial_degree': context.metadata['initial_extension_degree']}


def solve_hsc(base, lags, n, w, rhs=None, *, precision_mode='full',precision_probe=False):
    """Raw native lags -> every-rank determinant and one nonsingular RHS.

    lags[d][n-1+i-j] represents F_d[i,j]; higher coefficients are zero.
    Only an algebra added by this invocation is descended. No supplied
    computational source, labels, field extension or backend hook is accepted.
    """
    flat, row_count, has_rhs = _input(lags, n, w, rhs)
    restriction, restarts, retained_degree = None, [], None
    attempts = 0
    while True:
        attempts += 1
        result, receipt = _attempt(base, flat, row_count, n, w, has_rhs, restriction,precision_mode,precision_probe)
        if receipt is None:
            result.update(attempts=attempts, restarts=restarts)
            return result
        old, new = receipt['old_degree'], receipt['new_degree']
        if not 1 <= new < old or (retained_degree is not None and old != retained_degree):
            raise ArithmeticError('global classical restriction did not strictly lower current degree')
        if len(restarts)+1 > receipt['initial_degree']-1:
            raise ArithmeticError('global classical restriction exceeded degree-drop bound')
        restriction, retained_degree = receipt['factor'], new
        restarts.append(dict(receipt, attempt=attempts,
            invalidated='all context/coefficient blocks/carrier/joint/terminal/RHS/descent state'))
