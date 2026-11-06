# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured half-strip comparison.

Role: Mixed. Trial structured half-strip solve with explicit source histories and whole-attempt restart/descent; its input/restriction helpers are also shared by HSC-q0.

Entry points: solve_full; _input; _retained_factor.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.exact import Restart, pd
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata
from bttb_solvers._internal.label_algebra import setup_label_algebra
from bttb_solvers._internal.initial_carrier import InitialCarrier
from bttb_solvers._internal.initial_joint import InitialJoint
from bttb_solvers._internal.full_source_readout import solve_prepared


def _input(lags, n, w, rhs):
    if type(n) is not int or type(w) is not int or n < 2 or not 1 <= w < n:
        raise ValueError('integer n>=2 and 1<=w<n required')
    rows = tuple(tuple(row) for row in lags)
    if not rows or len(rows) > n+w or any(len(row) != 2*n-1 for row in rows):
        raise ValueError('one to n+w raw lag vectors of length 2n-1 required')
    vector = None if rhs is None else tuple(rhs)
    if vector is not None and len(vector) != n*n:
        raise ValueError('RHS must contain n*n native time-major entries')
    flat = tuple(value for row in rows for value in row)
    if vector is not None:
        flat += vector
    return flat, len(rows), vector is not None


def _retained_factor(context, event):
    """Validate the ACTUAL guard factor in the current added algebra."""
    if not context.added_algebra:
        raise event
    field, factor = context.field, list(event.factor)
    base = field.base
    if not 2 <= len(factor) < len(field.modulus):
        raise event
    if not base.is_zero(base.sub(factor[-1], base.one())):
        raise event
    _quotient, remainder = pd(base, field.modulus, factor)
    if any(not base.is_zero(value) for value in remainder):
        raise event
    return tuple(factor)


def _attempt(base, flat, row_count, n, w, has_rhs, restriction):

    
    context = setup_label_algebra(base, flat, n, restriction=restriction)
    width, lag_count = 2*n-1, row_count*(2*n-1)
    rows = [list(context.coefficients[i*width:(i+1)*width]) for i in range(row_count)]
    rhs = list(context.coefficients[lag_count:]) if has_rhs else None
    stage = 'initial_carrier'
    try:
        carrier = InitialCarrier(context.field, rows, n, w, context.labels)
        stage = 'initial_joint'
        initial = InitialJoint(carrier)
        stage = 'source_terminal_recovery'
        prepared = solve_prepared(initial, rhs)
        stage = 'descent'
        answer = {'det': context.descend(prepared['det']),
                  'singular': prepared['singular'], 'physical_solution': None,
                  'n': n, 'w': w, 'q': prepared['q'],
                  'overcap': prepared['overcap'], 'label_setup': context.metadata,
                  'metadata': metadata()}
        if prepared['physical_solution'] is not None:
            answer['physical_solution'] = [context.descend(value)
                                           for value in prepared['physical_solution']]
        if 'delta' in prepared:
            answer['delta'], answer['k'] = tuple(prepared['delta']), prepared['k']
        return answer, None
    except Restart as event:
        factor = _retained_factor(context, event)

        

        return None, {'old_degree': context.field.degree,
                      'new_degree': len(factor)-1, 'factor': factor,
                      'guard_index': event.guard_index, 'stage': stage,
                      'initial_degree': context.metadata['initial_extension_degree']}


def solve_full(base, lags, n, w, rhs=None):
    """Raw native lags -> det(T), and one physical solution if nonsingular.

    lags[d][n-1+i-j] is F_d[i,j], higher temporal coefficients zero; original
    T has n*n coordinates and blocks F_(time-other+w). Physical RHS/solution
    are flat time-major native vectors. Only an algebra ADDED by this call is
    removed on descent; a caller's native extension remains the output field.
    """
    flat, row_count, has_rhs = _input(lags, n, w, rhs)
    restriction, restarts, retained_degree = None, [], None
    attempts = 0
    while True:
        attempts += 1
        answer, restart = _attempt(base, flat, row_count, n, w, has_rhs, restriction)
        if restart is None:
            answer.update(attempts=attempts, restarts=restarts)
            return answer
        old, new, initial_degree = restart['old_degree'], restart['new_degree'], restart['initial_degree']
        if not 1 <= new < old or (retained_degree is not None and old != retained_degree):
            raise ArithmeticError('global restriction did not strictly lower current algebra degree')
        if len(restarts)+1 > initial_degree-1:
            raise ArithmeticError('global restriction exceeded the degree-drop bound')
        restriction, retained_degree = restart['factor'], new
        restarts.append({'attempt': attempts, 'old_degree': old, 'new_degree': new,
                         'factor': restriction, 'guard_index': restart['guard_index'],
                         'stage': restart['stage'],
                         'invalidated': 'all attempt-bound context/carrier/source/histories/terminal/RHS state'})
