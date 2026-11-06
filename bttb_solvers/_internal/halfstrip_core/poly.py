# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier, NATIVE_ACCELERATOR_MAX
from .flint_backend import enabled_prime, product as flint_product, selection as kernel_selection, record as kernel_record


def _state(F):
    """Keep one multiplier on the field; no field identity/global cache leak."""
    state = getattr(F, '_h0_poly_state', None)
    if state is None:

        
        multiplier = make_multiplier(F)
        state = {'multiplier': multiplier, 'products': 0, 'sparse_products': 0,
                 'sparse_pairs': 0, 'dense_products': 0, 'input_cells': 0,
                 'largest_span': 0, 'newton_steps': 0}
        setattr(F, '_h0_poly_state', state)
    return state


def stats(F):
    """Structural counters; actual field operations belong to the field adapter."""
    state = getattr(F, '_h0_poly_state', None)
    if state is None:
        return {'products': 0, 'sparse_products': 0, 'sparse_pairs': 0,
                'dense_products': 0, 'input_cells': 0, 'largest_span': 0,
                'newton_steps': 0, 'multiplier': {}}
    return {**{k: v for k, v in state.items() if k != 'multiplier'},
            'multiplier': dict(state['multiplier'].stats)}


def reset_stats(F):
    """Reset counters, retaining the already guarded multiplier constants."""
    state = getattr(F, '_h0_poly_state', None)
    if state is None:
        return
    for key in state:
        if key != 'multiplier':
            state[key] = 0
    for key in state['multiplier'].stats:
        if key != 'radix':
            state['multiplier'].stats[key] = 0


def clean(F, a, lo=None, hi=None):
    return {k: v for k, v in a.items()
            if (lo is None or k >= lo) and (hi is None or k < hi)
            and not F.is_zero(v)}


def sa(F, a, b, lo=None, hi=None):
    out = {k: v for k, v in a.items()
           if (lo is None or k >= lo) and (hi is None or k < hi)}
    for k, value in b.items():
        if (lo is None or k >= lo) and (hi is None or k < hi):
            out[k] = F.add(out.get(k, F.zero()), value)
    return clean(F, out)


def sn(F, a):
    return {k: F.neg(v) for k, v in a.items()}


def ss(F, a, b, lo=None, hi=None):
    return sa(F, a, sn(F, b), lo, hi)


def sm(F, a, b, lo=None, hi=None):
    """Return a*b restricted to lo <= exponent < hi, including Laurent inputs.

    Input trimming uses the other factor's *original* endpoint extrema. It
    discards only terms incapable of contributing to the requested window;
    truncation does not assume that either factor has nonnegative valuation.
    """
    if not a or not b or (lo is not None and hi is not None and hi <= lo):
        return {}
    amin, amax, bmin, bmax = min(a), max(a), min(b), max(b)
    a = clean(F, a, None if lo is None else lo-bmax,
              None if hi is None else hi-bmin)
    b = clean(F, b, None if lo is None else lo-amax,
              None if hi is None else hi-amin)
    if not a or not b:
        return {}
    amin, amax, bmin, bmax = min(a), max(a), min(b), max(b)
    na, nb = amax-amin+1, bmax-bmin+1
    if (lo is not None and amax+bmax < lo) or (hi is not None and amin+bmin >= hi):
        return {}
    state = _state(F)
    state['products'] += 1
    state['input_cells'] += na+nb
    state['largest_span'] = max(state['largest_span'], na, nb)
    pairs = len(a)*len(b)
    if enabled_prime(F) is not None:
        state['dense_products'] += 1
        aa, bb = [0]*na, [0]*nb
        for k, value in a.items():
            aa[k-amin] = value
        for k, value in b.items():
            bb[k-bmin] = value
        result = flint_product(F, aa, bb)
        origin = amin+bmin
        return {origin+k: value for k, value in enumerate(result)
                if (lo is None or origin+k >= lo) and (hi is None or origin+k < hi)
                and not F.is_zero(value)}
    if kernel_selection() == 'flint':
        kernel_record('poly',False)
    packed_leaf = (hasattr(F, 'polynomial_native') and max(na, nb) <= NATIVE_ACCELERATOR_MAX
                   and pairs >= 64 and pairs > na+nb)
    if not packed_leaf and (pairs <= 4096 or pairs <= na+nb):
        state['sparse_products'] += 1
        state['sparse_pairs'] += pairs
        out = {}
        add, mul = F.add, F.mul
        for i, x in a.items():
            for j, y in b.items():
                k = i+j
                if (lo is None or k >= lo) and (hi is None or k < hi):
                    product = mul(x, y)

                    

                    out[k] = add(out[k], product) if k in out else product
        return clean(F, out)
    state['dense_products'] += 1
    aa, bb = [F.zero()]*na, [F.zero()]*nb
    for k, value in a.items():
        aa[k-amin] = value
    for k, value in b.items():
        bb[k-bmin] = value
    result = state['multiplier'](aa, bb)
    origin = amin+bmin
    start = max(0, lo-origin) if lo is not None else 0
    stop = min(len(result), hi-origin) if hi is not None else len(result)
    return {origin+k: result[k] for k in range(start, stop)
            if not F.is_zero(result[k])}


def scale(F, a, c):
    return clean(F, {k: F.mul(v, c) for k, v in a.items()})


def shift(a, k):
    return {i+k: v for i, v in a.items()}


def sinv(F, a, precision):
    """Inverse modulo x**precision of an ordinary series with a unit constant.

    F.inv is the only constant inversion route, preserving etale restriction
    exceptions. Negative exponents must first be removed by valuation shift.
    Newton b <- b + b*(1-a*b) works also in characteristic two.
    """
    if type(precision) is not int or precision < 0:
        raise ValueError('nonnegative integer precision required')
    if any(k < 0 for k in a):
        raise ValueError('shift Laurent input before ordinary series inversion')
    if precision == 0:
        return {}
    out = {0: F.inv(a.get(0, F.zero()))}
    order = 1
    while order < precision:
        target = min(2*order, precision)
        ab = sm(F, a, out, 0, target)
        error = ss(F, {0: F.one()}, ab, 0, target)

        
        error = clean(F, error, order, target)
        out = sa(F, out, sm(F, out, error, 0, target), 0, target)
        _state(F)['newton_steps'] += 1
        order = target
    return out


def trim(F, a):
    out = list(a)
    while out and F.is_zero(out[-1]):
        out.pop()
    return out


def pm(F, a, b):
    """Dense-list product: default all-field primitive or explicit FLINT scope."""
    if not a or not b:
        return []
    state = _state(F)
    state['products'] += 1
    state['dense_products'] += 1
    state['input_cells'] += len(a)+len(b)
    state['largest_span'] = max(state['largest_span'], len(a), len(b))
    result = flint_product(F, a, b)
    return trim(F, state['multiplier'](list(a), list(b)) if result is None else result)


def pzeros(m, n):
    return [[{} for _ in range(n)] for _ in range(m)]


def peye(F, n):
    out = pzeros(n, n)
    for i in range(n):
        out[i][i] = {0: F.one()}
    return out


def pmat(F, a):
    return [[({0: x} if not F.is_zero(x) else {}) for x in row] for row in a]


def pmm(F, a, b, lo=None, hi=None):
    if not a:
        return []
    n = len(b[0]) if b else 0
    if any(len(row) != len(b) for row in a) or any(len(row) != n for row in b):
        raise ValueError('incompatible rectangular polynomial matrices')
    out = pzeros(len(a), n)
    for i, row in enumerate(a):
        for k, x in enumerate(row):
            if x:
                for j, y in enumerate(b[k]):
                    if y:
                        out[i][j] = sa(F, out[i][j], sm(F, x, y, lo, hi), lo, hi)
    return out


def pcut(F, a, lo=None, hi=None):
    return [[clean(F, x, lo, hi) for x in row] for row in a]


def coefficient(F, a, k):
    return [[x.get(k, F.zero()) for x in row] for row in a]
