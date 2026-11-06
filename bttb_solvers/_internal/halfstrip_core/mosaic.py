# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip.approximant import pm_basis, shifted_row_degrees, polynomial_matrix_product
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier


def _zeros(F, rows, cols):
    return [[F.zero() for _ in range(cols)] for _ in range(rows)]


def _require_zero(F, value, message):
    if not F.is_zero(value):
        raise AssertionError(message)


def _check_basis(F, coefficients, order, shift, result, mul,
                 product=polynomial_matrix_product):
    """Charged field guards also cover all validation coefficient decisions."""
    degrees, pivots, profile = shifted_row_degrees(F, result['basis'], shift)
    if degrees != result['row_degree'] or profile != result['pivot_degree']:
        raise AssertionError('PM-Basis degree/profile trace mismatch')
    if pivots != list(range(len(shift))):
        raise AssertionError('basis is not ordered weak shifted Popov')
    for i, (p, d) in enumerate(zip(pivots, profile)):
        _require_zero(F, F.sub(result['basis'][d][i][p], F.one()),
                      'basis pivot is not monic')
    residual = product(F, result['basis'], coefficients, mul, limit=order)
    for k, C in enumerate(residual):
        for row in C:
            for value in row:
                _require_zero(F, value, 'PM-Basis approximant residual')
    return {'monic_ordered_weak_popov': True, 'kernel_residual_zero': True,
            'actual_degrees': degrees, 'actual_pivots': pivots,
            'actual_profile': profile,
            'bounded_dimension': sum(-d for d in degrees if d < 0),
            'negative_rows': [i for i, d in enumerate(degrees) if d < 0]}


def _chunks(lengths, bound):
    out = []
    global_start = 0
    for chain, length in enumerate(lengths):
        for offset in range(0, length, bound):
            out.append({'chain': chain, 'offset': offset,
                        'length': min(bound, length-offset),
                        'start': global_start+offset})
        global_start += length
    return out


def _entry_locator(lengths):
    return [(i, a) for i, length in enumerate(lengths) for a in range(length)]


def _compact_ledger(node):

    
    keep = {'window_determinant': node['window_determinant'],
            'all_onto': node['all_onto'], 'order': node['order']}
    node.clear()
    node.update(keep)


def run_mosaic(F, chain_lengths, coefficient, rhs=None, dictionary=None,
               multiplier=None, verify=False, *, backend='uniform_fast',
               column_lengths=None):
    """Compute det/rank and optional affine solve from a literal lag accessor.

    coefficient(i,j,k) supplies M[(i,a),(j,b)]=coefficient(i,j,a-b).
    It must support exactly -(column_lengths[j]-1)<=k<=chain_lengths[i]-1,
    with column_lengths defaulting to chain_lengths. Supplied dictionary
    origins are labels: the accessor uses bounded relative lag indices only.
    Zero-length
    channels must already be omitted. Scalars and all coefficient decisions
    use the supplied guarded field adapter. Optional dictionary is provenance.
    """
    engine, product, stats = _select_backend(backend)
    lengths = list(chain_lengths)
    if not lengths or any(type(x) is not int or x <= 0 for x in lengths):
        raise ValueError('positive nonempty actual chain lengths required')
    columns = list(lengths if column_lengths is None else column_lengths)
    if not columns or any(type(x) is not int or x <= 0 for x in columns):
        raise ValueError('positive nonempty actual column lengths required')
    r, m = max(len(lengths), len(columns)), sum(lengths)
    if sum(columns) != m:
        raise ValueError('row and column lengths must have equal totals')
    bound = (m+r-1)//r
    chunks = _chunks(lengths, bound)
    column_chunks = _chunks(columns, bound)
    A, B = len(chunks), len(column_chunks)
    rho = [c['length'] for c in chunks]
    gamma = [c['length'] for c in column_chunks]
    orders = [bound-1+x for x in rho]
    D = max(orders)
    offsets = [D-x for x in orders]
    delta = gamma+[bound-1]*A
    components = B+A

    blocks = [[[coefficient(i,j,k) for k in range(1-columns[j], lengths[i])]
               for j in range(len(columns))] for i in range(len(lengths))]
    def value(i,j,k):
        if 1-columns[j] <= k < lengths[i]:
            return blocks[i][j][k+columns[j]-1]
        return F.zero()
    P = []
    for row in chunks:
        P.append([[value(row['chain'], col['chain'],
                         row['offset']-col['offset']+ell-(bound-1))
                   for ell in range(2*bound-1)] for col in column_chunks])

    coefficients = [_zeros(F, components, A) for _ in range(D)]
    for i in range(A):
        e = offsets[i]
        for j in range(B):
            for ell, x in enumerate(P[i][j]):
                if e+ell < D:
                    coefficients[e+ell][j][i] = x
        coefficients[e][B+i][i] = F.neg(F.one())
    mul = multiplier if multiplier is not None else _make_backend_multiplier(F, backend)
    shift = [-x for x in delta]
    basis = engine(F, coefficients, D, shift, mul, offsets=offsets,
                     ledger_hook=_compact_ledger, skip_zeros=True)
    checks = (_check_basis(F, coefficients, D, shift, basis, mul, product) if verify else
              {'bounded_dimension': sum(-d for d in basis['row_degree'] if d < 0),
               'negative_rows': [i for i, d in enumerate(basis['row_degree']) if d < 0]})
    if sum(basis['pivot_degree']) != sum(delta):
        raise AssertionError('onto effective-window index mismatch')
    matches = basis['pivot_degree'] == delta
    nullity = checks['bounded_dimension']
    if matches != (nullity == 0):
        raise AssertionError('bounded kernel/profile criterion mismatch')
    exponent = (bound-1)*sum((i+1)*x for i,x in enumerate(rho))+A*(bound-1)
    window_det = basis['ledger']['window_determinant'] if matches else F.zero()
    determinant = F.neg(window_det) if exponent%2 else window_det
    if matches and not basis['ledger']['all_onto']:
        raise AssertionError('target profile lacks onto leaf ledger')
    out = {'algorithm': 'compact ragged coefficient-window PM-Basis',
           'det': determinant, 'rank': m-nullity, 'nullity': nullity,
           'solution': None, 'chain_lengths': lengths, 'order': m, 'r': r,
           'chunk_bound': bound, 'chunks': chunks, 'column_chunks': column_chunks,
           'column_lengths': columns, 'rho': rho, 'gamma': gamma,
           'effective_orders': orders, 'approximant_order': D,
           'offsets': offsets, 'delta': delta, 'P': P,
           'coefficients': coefficients, 'basis': basis, 'basis_checks': checks,
           'profile_matches': matches, 'target_window_det': window_det,
           'output_slack_sign_exponent': exponent,
           'dictionary': dictionary, 'coefficient_blocks': blocks,
           'pm_basis_observed': True,
           'terminal_backend': backend, 'terminal_stats': stats,
           'cost_claim': _cost_claim(backend)}
    if rhs is not None:
        affine = solve_mosaic(F, out, rhs, multiplier=mul, verify=verify,
                              terminal_backend=backend)
        out['affine'] = affine
        out['solution'] = affine['solution']
    out['multiplier_stats'] = dict(getattr(mul, 'stats', {}))
    return out


def solve_mosaic(F, backend, rhs, multiplier=None, verify=False, *,
                 terminal_backend=None):
    """Append the bounded affine RHS component and read an actual basis row."""
    selected_backend = (backend.get('terminal_backend', 'classical')
                        if terminal_backend is None else terminal_backend)
    engine, product, stats = _select_backend(selected_backend)
    rhs = list(rhs)
    if len(rhs) != backend['order']:
        raise ValueError('one RHS entry per actual terminal coordinate required')
    mul = (multiplier if multiplier is not None else
           _make_backend_multiplier(F, selected_backend))
    D = backend['approximant_order']
    coefficients = [[list(row) for row in C]+[[F.zero() for _ in backend['rho']]]
                    for C in backend['coefficients']]
    bpolys = []
    for i, chunk in enumerate(backend['chunks']):
        polynomial = [F.zero()]*(backend['chunk_bound']-1)
        polynomial += rhs[chunk['start']:chunk['start']+chunk['length']]
        bpolys.append(polynomial)
        for k,x in enumerate(polynomial):
            coefficients[backend['offsets'][i]+k][-1][i] = F.neg(x)
    delta = backend['delta']+[1]
    shift = [-x for x in delta]
    result = engine(F, coefficients, D, shift, mul,
                      offsets=backend['offsets'], ledger_hook=_compact_ledger, skip_zeros=True)
    checks = (_check_basis(F, coefficients, D, shift, result, mul, product) if verify else
              {'bounded_dimension': sum(-d for d in result['row_degree'] if d < 0),
               'negative_rows': [i for i, d in enumerate(result['row_degree']) if d < 0]})
    selected = None
    for i in checks['negative_rows']:

        c = result['basis'][0][i][-1]
        if not F.is_zero(c):
            if result['row_degree'][i] != -1:
                raise AssertionError('affine c row must have shifted degree -1')
            selected = i
            break
    solution = None
    denominator = None
    if selected is not None:
        denominator = result['basis'][0][selected][-1]
        inverse = F.inv(denominator)
        solution = []
        for j, length in enumerate(backend['gamma']):
            for k in range(length):
                x = (result['basis'][k][selected][j]
                     if k < len(result['basis']) else F.zero())
                solution.append(F.mul(x,inverse))
    if backend['profile_matches'] and selected is None:
        raise AssertionError('nonsingular affine module has no nonzero c row')
    return {'algorithm': 'appended RHS affine PM-Basis', 'rhs': rhs,
            'rhs_polynomials': bpolys, 'coefficients': coefficients,
            'delta': delta, 'basis': result, 'basis_checks': checks,
            'selected_row': selected, 'denominator': denominator,
            'consistent': selected is not None, 'solution': solution,
            'pm_basis_observed': True, 'terminal_backend': selected_backend,
            'terminal_stats': stats, 'cost_claim': _cost_claim(selected_backend)}


def _select_backend(name):
    if name == 'classical':
        return pm_basis, polynomial_matrix_product, {}
    if name != 'uniform_fast':
        raise ValueError('terminal backend must be classical or uniform_fast')
    from . import uniform_terminal
    stats = {}
    def engine(*args, **kwargs):
        return uniform_terminal.pm_basis(*args, stats=stats, **kwargs)
    def product(*args, **kwargs):
        return uniform_terminal.polynomial_matrix_product(*args, stats=stats, **kwargs)
    return engine, product, stats


def _make_backend_multiplier(F, name):
    if name == 'uniform_fast':
        from .uniform_terminal import make_multiplier as make_uniform_multiplier
        return make_uniform_multiplier(F)
    return make_multiplier(F)


def _cost_claim(name):
    if name == 'uniform_fast':
        return ('soft O(r^(log2(7)-1)*m) arithmetic/tests/copies/control and '
                'conservative retained coefficients, conditional on reference or '
                'an equivalent supplied multiplier; supplied dictionary only')
    return 'O(r^3 M(ceil(m/r)) log(m/r+1)); O(r*m) retained coefficients'


