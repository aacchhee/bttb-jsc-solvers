# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured CC comparison.

Role: Trial. Compose the structured supplied-root carrier, projected correction and terminal into a one-pair comparison, with an optional prepared classical inner base.

Entry points: solve_hb; prepare_hb.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata as dense_metadata, terminal_method, selection
from bttb_solvers._internal.hb_frontend import prepare_hb_frontend
from bttb_solvers._internal.weighted_carrier import acquire_weighted_carrier
from bttb_solvers._internal.weighted_projections import acquire_weighted_projections
from bttb_solvers._internal.weighted_correction import acquire_weighted_correction
from bttb_solvers._internal.kn_services import Services, METHOD as TERMINAL_METHOD
from bttb_solvers._internal.wide_physical_solver import SingularPhysicalMatrix
from bttb_solvers._internal.hb_prepared_dense import HBPreparedDense


METHOD = 'STRUCTURED_CC_SUPPLIED_ROOT_MU_TRUE_TRANSPOSE_STRASSEN'


def _method():
    backend,leaf=selection()
    if (backend,leaf)==('strassen',1):return METHOD
    if backend=='classical':return 'STRUCTURED_CC_SUPPLIED_ROOT_MU_TRUE_TRANSPOSE_CLASSICAL_DENSE'
    return 'STRUCTURED_CC_SUPPLIED_ROOT_MU_TRUE_TRANSPOSE_STRASSEN_FIXED_LEAF_'+str(leaf)


def _scale(F, vector, n, mu, inverse=False):
    step = F.inv(mu) if inverse else mu
    factor, result = F.one(), []
    for i in range(n):
        result.extend(F.mul(factor, value) for value in vector[i*n:(i+1)*n])
        factor = F.mul(factor, step)
    return result


def _wrap_action(F, correction, multiply, vector, *, transpose=False):
    """U or U^t on ell blocks, n² actual O(k)-length scalar products."""
    n, ell = correction.n, correction.ell
    result = [F.zero() for _ in range(ell*n)]
    for a in range(n):
        for b in range(n):
            kernel = []
            for d in range(1-ell, ell):
                block = correction.u.get(-d if transpose else d)
                kernel.append(F.zero() if block is None else block[b][a] if transpose else block[a][b])
            product = multiply(kernel, [vector[q*n+b] for q in range(ell)])
            for p in range(ell):
                degree = ell-1+p
                if degree < len(product):
                    result[p*n+a] = F.add(result[p*n+a], product[degree])
    return result


def _reduce(F, projected, correction, vector, transpose):
    n, s, ell = correction.n, correction.s, correction.ell
    boundary = [vector[p*n+a] for p in correction.positions for a in range(n)]
    if transpose:
        boundary = _wrap_action(F, correction, projected.carrier.service.mul_scalar,
                                boundary, transpose=True)
    factor = projected.X if transpose else projected.Y
    result = list(boundary)
    for i in range(n):
        for b in range(s):
            value = F.zero()
            for a in range(n):
                value = F.add(value, F.mul(factor[a][b], vector[i*n+a]))
            result.append(value)
    return result


def _lift(F, projected, correction, vector, transpose):
    n, s, ell = correction.n, correction.s, correction.ell
    boundary = vector[:ell*n]
    if not transpose:
        boundary = _wrap_action(F, correction, projected.carrier.service.mul_scalar, boundary)
    result = [F.zero() for _ in range(n*n)]
    for p, position in enumerate(correction.positions):
        for a in range(n): result[position*n+a] = boundary[p*n+a]
    factor = projected.Y if transpose else projected.X
    for i in range(n):
        for a in range(n):
            for b in range(s):
                at = i*n+a
                result[at] = F.add(result[at], F.mul(factor[a][b], vector[ell*n+i*s+b]))
    return result


def _query(F, projected, correction, terminal, mu, rhs, transpose, metadata):
    n = correction.n

    scaled_rhs = _scale(F, rhs, n, mu, inverse=transpose)
    y = projected.physical_action(scaled_rhs, transpose=transpose)
    reduced = _reduce(F, projected, correction, y, transpose)
    pi = correction.permutation
    represented = list(reversed([reduced[i] for i in pi]))
    g, h = (correction.H0, correction.G0) if transpose else (correction.G0, correction.H0)
    answer = terminal.solve_stein(g, h, represented)
    call = 'transpose_swapped_H0_G0' if transpose else 'forward_G0_H0'
    metadata['inner_calls'].append(call)
    if metadata['hb_arm'] == 'current': metadata['kn_calls'].append(call)
    if answer['status'] != 'CONSISTENT':
        raise ArithmeticError('inconsistent KN query after physical nonsingularity gate')
    z = [F.zero() for _ in pi]
    for old, value in zip(pi, reversed(answer['solution'])): z[old] = value
    lifted = _lift(F, projected, correction, z, transpose)
    correction_action = projected.physical_action(lifted, transpose=transpose)
    recovered = [F.add(a,b) for a,b in zip(y, correction_action)]

    return _scale(F, recovered, n, mu, inverse=not transpose)


class PreparedHB:
    """One guarded HB attempt; reuse frontend, carrier and inner preparation.

    Generator and field objects must remain unmodified until invalidate().
    """
    def __init__(self, field, projected, correction, terminal, mu, metadata, stats):
        self.field, self.projected, self.correction = field, projected, correction
        self.terminal, self.mu = terminal, mu
        self.metadata, self.statistics = metadata, stats

    def query(self, rhs, *, transpose=False):
        if len(rhs) != self.correction.n**2:
            raise ValueError('physical RHS must have n*n native entries')
        result = _query(self.field,self.projected,self.correction,self.terminal,
                        self.mu,tuple(rhs),transpose,self.metadata)
        key = 'transpose_calls' if transpose else 'forward_calls'
        self.metadata[key] += 1
        self.metadata['actual_forward_carrier_actions'] = self.projected.stats['physical_forward_actions']
        self.metadata['actual_transpose_carrier_actions'] = self.projected.stats['physical_transpose_actions']
        return result

    def query_many(self, right_hand_sides, *, transpose=False):
        return [self.query(rhs,transpose=transpose) for rhs in right_hand_sides]

    def solve(self, rhs):
        return self.query(rhs)

    def invalidate(self):
        self.terminal.invalidate()
        self.__dict__.clear()
        self.invalidated = True


def prepare_hb(F, n, k, raw, omega, *, arm='current'):
    """Prepare one reusable raw HB attempt with supplied finite field/root.

    Domain: n>=3, 1<=2k<n, n|(q-1), (q-1)/n>2kn, omega of exact order n,
    supplied historical PrimeField/ExtensionField coordinate representation.
    Returns a query object after a homogeneous nonsingularity gate.
    Native guard exceptions propagate without replacement.
    """
    if type(n) is not int or type(k) is not int:
        raise ValueError('actual integer HB dimensions required')
    if arm not in ('current','reuse_inner','retained'):
        raise ValueError('HB arm must be current, reuse_inner, or retained')
    if arm != 'current' and selection()[0] != 'classical':
        raise ValueError('HB prepared dense arms require the classical backend')
    prepared = prepare_hb_frontend(F,n,k,omega,raw)
    field = prepared['field']
    metadata = {'method':_method(), 'terminal_method':terminal_method(),
                'source':'algorithm / algorithm actual mu-scaled transposed Woodbury maps',
                'n':n,'k':k,'physical_order':n*n,'dimension_branch':'2k<n',
                'normal_rank':prepared['normal_rank'], 'kn_calls':[], 'inner_calls':[],
                'nonsingularity_certified':False,'forward_calls':0,'transpose_calls':0,
                'field_construction':False,'coefficient_algebra_added':False,
                'transpose_method':'actual C^-t, Bc^t reduction, swapped H0/G0, A^t lift',
                'physical_persymmetry_used':False}
    metadata.update(dense_metadata())
    metadata.update(hb_arm=arm, multi_rhs_prepared=True,
                    inner_preparation='per_query_KN' if arm=='current' else 'once_per_attempt_PLU',
                    carrier_representation='existing_four_vector_cyclic_upper',
                    retained_representation=False, carrier_setup_retains_kn=True,
                    fully_kn_free=False)
    stats = {'frontend':prepared['stats']}
    def singular(stage):
        error = SingularPhysicalMatrix('singular original HB physical matrix')
        error.metadata = dict(metadata, singularity_stage=stage)
        error.statistics = stats
        raise error
    if prepared['status'] == 'PHYSICAL_SINGULAR_CERTIFICATE': singular('finite_window')
    if prepared['status'] != 'HB_FRONTEND_READY': raise ArithmeticError('unexpected HB frontend status')
    carrier = acquire_weighted_carrier(prepared['H'],k,prepared['chart'])
    projected = acquire_weighted_projections(carrier,prepared['X'],prepared['Y'])
    correction = acquire_weighted_correction(prepared['raw'],projected)
    stats.update(carrier=carrier.service.stats,projections=projected.stats,
                 correction=correction.stats,terminal={})
    terminal = Services(field,stats['terminal'])
    if arm != 'current':
        terminal = HBPreparedDense(field,correction.G0,correction.H0,stats['terminal'])
        metadata['method'] = 'Structured-CC'
        metadata['terminal_method'] = 'HB_CLASSICAL_PREPARED_PLU'
    metadata.update(mu=prepared['mu'],selected_coset=prepared['search']['selected_coset'],
                    repair_rank=projected.s,represented_order=correction.order,
                    stein_width=len(correction.G0[0]),
                    chart_c_is_one=True,scaled_raw_acquired=True,
                    forward_scaling='D rhs then D^-1 recovery',
                    transpose_scaling='D^-1 rhs then D recovery')
    kernel = terminal.solve_stein(correction.G0,correction.H0,homogeneous=True)
    metadata['inner_calls'].append('homogeneous_G0_H0')
    if arm == 'current': metadata['kn_calls'].append('homogeneous_G0_H0')
    metadata['homogeneous_setup_calls'] = 1
    metadata['paid_dense_base'] = kernel['paid_dense_base']
    if kernel['status'] != 'NULLSPACE': raise ArithmeticError('KN homogeneous nullspace status required')
    if kernel['nullity']: singular('homogeneous_correction')
    metadata['nonsingularity_certified'] = True
    return PreparedHB(field,projected,correction,terminal,prepared['mu'],metadata,stats)


def solve_hb(F, n, k, raw, omega, forward_rhs, transpose_rhs, *, arm='current'):
    """Compatible two-RHS convenience API; prepare_hb exposes reusable queries."""
    if len(forward_rhs) != n*n or len(transpose_rhs) != n*n:
        raise ValueError('both physical RHS must have n*n native entries')
    attempt = prepare_hb(F,n,k,raw,omega,arm=arm)
    solution = attempt.query(forward_rhs)
    transpose_solution = attempt.query(transpose_rhs,transpose=True)
    return {'solution':solution,'transpose_solution':transpose_solution,
            'metadata':attempt.metadata,'statistics':attempt.statistics}
