# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial. Narrow-band acquisition/restart, weighted carrier/correction construction, generator terminal and physical answer reconstruction.

Entry points: solve_narrow; solve_narrow_many.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata as dense_metadata, terminal_method, selection
from bttb_solvers._internal.halfstrip_core.exact import Restart
from bttb_solvers._internal.narrow_raw_frontend import RawNarrowFrontend
from bttb_solvers._internal.weighted_carrier import acquire_weighted_carrier
from bttb_solvers._internal.weighted_projections import acquire_weighted_projections
from bttb_solvers._internal.weighted_correction import acquire_weighted_correction
from bttb_solvers._internal.kn_services import Services, METHOD, phase_call
from bttb_solvers._internal.wide_physical_solver import SingularPhysicalMatrix
from bttb_solvers._internal.generator_compression import compress_generators as _compress_generators
import time


def _singular(metadata, statistics):
    error = SingularPhysicalMatrix('singular raw physical matrix')
    error.metadata, error.statistics = metadata, statistics
    raise error


def _query(F, projected, correction, terminal, g, h, rhs):
    n, s, ell = correction.n, correction.s, correction.ell
    y = projected.physical_action(rhs)
    reduced = [y[p*n+a] for p in correction.positions for a in range(n)]
    for i in range(n):
        for b in range(s):
            value = F.zero()
            for a in range(n):
                value = F.add(value, F.mul(projected.Y[a][b], y[i*n+a]))
            reduced.append(value)
    pi = correction.permutation
    answer = terminal.solve_stein(g, h,
                                 list(reversed([reduced[i] for i in pi])))
    if answer['status'] != 'CONSISTENT':
        raise ArithmeticError('inconsistent query after physical nonsingularity gate')
    z = [F.zero() for _ in pi]
    for i, value in zip(pi, reversed(answer['solution'])):
        z[i] = value
    lifted = [F.zero() for _ in range(n*n)]

    
    multiply = projected.carrier.service.mul_scalar
    right_columns = [[z[q*n+b] for q in range(ell)] for b in range(n)]
    for a in range(n):
        for b in range(n):
            product = multiply(correction.query_left[a][b], right_columns[b])
            for p, position in enumerate(correction.positions):
                index = ell-1+p
                if index < len(product):
                    at = position*n+a
                    lifted[at] = F.add(lifted[at], product[index])
    for i in range(n):
        for a in range(n):
            for b in range(s):
                at = i*n+a
                lifted[at] = F.add(lifted[at], F.mul(projected.X[a][b], z[ell*n+i*s+b]))
    correction_action = projected.physical_action(lifted)
    return [F.add(a, b) for a, b in zip(y, correction_action)]


class _PreparedBatchAttempt:
    """Private preparation; never escapes its whole-batch restart owner."""
    def __init__(self,context,F,projected,correction,terminal,g,h,metadata,stats):
        self.context,self.F=context,F
        self.projected,self.correction,self.terminal=projected,correction,terminal
        self.g,self.h=g,h
        self.metadata,self.stats=metadata,stats
        self.stage='batch_queries'
        self.pending=[]

    def solve_many(self,pairs,setup_seconds):
        started=time.perf_counter()
        for index,(forward,transposed) in enumerate(pairs):
            self.stage='batch_forward_query_'+str(index)
            answer=phase_call(self.stats,'forward_queries',_query,self.F,self.projected,self.correction,self.terminal,self.g,self.h,
                          [self.context.embed(value) for value in forward])
            self.metadata['forward_calls']+=1
            self.stage='batch_literal_transpose_query_'+str(index)
            answer_t=list(reversed(phase_call(self.stats,'transpose_queries',_query,self.F,self.projected,self.correction,self.terminal,self.g,self.h,
                         [self.context.embed(value) for value in reversed(transposed)])))
            self.metadata['transpose_calls']+=1
            self.pending.append((answer,answer_t))
        query_seconds=time.perf_counter()-started
        self.stage='batch_physical_descent'
        started=time.perf_counter()
        solutions=[[self.context.descend(value) for value in pair[0]] for pair in self.pending]
        transpose_solutions=[[self.context.descend(value) for value in pair[1]] for pair in self.pending]
        self.metadata.update(label_setup=self.context.metadata,batch_size=len(pairs),
                             batch_preparation='one preparation and homogeneous gate per attempt',
                             batch_restart_scope='discard every pending output; replay all original native RHS pairs')
        self.stats['batch_phase_seconds']={'scope':'successful attempt only; public total includes failed attempts',
            'setup_wall_seconds':setup_seconds,'queries_wall_seconds':query_seconds,
            'descent_wall_seconds':time.perf_counter()-started}
        return {'solutions':solutions,'transpose_solutions':transpose_solutions,
                'metadata':self.metadata,'statistics':self.stats}

    def invalidate(self):
        self.__dict__.clear()
        self.invalidated=True


def _attempt(driver, native_forward, native_transpose, compress_generators, native_pairs=None):
    batch_started=time.perf_counter() if native_pairs is not None else None
    batch=None
    stats={}
    prepared, receipt = phase_call(stats,'frontend',driver._attempt)
    if receipt is not None:
        return None, receipt
    context, F = prepared['context'], prepared['field']
    stage = 'physical_singularity_certificate'
    metadata = {'method': ('reference_DEDICATED_NARROW_CONDITIONAL_COMPRESSION' if compress_generators
                           else 'reference_DEDICATED_NARROW_UNCOMPRESSED'),
                'terminal_method': terminal_method(), 'dimension_branch': '2k<n',
                'source': 'algorithm original-lag weighted cyclic correction',
                'n': driver.n, 'k': driver.k, 'physical_order': driver.n**2,
                'homogeneous_setup_calls': 0, 'forward_calls': 0,
                'transpose_calls': 0, 'nonsingularity_certified': False,
                'compression_enabled': compress_generators,
                'compression_applied': False, 'compression_pass_ranks': [],
                'compression_pass_widths': [], 'compression_source': None}
    metadata.update(dense_metadata())
    stats['frontend'] = prepared['stats']
    try:
        if prepared['physical_singular']:
            _singular(dict(metadata, singularity_stage=stage), stats)
        stage = 'weighted_carrier'
        carrier = phase_call(stats,'carrier',acquire_weighted_carrier,prepared['H'], driver.k, prepared['chart'])
        stage = 'weighted_projections'
        projected = phase_call(stats,'projections',acquire_weighted_projections,carrier, prepared['X'], prepared['Y'])
        stage = 'weighted_correction'
        correction = phase_call(stats,'correction',acquire_weighted_correction,prepared['raw'], projected)
        stats.update(carrier=carrier.service.stats, projections=projected.stats,
                     correction=correction.stats, terminal={}, generator_compression_passes=[])
        g, h = correction.G0, correction.H0
        original_width = len(g[0])
        metadata.update(represented_order=correction.order, repair_rank=projected.s,
                        acquired_stein_width=original_width,
                        original_stein_width=original_width)

        

        if compress_generators and correction.order > original_width:
            metadata.update(compression_applied=True,
                            compression_source='accepted checked exact two-pass rank-factor compression')
            stage = 'generator_compression_first'
            first = phase_call(stats,'compression_first',_compress_generators,F, g, h)
            g, h = first.G, first.H
            metadata['compression_pass_ranks'].append(first.rank)
            metadata['compression_pass_widths'].append(
                {'input_width': first.input_width, 'output_width': first.rank})
            stats['generator_compression_passes'].append(first.stats)
            del first
            if metadata['compression_pass_ranks'][-1]:
                stage = 'generator_compression_second'
                second = phase_call(stats,'compression_second',_compress_generators,F, h, g)
                g, h = second.H, second.G
                metadata['compression_pass_ranks'].append(second.rank)
                metadata['compression_pass_widths'].append(
                    {'input_width': second.input_width, 'output_width': second.rank})
                stats['generator_compression_passes'].append(second.stats)
                del second
        metadata.update(stein_width=len(g[0]), effective_stein_width=len(g[0]))
        terminal = phase_call(stats,'terminal_preparation',Services(F, stats['terminal']).prepare_classical_base,g,h)
        stage = 'homogeneous_physical_nonsingularity'
        kernel = phase_call(stats,'homogeneous_gate',terminal.solve_stein,g, h, homogeneous=True)
        metadata.update(homogeneous_setup_calls=1, paid_dense_base=kernel['paid_dense_base'])
        if kernel['status'] != 'NULLSPACE':
            raise ArithmeticError('KN homogeneous service did not return nullspace status')
        if kernel['nullity']:
            _singular(dict(metadata, singularity_stage=stage), stats)
        metadata['nonsingularity_certified'] = True

        
        correction.query_left = [
            [[correction.u[d][a][b] if d in correction.u else F.zero()
              for d in range(1-correction.ell, correction.ell)]
             for b in range(driver.n)] for a in range(driver.n)]
        if native_pairs is not None:
            batch=_PreparedBatchAttempt(context,F,projected,correction,terminal,g,h,metadata,stats)
            return batch.solve_many(native_pairs,time.perf_counter()-batch_started),None
        stage = 'forward_query'
        forward = phase_call(stats,'forward_queries',_query,F, projected, correction, terminal, g, h,
                         [context.embed(value) for value in native_forward])
        metadata['forward_calls'] = 1
        stage = 'literal_transpose_query'
        transposed = list(reversed(phase_call(stats,'transpose_queries',_query,F, projected, correction, terminal, g, h,
                                [context.embed(value) for value in reversed(native_transpose)])))
        metadata['transpose_calls'] = 1
        stage = 'physical_descent'
        forward = [context.descend(value) for value in forward]
        transposed = [context.descend(value) for value in transposed]
        metadata['label_setup'] = context.metadata
        return {'solution': forward, 'transpose_solution': transposed,
                'metadata': metadata, 'statistics': stats}, None
    except Restart as event:

        
        if batch is not None:stage=batch.stage
        receipt=driver._restriction_receipt(context,event,stage)
        if 'terminal' in locals():terminal.invalidate()
        if batch is not None:batch.invalidate()
        return None,receipt


def solve_narrow(base, n, k, raw, forward_rhs, transpose_rhs, *, compress_generators=True):
    """One complete forward and independent literal-transpose native query."""
    if type(compress_generators) is not bool:
        raise ValueError('compress_generators must be a Boolean route choice')
    forward, transposed = tuple(forward_rhs), tuple(transpose_rhs)
    if len(forward) != n*n or len(transposed) != n*n:
        raise ValueError('both physical RHS must have n*n native entries')
    driver = RawNarrowFrontend(base, n, k, raw)
    while True:
        driver.attempts += 1
        try:
            answer, receipt = _attempt(driver, forward, transposed, compress_generators)
        except SingularPhysicalMatrix as error:
            error.metadata.update(attempts=driver.attempts, restarts=list(driver.restarts))
            raise
        if receipt is None:
            answer['metadata'].update(attempts=driver.attempts,
                                      restarts=list(driver.restarts))
            return answer
        driver._record_restriction(receipt)
        driver.restarts[-1]['invalidated'] = 'all context/frontend/carrier/projection/correction/terminal/query/descent state'


def solve_narrow_many(base,n,k,raw,forward_rhs_list,transpose_rhs_list,*,compress_generators=True):
    """One owned finite batch; any restriction replays all original native RHS."""
    if type(compress_generators) is not bool:raise ValueError('compress_generators must be Boolean')
    forwards=tuple(tuple(rhs) for rhs in forward_rhs_list)
    transposes=tuple(tuple(rhs) for rhs in transpose_rhs_list)
    if not forwards or len(forwards)!=len(transposes):raise ValueError('nonempty matching RHS batches required')
    if any(len(rhs)!=n*n for rhs in forwards+transposes):raise ValueError('physical RHS dimension mismatch')
    pairs=tuple(zip(forwards,transposes))
    driver=RawNarrowFrontend(base,n,k,dict(raw))
    while True:
        driver.attempts+=1
        try:
            answer,receipt=_attempt(driver,None,None,compress_generators,native_pairs=pairs)
        except SingularPhysicalMatrix as error:
            error.metadata.update(attempts=driver.attempts,restarts=list(driver.restarts),batch_size=len(pairs))
            raise
        if receipt is None:
            answer['metadata'].update(attempts=driver.attempts,restarts=list(driver.restarts))
            return answer
        driver._record_restriction(receipt)
        driver.restarts[-1].update(
            invalidated='all context/frontend/carrier/projection/correction/terminal/query/descent state',
            discarded_pending_batch=True,replay_rhs_pairs=len(pairs))
