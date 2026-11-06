# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
import time
from bttb_solvers._internal.banded_bttb.field import PrimeField
from bttb_solvers._internal.halfstrip.counter import HalfStripCounter
from bttb_solvers._internal.halfstrip.symbol import HalfStripSymbol
from bttb_solvers._internal.halfstrip.solver import HalfStripSolver
from bttb_solvers._internal.halfstrip_core.dense_backend import dense_scope, metadata


def solve_hir(p,lags,n,w,rhs=None,*,response_layout='wn_homogeneous',terminal_backend='structured_classical',count_ops=True):
    """Raw prime-field lags -> determinant and optional one RHS.

    Explicit NTT is required. Existing SingularLeadingBlock/NoNTTRoot escape
    unchanged; physical singularity returns zero determinant and no solution.
    No automatic added algebra, generic all-field claim or transpose adapter.
    This one-shot API defaults to wn_homogeneous; prepare_hir defaults
    to n_impulse. Both terminal backends support either explicit layout.
    """
    if type(p) is not int or p<2:raise ValueError('explicit finite prime characteristic required')
    if type(n) is not int or type(w) is not int or n<2 or not 1<=w<n:raise ValueError('integer halfstrip dimensions required')
    if not lags or len(lags)>n+w or any(len(row)!=2*n-1 for row in lags):raise ValueError('raw halfstrip lag dimensions')
    if rhs is not None and len(rhs)!=n*n:raise ValueError('physical RHS dimension')
    cnt=HalfStripCounter(count_ops=count_ops);F=PrimeField(p,cnt,count_ops=count_ops)
    symbol=HalfStripSymbol(F,n,w,{d-w:tuple(row) for d,row in enumerate(lags)})
    solver=HalfStripSolver(symbol,cnt,route='cdq',convolution_backend='ntt',terminal_backend=terminal_backend,response_layout=response_layout)
    with dense_scope('classical',1,polynomial_entry_policy='off') as ledger:
        started,cpu=time.perf_counter(),time.process_time();report=solver.prepare()
        phases={'prepare':{'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu}}
        det=solver.det();singular=F.is_zero(det);solution=None
        if rhs is not None and not singular:
            started,cpu=time.perf_counter(),time.process_time();solution=solver.solve(rhs)
            phases['solve']={'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu}
        terminal=solver.terminal.summary() if hasattr(solver.terminal,'summary') else {'terminal_backend':'dense','matrix_kind':'K' if response_layout=='n_impulse' else 'Bt'}
        return {'det':det,'singular':singular,'physical_solution':solution,'n':n,'w':w,
                'metadata':{'method':('RECURRENCE_STRUCTURED_CLASSICAL_TERMINAL' if terminal_backend=='structured_classical'
                              else 'RECURRENCE_DENSE_TERMINAL'),
                    'source_backend':('recurrence_structured_classical' if terminal_backend=='structured_classical'
                                      else 'recurrence_dense'),
                    'source':'n length-L inverse impulses; one-response query reconstruction' if response_layout=='n_impulse' else 'original wn homogeneous/particular/H reconstruction',
                    'terminal_engine':'uniform_fast' if terminal_backend=='structured_classical' else 'dense_classical','terminal_order':n*w,
                    'terminal_matrix_materialized':terminal_backend=='dense','source_horizon':n+w,
                    'response_layout':response_layout,'prepare_report':report,
                    'convolution_backend':solver.backend.summary(),
                    'cubic_ledger_achieved':bool(report['cubic_ledger_achieved']) and terminal_backend=='structured_classical',
                    'cost_scope':('compiled polynomial kernels; no inherited cubic total-work claim'
                                  if not solver.backend.achieved else
                                  'soft O(w*n^3) with actual recurrence NTT; includes generator extraction'
                                  if terminal_backend=='structured_classical' else
                                  'dense terminal comparison arm; no cubic total-work claim'),
                    'H0_domain':'existing nonsingular leading-block requirement',
                    'field_scope':'explicit supplied prime; required NTT roots; no added algebra',
                    'dense_physical_matrix':False,**metadata()},
                'statistics':{'dense_scope_ledger':dict(ledger),'terminal':terminal,
                    'public_phase_seconds':phases,'ops':solver.counts(),
                    **({'operation_counts_valid':False} if not (count_ops and solver.backend.achieved) else {})}}


def prepare_hir(p,lags,n,w,*,terminal_backend='structured_classical',
                         response_layout='n_impulse',count_ops=True):
    """Prepare once; det/forward/literal-transpose reuse this exact object.

    The default n_impulse layout differs from solve_hir's default
    wn_homogeneous layout; callers may select either explicitly.
    Explicit prime + NTT roots and nonsingular F0 are still required. The
    returned solver's solve_transpose implements Pi*solve(Pi*b), with the
    reversal of BOTH physical axes, as in algorithm(c).
    """
    if type(p) is not int or p<2:raise ValueError('explicit finite prime characteristic required')
    if type(n) is not int or type(w) is not int or n<2 or not 1<=w<n:raise ValueError('integer halfstrip dimensions required')
    rows=tuple(tuple(row) for row in lags)
    if not rows or len(rows)>n+w or any(len(row)!=2*n-1 for row in rows):raise ValueError('raw halfstrip lag dimensions')
    cnt=HalfStripCounter(count_ops=count_ops);F=PrimeField(p,cnt,count_ops=count_ops)
    symbol=HalfStripSymbol(F,n,w,{d-w:row for d,row in enumerate(rows)})
    solver=HalfStripSolver(symbol,cnt,route='cdq',convolution_backend='ntt',
                           terminal_backend=terminal_backend,response_layout=response_layout)
    with dense_scope('classical',1,polynomial_entry_policy='off'):
        solver.prepare()
    return solver

