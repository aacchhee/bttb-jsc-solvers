# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial. Select classical or explicit Strassen dense operations around the complete dedicated physical solve; the classical many-pair service has a separate reuse contract.

Entry points: solve_dedicated_banded; solve_dedicated_banded_many.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.dense_backend import dense_scope, metadata as dense_metadata
from bttb_solvers._internal.banded_physical_solver import solve_banded as _solve_banded, SingularPhysicalMatrix


def solve_dedicated_banded(base,n,k,raw,forward_rhs,transpose_rhs,*,compress_generators=True,
                 dense_backend='strassen',strassen_leaf=1,polynomial_entry_policy='off'):
    """Complete dedicated solve with an explicit context-local dense choice.

    Scope includes raw acquisition, every restart, gate, both queries and
    descent. Fast scalar polynomial arithmetic is retained in every mode.
    """
    with dense_scope(dense_backend,strassen_leaf,polynomial_entry_policy=polynomial_entry_policy) as ledger:
        try:
            result=_solve_banded(base,n,k,raw,forward_rhs,transpose_rhs,
                                 compress_generators=compress_generators)
        except SingularPhysicalMatrix as error:
            error.metadata.update(dense_metadata(),dense_scope='complete_raw_and_both_queries')
            error.statistics['dense_scope_ledger']=dict(ledger)
            raise
        result['metadata'].update(dense_metadata(),dense_scope='complete_raw_and_both_queries',
            dedicated_backend_method=('Dedicated-classical' if dense_backend=='classical'
                                      else 'Dedicated-Strassen'))
        result['statistics']['dense_scope_ledger']=dict(ledger)
        return result


def solve_dedicated_banded_classical(base,n,k,raw,forward_rhs,transpose_rhs,*,compress_generators=True,polynomial_entry_policy='off'):
    """Actual classical dense algorithm counterpart; scalar arithmetic stays fast."""
    return solve_dedicated_banded(base,n,k,raw,forward_rhs,transpose_rhs,
                        compress_generators=compress_generators,dense_backend='classical',polynomial_entry_policy=polynomial_entry_policy)


def solve_dedicated_banded_many(base,n,k,raw,forward_rhs_list,transpose_rhs_list,*,compress_generators=True,
                              dense_backend='classical',strassen_leaf=1,polynomial_entry_policy='off'):
    """Classical owned RHS batch; preparation is shared inside one public call.

    No long-lived prepared object or partial result escapes. Narrow restriction
    replay owns the complete original-native batch, not individual queries.
    """
    import time
    from bttb_solvers._internal.narrow_physical_solver import solve_narrow_many
    from bttb_solvers._internal.wide_physical_solver import prepare_wide
    if dense_backend!='classical' or strassen_leaf!=1:raise ValueError('batch reuse is classical-only')
    if type(compress_generators) is not bool:raise ValueError('compress_generators must be Boolean')
    if type(n) is not int or type(k) is not int or n<2 or not 1<=k<n:raise ValueError('integer banded dimensions required')
    keys=tuple((a,b) for a in range(-k,k+1) for b in range(1-n,n))
    if any(key not in raw for key in keys):raise ValueError('complete original raw lag mapping required')
    cached={key:raw[key] for key in keys}
    forwards=tuple(tuple(rhs) for rhs in forward_rhs_list)
    transposes=tuple(tuple(rhs) for rhs in transpose_rhs_list)
    if not forwards or len(forwards)!=len(transposes):raise ValueError('nonempty matching RHS batches required')
    if any(len(rhs)!=n*n for rhs in forwards+transposes):raise ValueError('physical RHS dimension mismatch')
    with dense_scope('classical',1,polynomial_entry_policy=polynomial_entry_policy) as ledger:
        try:
            if 2*k<n:
                result=solve_narrow_many(base,n,k,cached,forwards,transposes,compress_generators=compress_generators)
            else:
                started=time.perf_counter()
                prepared=prepare_wide(base,n,k,cached,compress_generators=compress_generators)
                setup_seconds=time.perf_counter()-started
                started=time.perf_counter()
                pairs=[(prepared.solve(f),prepared.solve_transpose(t)) for f,t in zip(forwards,transposes)]
                result={'solutions':[pair[0] for pair in pairs],'transpose_solutions':[pair[1] for pair in pairs],
                        'metadata':dict(prepared.metadata),'statistics':dict(prepared.stats)}
                result['metadata'].update(dimension_branch='2k>=n',attempts=1,restarts=[],batch_size=len(pairs),
                    batch_preparation='one owned wide preparation and homogeneous gate',
                    batch_restart_scope='no partial batch returned')
                result['statistics']['batch_phase_seconds']={'scope':'complete wide batch',
                    'setup_wall_seconds':setup_seconds,'queries_wall_seconds':time.perf_counter()-started}
        except SingularPhysicalMatrix as error:
            error.metadata.update(dense_metadata(),dense_scope='complete_raw_and_all_batch_queries',batch_size=len(forwards))
            error.statistics['dense_scope_ledger']=dict(ledger)
            raise
        result['metadata'].update(dense_metadata(),dense_scope='complete_raw_and_all_batch_queries',
            dedicated_backend_method='Dedicated-classical')
        result['statistics']['dense_scope_ledger']=dict(ledger)
        return result
