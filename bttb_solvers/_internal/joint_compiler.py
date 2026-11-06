# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured half-strip comparison.

Role: Trial/shared. Compile the explicit-history joint-defect representation. Main HSC/HPR use halfstrip_core/joint.py instead.

Entry points: compile_joint.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from types import SimpleNamespace
from bttb_solvers._internal.joint_state import JointState
from bttb_solvers._internal.joint_valuation import minimum_active, extract_common
from bttb_solvers._internal.joint_blocks import whole_principal_block, materialize_physical
from bttb_solvers._internal.source_histories import PairedHistories


def compile_joint(initial, *, phase_hook=None):
    state = JointState.from_initial(initial)
    F, n, L = state.F, state.n, state.L
    histories = PairedHistories(F, state.active_ids, 2*L)
    result = SimpleNamespace(initial=initial, state=state, histories=histories,
        delta=[], B_low=[[] for _ in range(n)], G_low=[], dU=F.one(), dV=F.one(),
        overcap=False, lower_bound=None, phases=[], residual_ids=())

    def emit(C, D, delay):
        physicalC = materialize_physical(C, delay)
        physicalDt = materialize_physical(D.upper, delay)
        for j in range(len(C.active_ids)):
            for i in range(n): result.B_low[i].append(physicalC[i][j])
            result.G_low.append([physicalDt[n-1-i][j] for i in range(n)])
            result.delta.append(delay)

    while state.active_ids:
        s, d, K = len(state.active_ids), state.common_depth, state.emitted_delay_sum
        if state.precision != 8*L-5*d:
            raise ArithmeticError("common-depth precision invariant violated")
        minimum = minimum_active(state, L-d)
        a = minimum["valuation"]
        lower = K+s*(L if minimum["zero_mod_threshold"] else d+a)
        if lower > state.h:
            result.overcap, result.lower_bound, result.state = True, lower, state
            result.k = sum(result.delta)
            return result
        if minimum["zero_mod_threshold"]:

            emit(state.upper(state.active_ids), state.lower(state.active_ids), L)
            result.residual_ids, result.state = state.active_ids, state
            break
        if a:
            state = extract_common(state, a)
        before = state
        block = whole_principal_block(state)
        emit(block["C"], block["Dhat"], state.common_depth)
        histories.append(block["history"])
        result.dU = F.mul(result.dU, F.mul(block["permutation_sign"], F.inv(block["determinant"])))
        result.dV = F.mul(result.dV, block["permutation_sign"])
        state = block["state"]
        receipt = {"active_before": s, "rank_removed": block["rank"],
            "extraction": a, "common_depth": state.common_depth,
            "emitted_delay_sum": state.emitted_delay_sum, "precision": state.precision,
            "potential": state.emitted_delay_sum+len(state.active_ids)*state.common_depth,
            "emitted_cells": 2*n*block["rank"]*state.common_depth}
        result.phases.append(receipt)
        if phase_hook is not None: phase_hook(before, block, receipt)
        result.state = state
    result.k = sum(result.delta)
    if result.k > state.h:
        raise ArithmeticError("accepted channel delays exceed cap")
    return result
