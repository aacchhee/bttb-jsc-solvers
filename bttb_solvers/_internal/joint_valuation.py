# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured half-strip comparison.

Role: Trial/shared. Select active valuations and extract common factors/column quotients during joint compilation.

Entry points: minimum_active; column_quotient; extract_common.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.joint import module_rows
from bttb_solvers._internal.halfstrip_core.uniform_terminal import _FieldOperations
from bttb_solvers._internal.joint_arrays import SeriesOps, trim_factor


def minimum_active(state, threshold):
    """Actual active-entry minimum below threshold, including all collisions."""
    F, ops, N = state.F, state.ops, state.precision
    if type(threshold) is not int or not 0 <= threshold <= N:
        raise ValueError("invalid minimum threshold")

    
    sparse = [[{a: value for a, value in enumerate(poly) if not F.is_zero(value)}
               for poly in row] for row in state.g]

    

    

    selected = module_rows(_FieldOperations(F), sparse, N)
    best = threshold
    for row in selected:
        for column, label_id in enumerate(state.active_ids):
            if state.active_ids[row] == label_id: continue
            numerator = ops.dot(state.g[row], state.j[column], threshold)

            
            F.inv(F.sub(state.catalog.labels[state.active_ids[row]],
                        state.catalog.labels[label_id]))
            best = min(best, ops.valuation(numerator, threshold))
    for poly in state.active_diagonal:
        best = min(best, ops.valuation(poly, threshold))
    return {"valuation": best, "zero_mod_threshold": best == threshold,
            "selected_original_rows": selected}


def column_quotient(F, g, j, a, N, ops=None):
    """Return V,Vinv,eta and analytic quotient factors at N-(r+1)a.

    Literal streamed minimum-pivot rectangular elimination of algorithm §2.
    Row operations affect only the s-by-r scratch; only r-square column maps
    are retained. Every processed-column quotient guard remains native.
    """
    ops = ops or SeriesOps(F)
    s, r = len(g), len(g[0]) if g else 0
    if type(a) is not int or type(N) is not int or a < 1 or not 1 <= r <= 4 or N <= (r+1)*a:
        raise ValueError("small-column quotient scope/precision violated")
    if len(j) != s or any(len(row) != r for factor in (g, j) for row in factor):
        raise ValueError("factor shapes disagree")
    if any(len(poly) != N for factor in (g, j) for row in factor for poly in row):
        raise ValueError("factor coefficient count disagrees with precision")
    A = trim_factor(g, N)
    V = [[ops.unit(N) if b == c else ops.zero(N) for c in range(r)] for b in range(r)]
    Vinv = [[list(poly) for poly in row] for row in V]
    eta, current = [], N
    for p in range(min(s, r)):
        best = (a, p, p)
        for i in range(p, s):
            for c in range(p, r):
                candidate = (ops.valuation(A[i][c], a), i, c)
                if candidate < best: best = candidate
        v, i, c = best
        if v == a: break
        A[p], A[i] = A[i], A[p]
        if c != p:
            for row in A: row[p], row[c] = row[c], row[p]
            for row in V: row[p], row[c] = row[c], row[p]
            Vinv[p], Vinv[c] = Vinv[c], Vinv[p]
        new_precision = current-v
        inverse = ops.inv(A[p][p][v:], new_precision)

        
        A[p] = [ops.mul(inverse, poly, current) for poly in A[p]]
        for i in range(p+1, s):
            quotient = list(A[i][p][v:current])
            for c in range(p, r):
                A[i][c] = ops.sub(A[i][c], ops.mul(quotient, A[p][c], current))
        for c in range(p+1, r):
            quotient = list(A[p][c][v:current])
            for i in range(s):
                A[i][c] = ops.sub(A[i][c], ops.mul(A[i][p], quotient, current))
            for i in range(r):
                V[i][c] = ops.sub(V[i][c], ops.mul(V[i][p], quotient, current))

            Vinv[p] = [ops.add(x, ops.mul(quotient, y, current))
                       for x, y in zip(Vinv[p], Vinv[c])]
        eta.append(v)
        current = new_precision
        A, V, Vinv = trim_factor(A, current), trim_factor(V, current), trim_factor(Vinv, current)
    eta += [a]*(r-len(eta))
    target = N-(r+1)*a
    gbar = ops.right(g, V, current)
    jbar = ops.right(j, Vinv, current, inverse_transpose=True)
    newg = [[ops.divide_shift(poly, v, target) for poly, v in zip(row, eta)] for row in gbar]
    newj = [[ops.divide_shift(poly, a-v, target) for poly, v in zip(row, eta)] for row in jbar]
    return {"V": V, "Vinv": Vinv, "eta": eta, "g": newg, "j": newj,
            "precision": target, "transform_precision": current}


def extract_common(state, a):
    """algorithm (5)–(6), preserving raw C,D,X and both history units."""
    N, r, ops = state.precision, state.width, state.ops
    quotient = column_quotient(state.F, state.g, state.j, a, N, ops)
    target = N-(r+1)*a
    Ebar = ops.right(state.E, quotient["V"], quotient["transform_precision"])
    Fbar = ops.right(state.Fscaled, quotient["Vinv"],
                     quotient["transform_precision"], inverse_transpose=True)
    eta = quotient["eta"]
    result = state.copy()
    result.g, result.j = quotient["g"], quotient["j"]
    result.E = [[ops.delay(poly, a-v, target) for poly, v in zip(row, eta)] for row in Ebar]
    result.Fscaled = [[ops.delay(poly, v, target) for poly, v in zip(row, eta)] for row in Fbar]
    result.active_diagonal = [ops.divide_shift(poly, a, target) for poly in state.active_diagonal]
    result.C_collisions = {i: list(poly[:target]) for i, poly in state.C_collisions.items()}
    result.D_collisions = {i: list(poly[:target]) for i, poly in state.D_collisions.items()}

    result.X_first_row = [ops.delay(poly, a, target) for poly in state.X_first_row]
    result.X_passive_diagonal = {i: ops.delay(poly, a, target)
                                 for i, poly in state.X_passive_diagonal.items()}
    result.common_depth += a
    result.precision = target
    return result
