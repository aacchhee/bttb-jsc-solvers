# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.cauchy_transforms import CauchyCatalog
from bttb_solvers._internal.compact_cauchy import CompactCauchy
from bttb_solvers._internal.constant_profile import rank_profile_kernel
from bttb_solvers._internal.joint_arrays import permutation_sign
from bttb_solvers._internal.rectangular_cauchy import RectangularCauchy
from bttb_solvers._internal.mixed_cross import MixedLowerCross, _transpose_rect, external_product_boundary
from bttb_solvers._internal.source_histories import PrincipalHistory


def materialize_physical(upper, length):
    """Paid triangular recurrence only at the requested emitted window."""
    if type(length) is not int or not 0 <= length <= upper.precision:
        raise ValueError("physical emission window unavailable")
    F, n, s = upper.F, upper.n, len(upper.active_ids)
    if not length:
        return [[[] for _ in range(s)] for _ in range(n)]
    from bttb_solvers._internal.joint_arrays import SeriesOps
    arithmetic = SeriesOps(F)
    result = [[None for _ in range(s)] for _ in range(n)]
    for a, label_id in enumerate(upper.active_ids):
        scalar = F.inv(upper.catalog.labels[label_id])
        following = arithmetic.zero(length)
        for i in range(n-1, -1, -1):
            rhs = arithmetic.dot([p[:length] for p in upper.E[i]],
                                  [p[:length] for p in upper.H[a]], length)
            value = arithmetic.sub(following, rhs)
            result[i][a] = [F.mul(scalar, x) for x in value]
            following = result[i][a]
    return result


def whole_principal_block(state):
    """Remove all rank(A(0)) coordinates together, preserving width4 state."""
    F, ops, N, r = state.F, state.ops, state.precision, state.width
    ids = state.active_ids
    s = len(ids)
    constant = [[F.zero() for _ in ids] for _ in ids]
    for a, i in enumerate(ids):
        for b, j in enumerate(ids):
            if i == j:
                constant[a][b] = state.active_diagonal[a][0]
            else:
                value = F.zero()
                for u in range(r):
                    value = F.add(value, F.mul(state.g[a][u][0], state.j[b][u][0]))
                constant[a][b] = F.neg(F.mul(value, F.inv(F.sub(state.catalog.labels[i], state.catalog.labels[j]))))
    profile = rank_profile_kernel(F, constant)
    t = profile["rank"]
    if not t:
        raise ValueError("whole unit-block step requires positive constant rank")
    I = tuple(ids[a] for a in profile["column_profile"])
    chosen = set(I)
    R = tuple(i for i in ids if i not in chosen)
    del constant, profile
    pivot, ARI, AIR = (state.active_block(rows, columns)
                       for rows, columns in ((I, I), (R, I), (I, R)))
    local = CauchyCatalog(F, [state.catalog.labels[i] for i in I])
    compact = CompactCauchy(local, pivot.G, pivot.H, [pivot.collisions[i] for i in I], N, width=r)
    compact_inverse, determinant = compact.inverse(return_determinant=True)
    inverse = RectangularCauchy(state.catalog, I, I, compact_inverse.G, compact_inverse.H,
        {i: compact_inverse.diagonal[a] for a, i in enumerate(I)}, N, width=r)
    CI, DI = state.upper(I), state.lower(I)
    pos = {i: a for a, i in enumerate(ids)}
    gI, jI = [state.g[pos[i]] for i in I], [state.j[pos[i]] for i in I]
    pinvg, pinvtj = [[None]*r for _ in I], [[None]*r for _ in I]
    Enew = [[list(poly) for poly in row] for row in state.E]
    Fnew = [[list(poly) for poly in row] for row in state.Fscaled]
    gnew, jnew = [[None]*r for _ in R], [[None]*r for _ in R]
    for u in range(r):
        x = inverse.action([row[u] for row in gI])
        y = inverse.action([row[u] for row in jI], transpose=True)
        gx, jy = ARI.action(x), AIR.action(y, transpose=True)
        ex, fy = CI.action(x), DI.action(y, transpose=True)
        for a in range(t):
            pinvg[a][u], pinvtj[a][u] = x[a], y[a]
        for a, i in enumerate(R):
            gnew[a][u] = ops.sub(state.g[pos[i]][u], gx[a])
            jnew[a][u] = ops.sub(state.j[pos[i]][u], jy[a])
        for a in range(len(state.E)):
            Enew[a][u] = ops.add(state.E[a][u], ex[a])
            Fnew[a][u] = ops.add(state.Fscaled[a][u], fy[a])
    active_correction = ARI.product(inverse).product(AIR)
    upper_correction = CI.normalize(inverse).passive.product(AIR)
    lower_correction = ARI.product(inverse).product(_transpose_rect(DI.upper.passive))
    boundary = external_product_boundary(CI, inverse, DI)

    generic = DI.normalize_left(inverse)
    thin = [[None]*r for _ in state.E]
    for u in range(r):
        column = generic.action([row[u] for row in jI], transpose=True)
        for a in range(len(state.E)):
            thin[a][u] = ops.add(column[a], state.Fscaled[a][u])
    Dhat = MixedLowerCross.from_factors(state.catalog, I, pinvg,
        thin[:state.n], state.passive_ids, thin[state.n:],
        generic.upper.passive.collisions, N, width=r)
    result = state.copy()
    result.active_ids, result.g, result.j = R, gnew, jnew
    result.E, result.Fscaled = Enew, Fnew
    result.active_diagonal = [ops.sub(state.active_diagonal[pos[i]], active_correction.collisions[i]) for i in R]
    result.C_collisions = {i: ops.sub(state.C_collisions[i], upper_correction.collisions[i]) for i in R}
    result.D_collisions = {i: ops.sub(state.D_collisions[i], lower_correction.collisions[i]) for i in R}
    result.X_first_row = [ops.add(poly, term) for poly, term in zip(state.X_first_row, boundary["first_row"])]
    result.X_passive_diagonal = {i: ops.add(poly, boundary["passive_diagonal"][i])
                                 for i, poly in state.X_passive_diagonal.items()}
    result.emitted_delay_sum += t*state.common_depth
    sign = permutation_sign(F, ids, I+R)
    history = PrincipalHistory(ids, I, R, inverse, ARI, AIR, 2*state.L)
    return {"state": result, "pivot_ids": I, "remainder_ids": R,
            "determinant": determinant, "permutation_sign": sign,
            "history": history, "C": CI, "Dhat": Dhat, "rank": t}
