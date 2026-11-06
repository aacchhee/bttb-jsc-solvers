# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""CC Toeplitz, symbol and correction primitives.

Role: Shared. Frequency symbols, direct/radix-two transforms, inverse-corner accumulation and circulant inverse application.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from .errors import SymbolSingular
from .field import int_element
from .symbol import symbol_lag_at
from .toeplitz import (mat_mul, toeplitz_inverse, toeplitz_solve,
                       toeplitz_det, zeros)


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def transform_route(m, requested, k=None, phase="apply"):
    """Resolve an exact length-m transform route.

    ``naive``/``direct`` is the unconditional O(m^2) fallback.  ``auto`` uses
    the algorithm minimum separately in setup and application.  At a power-of-two
    length, application uses radix-2, while setup uses radix-2 only when
    ``log2(m) < k``; ties go to the lower-memory direct route.  This preserves
    the fixed-k, m=n cubic bound instead of silently selecting n^3 log n.
    """
    if phase not in {"setup", "apply"}:
        raise ValueError("transform phase must be 'setup' or 'apply'")
    if requested == "naive":
        requested = "direct"
    if requested == "auto":
        if not _is_power_of_two(m):
            return "direct"
        if phase == "setup":
            if k is None:
                raise ValueError("automatic setup route requires k")
            log2m = m.bit_length() - 1
            return "radix2" if log2m < k else "direct"
        return "radix2"
    if requested == "radix2" and not _is_power_of_two(m):
        raise ValueError("dft='radix2' requires a power-of-two transform length")
    if requested not in {"direct", "radix2"}:
        raise ValueError("dft must be 'naive', 'direct', 'auto', or 'radix2'")
    return requested


def _radix2_dft(F, values, root):
    n = len(values)
    if n == 1:
        return list(values)
    even = _radix2_dft(F, values[0::2], F.mul(root, root))
    odd = _radix2_dft(F, values[1::2], F.mul(root, root))
    out = [F.zero()] * n
    w = F.one()
    for k in range(n // 2):
        term = F.mul(w, odd[k])
        out[k] = F.add(even[k], term)
        out[k + n // 2] = F.sub(even[k], term)
        w = F.mul(w, root)
    return out


def _dft(F, values, root, route):
    if route == "radix2":
        return _radix2_dft(F, values, root)
    m = len(values)
    out = [F.zero()] * m
    step = F.one()
    for j in range(m):
        w = F.one()
        for i, value in enumerate(values):
            if not F.is_zero(value):
                out[j] = F.add(out[j], F.mul(w, value))
            if i + 1 < m:
                w = F.mul(w, step)
        if j + 1 < m:
            step = F.mul(step, root)
    return out


def frequency_symbol_lags(F, sym, m, mu, omega):
    """Return ``a(mu*omega^-j)`` lag vectors using one linear power walk.

    This is algorithm L0: no per-frequency binary exponentiation, and the result
    is reusable by setup, every right-hand side, and the transposed solve.
    The caller chooses the counter stage so standalone use stays composable.
    """
    invomega = F.inv(omega)
    x = mu
    out = []
    for j in range(m):
        out.append(symbol_lag_at(F, sym, x))
        if j + 1 < m:
            x = F.mul(x, invomega)
    return out


def needed_shifts(m, k):
    """The distinct t such that block (i,j) of C^{-1}, which is G_{(i-j) mod m},
    is touched by the corner index set S_blk = {0..k-1} U {m-k..m-1}.

    DERIVED from the index arithmetic, never hard-coded, and the derived set is
    asserted against the closed form {0..2k-1} U {m-2k+1..m-1}."""
    S = list(range(k)) + list(range(m - k, m))
    ts = sorted({(i - j) % m for i in S for j in S})
    closed = sorted({t % m for t in list(range(0, 2 * k)) + list(range(m - 2 * k + 1, m))})
    assert ts == closed, ("needed_shifts derivation disagrees with the closed form: "
                          "%r vs %r" % (ts, closed))
    if m >= 4 * k:
        assert len(ts) == 4 * k - 1, ("expected 4k-1 = %d shifts at m >= 4k, got %d"
                                      % (4 * k - 1, len(ts)))
    return ts


def block_index_set(m, k):
    """S_blk in the order used everywhere: the first k and the last k block
    indices, so that the selected coordinates are the first kn and the last kn."""
    return list(range(k)) + list(range(m - k, m))


def accumulate_G(F, sym, m, k, mu, omega, all_shifts=False, stats=None,
                 X=None, Y=None, dft="direct", frequency_lags=None):
    """The blocks G_t = (1/m) sum_j omega^{+jt} Chat_j(mu)^{-1}.

    The unconditional direct route is streamed: for j = 0..m-1, invert the
    precomputed lag vector with cyclic-displacement (one live n x n inverse), accumulate it in
    each needed block, then discard it.  The optional radix-2 route stores the
    m inverses and transforms their entry sequences; its advertised workspace
    is therefore O(m n^2), not the direct route's O(k n^2).

    ``X,Y`` request the repair projections ``R_t=G_t X`` and
    ``L_t=Y^T G_t``.  No full all-shifts G array is materialised on either
    route: direct accumulates the projections while each inverse is live;
    radix-2 transforms the already projected inverse-entry sequences.

    `stats['det_C']` receives prod_j det Chat_j(mu): the determinant clause of
    T05 (b2) costs O(m D) and, per contract section 7, it is booked inside the
    `invert` stage rather than in a second pass over the symbols.

    raises SymbolSingular(j, mu) if any Chat_j is singular - the mu search
    treats that as 'mu is bad' and moves on."""
    n = sym.n
    route = transform_route(m, dft, k=k, phase="setup")
    if (X is None) != (Y is None):
        raise ValueError("X and Y projections must be supplied together")
    if frequency_lags is None:
        with F.cnt.stage("setup"):
            frequency_lags = frequency_symbol_lags(F, sym, m, mu, omega)
    if len(frequency_lags) != m:
        raise ValueError("frequency_lags must contain exactly m lag vectors")
    need = list(range(m)) if all_shifts else needed_shifts(m, k)
    acc = {t: zeros(F, n, n) for t in need}
    s = 0 if X is None else len(X[0])
    projected_R = {t: zeros(F, n, s) for t in range(m)} if s else None
    projected_L = {t: zeros(F, s, n) for t in range(m)} if s else None
    detC = F.one()
    cnt = F.cnt
    if route == "direct":

        
        weight_indices = list(range(m)) if s else need
        with cnt.stage("combine"):
            steps = {}
            if weight_indices == list(range(m)):
                step = F.one()
                for t in range(m):
                    steps[t] = step
                    if t + 1 < m:
                        step = F.mul(step, omega)
            else:
                steps = {t: F.pow(omega, t) for t in need}
            weights = {t: F.one() for t in weight_indices}
        for j, lag in enumerate(frequency_lags):
            with cnt.stage("invert"):
                d = toeplitz_det(F, lag, n)
                if F.is_zero(d):
                    raise SymbolSingular(j=j, mu=F.to_str(mu))
                detC = F.mul(detC, d)
                Xinv = toeplitz_inverse(F, lag, n)
                if s:
                    HX = mat_mul(F, Xinv, X)
                    YtH = zeros(F, s, n)
                    for c in range(s):
                        for b in range(n):
                            v = F.zero()
                            for a in range(n):
                                if (not F.is_zero(Y[a][c])
                                        and not F.is_zero(Xinv[a][b])):
                                    v = F.add(v, F.mul(Y[a][c], Xinv[a][b]))
                            YtH[c][b] = v
            with cnt.stage("combine"):
                for t in need:
                    w = weights[t]
                    A = acc[t]
                    if F.eq(w, F.one()):
                        for r in range(n):
                            Ar, Xr = A[r], Xinv[r]
                            for c in range(n):
                                Ar[c] = F.add(Ar[c], Xr[c])
                    else:
                        for r in range(n):
                            Ar, Xr = A[r], Xinv[r]
                            for c in range(n):
                                if not F.is_zero(Xr[c]):
                                    Ar[c] = F.add(Ar[c], F.mul(w, Xr[c]))
                if s:
                    for t in range(m):
                        w = weights[t]
                        Rt, Lt = projected_R[t], projected_L[t]
                        for a in range(n):
                            for c in range(s):
                                if not F.is_zero(HX[a][c]):
                                    Rt[a][c] = F.add(Rt[a][c], F.mul(w, HX[a][c]))
                        for c in range(s):
                            for b in range(n):
                                if not F.is_zero(YtH[c][b]):
                                    Lt[c][b] = F.add(Lt[c][b], F.mul(w, YtH[c][b]))
                if j + 1 < m:
                    for t in weight_indices:
                        weights[t] = F.mul(weights[t], steps[t])
            del Xinv                                 
        with cnt.stage("combine"):
            minv = F.inv(int_element(F, m))
            for t in need:
                A = acc[t]
                for r in range(n):
                    for c in range(n):
                        A[r][c] = F.mul(minv, A[r][c])
            if s:
                for t in range(m):
                    for a in range(n):
                        for c in range(s):
                            projected_R[t][a][c] = F.mul(minv, projected_R[t][a][c])
                    for c in range(s):
                        for b in range(n):
                            projected_L[t][c][b] = F.mul(minv, projected_L[t][c][b])
        peak_entries = ((len(need) + 1) * n * n
                        + (2 * m * n * s + 2 * n * s if s else 0)
                        + m * (2 * n - 1))
        peak_matrices = len(need) + 1 + (2 * m + 2 if s else 0)
    else:

        

        
        inverses = []
        hx_values = []
        yth_values = []
        for j, lag in enumerate(frequency_lags):
            with cnt.stage("invert"):
                d = toeplitz_det(F, lag, n)
                if F.is_zero(d):
                    raise SymbolSingular(j=j, mu=F.to_str(mu))
                detC = F.mul(detC, d)
                Xinv = toeplitz_inverse(F, lag, n)
                inverses.append(Xinv)
                if s:
                    hx_values.append(mat_mul(F, Xinv, X))
                    YtH = zeros(F, s, n)
                    for c in range(s):
                        for b in range(n):
                            v = F.zero()
                            for a in range(n):
                                if (not F.is_zero(Y[a][c])
                                        and not F.is_zero(Xinv[a][b])):
                                    v = F.add(v, F.mul(Y[a][c], Xinv[a][b]))
                            YtH[c][b] = v
                    yth_values.append(YtH)
        with cnt.stage("combine"):
            minv = F.inv(int_element(F, m))
            for a in range(n):
                for b in range(n):
                    transformed = _dft(F, [inverses[j][a][b] for j in range(m)],
                                       omega, route)
                    for t in need:
                        acc[t][a][b] = F.mul(minv, transformed[t])
            if s:
                for a in range(n):
                    for c in range(s):
                        transformed = _dft(
                            F, [hx_values[j][a][c] for j in range(m)], omega, route)
                        for t in range(m):
                            projected_R[t][a][c] = F.mul(minv, transformed[t])
                for c in range(s):
                    for b in range(n):
                        transformed = _dft(
                            F, [yth_values[j][c][b] for j in range(m)], omega, route)
                        for t in range(m):
                            projected_L[t][c][b] = F.mul(minv, transformed[t])
        peak_entries = ((m + len(need)) * n * n
                        + (4 * m * n * s if s else 0)
                        + m * (2 * n - 1))
        peak_matrices = m + len(need) + (4 * m if s else 0)
    if stats is not None:
        stats["det_C"] = detC
        stats["setup_transform_route"] = route
        stats["peak_live_matrices"] = peak_matrices
        stats["peak_live_matrix_entries"] = peak_entries
        stats["projection_mode"] = "G_tX/Y^TG_t" if s else "none"
        if s:
            stats["projections"] = {"R": projected_R, "L": projected_L}
    return acc


def corners(F, G, m, n, k):
    """(C^{-1})_{S,S} of size 2kn x 2kn, S the first kn and last kn coordinates.
    Pure copies from G by block (i,j) -> G[(i-j) mod m]: O(k^2 n^2), no arithmetic."""
    S = block_index_set(m, k)
    r = len(S) * n
    out = zeros(F, r, r)
    for p, i in enumerate(S):
        for qq, j in enumerate(S):
            Gt = G[(i - j) % m]
            for a in range(n):
                row = out[p * n + a]
                Ga = Gt[a]
                for b in range(n):
                    row[qq * n + b] = Ga[b]
    return out


def apply_C_inv(F, sym, m, mu, omega, v, transpose=False, dft="naive",
                frequency_lags=None):
    """C_mu^{-1} v, or C_mu^{-T} v, with O(mn+n^2) live memory.

    The two length-m block-transform arrays use O(mn) entries; symbol solves
    are still performed one frequency at a time.  Direct
    O(m^2) transforms support every m.  ``auto``/``radix2`` supplies the exact
    O(m log m) tier only for power-of-two m; unsupported explicit radix-2
    requests refuse instead of silently claiming a fast arbitrary-length DFT.

    transpose=True: C^T is block circulant with blocks A_{-t}^T, so its
    frequency-j symbol is (Chat_{(-j) mod m})^T.  Same lag vectors, same cost:
    this is what makes the transposed solve one more solve with NO new setup."""
    n = sym.n
    N = m * n
    if len(v) != N:
        raise ValueError("vector of length %d expected" % N)
    route = transform_route(m, dft, phase="apply")
    if frequency_lags is None:
        frequency_lags = frequency_symbol_lags(F, sym, m, mu, omega)
    if len(frequency_lags) != m:
        raise ValueError("frequency_lags must contain exactly m lag vectors")
    minv = F.inv(int_element(F, m))
    invomega = F.inv(omega)
    frequencies = [[F.zero()] * n for _ in range(m)]
    for a in range(n):
        transformed = _dft(F, [v[I * n + a] for I in range(m)], invomega, route)
        for j in range(m):
            frequencies[j][a] = F.mul(minv, transformed[j])
    solved = []
    for j in range(m):
        jj = (-j) % m if transpose else j
        lag = frequency_lags[jj]
        solved.append(toeplitz_solve(F, lag, n, frequencies[j], transpose=transpose))
    out = [F.zero()] * N
    for a in range(n):
        transformed = _dft(F, [solved[j][a] for j in range(m)], omega, route)
        for I in range(m):
            out[I * n + a] = transformed[I]
    return out
