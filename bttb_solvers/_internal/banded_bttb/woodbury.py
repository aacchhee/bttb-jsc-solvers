# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .errors import BandwidthOverlap, SingularInnerMatrix
from .circulant import block_index_set
from .toeplitz import zeros, mat_mul, identity


def wrap_columns(F, sym, m, mu=None):
    """U_S = V^T W V, of size 2kn x 2kn, and S, the selected coordinates.

    W = C - B is supported on the two off-corner k x k BLOCK triangles: the
    top-right one carries the blocks A_{+1..+k} and the bottom-left one the
    blocks A_{-1..-k}.  Both its nonzero block rows and its nonzero block
    columns lie in S_blk, so W = V U_S V^T exactly and U = W V = V U_S; the
    algorithm never needs U as a dense N x 2kn object.

    `mu` is the Lemma A scaling: W_mu = C_mu - B_mu has blocks mu^l A_l, so the
    lag vector of the block at outer lag l is scaled by mu^l here.  Forgetting
    that scaling is a silent wrong answer whenever mu != 1, and mu = 1 is the
    common case, so it is the exact kind of bug a test on easy instances misses.

    raises BandwidthOverlap when m < 2k+1: `-02` (d) proves that the
    two-triangle description AND the 2kn-column count both fail there.  Do not
    'handle' that case."""
    n, k = sym.n, sym.k
    if m < 2 * k + 1:
        raise BandwidthOverlap(m, k)
    S_blk = block_index_set(m, k)
    r = len(S_blk) * n
    US = zeros(F, r, r)
    for p, i in enumerate(S_blk):
        for qq, j in enumerate(S_blk):
            d = i - j
            if m - k <= d <= m - 1:
                l = d - m                     
            elif m - k <= -d <= m - 1:
                l = d + m                     
            else:
                continue
            lag = sym.block(l)
            if mu is not None:
                c = F.pow(mu, l)
                if not F.eq(c, F.one()):
                    lag = tuple(F.mul(c, v) for v in lag)
            for a in range(n):
                row = US[p * n + a]
                for b in range(n):
                    row[qq * n + b] = lag[(a - b) + (n - 1)]
    S = [i * n + a for i in S_blk for a in range(n)]
    return US, S


class MergedCorrection(object):
    """The columns of the ONE merged Woodbury step, as generators.

        main branch  : U' = V U_S,             V' = V,               r = 2kn
        repair branch: U' = [V U_S | I_m (x) X], V' = [V | I_m (x) Y], r = 2kn + m s

    Never materialised as a dense N x r matrix.  Unpacks as the contract's
    (apply_U, applyT_V, r)."""

    def __init__(self, F, m, n, k, US, S, X=None, Y=None):
        self.F, self.m, self.n, self.k = F, m, n, k
        self.US, self.S = US, S
        self.X, self.Y = X, Y
        self.s = 0 if X is None else len(X[0])
        self.r2 = 2 * k * n
        self.r = self.r2 + m * self.s

    def __iter__(self):
        return iter((self.apply_U, self.applyT_V, self.r))


    def apply_U(self, z):
        F, m, n = self.F, self.m, self.n
        out = [F.zero()] * (m * n)
        top = z[:self.r2]
        for a in range(self.r2):
            acc = F.zero()
            row = self.US[a]
            for b in range(self.r2):
                if not F.is_zero(row[b]) and not F.is_zero(top[b]):
                    acc = F.add(acc, F.mul(row[b], top[b]))
            out[self.S[a]] = acc
        if self.s:
            X, s = self.X, self.s
            for i in range(m):
                zi = z[self.r2 + i * s: self.r2 + (i + 1) * s]
                for a in range(n):
                    acc = out[i * n + a]
                    for c in range(s):
                        if not F.is_zero(X[a][c]) and not F.is_zero(zi[c]):
                            acc = F.add(acc, F.mul(X[a][c], zi[c]))
                    out[i * n + a] = acc
        return out


    def applyT_V(self, y):
        F, m, n = self.F, self.m, self.n
        out = [y[i] for i in self.S]
        if self.s:
            Y, s = self.Y, self.s
            for i in range(m):
                for c in range(s):
                    acc = F.zero()
                    for a in range(n):
                        if not F.is_zero(Y[a][c]) and not F.is_zero(y[i * n + a]):
                            acc = F.add(acc, F.mul(Y[a][c], y[i * n + a]))
                    out.append(acc)
        return out


    def apply_UT(self, y):
        F, m, n = self.F, self.m, self.n
        ys = [y[i] for i in self.S]
        out = []
        for b in range(self.r2):
            acc = F.zero()
            for a in range(self.r2):
                if not F.is_zero(self.US[a][b]) and not F.is_zero(ys[a]):
                    acc = F.add(acc, F.mul(self.US[a][b], ys[a]))
            out.append(acc)
        if self.s:
            X, s = self.X, self.s
            for i in range(m):
                for c in range(s):
                    acc = F.zero()
                    for a in range(n):
                        if not F.is_zero(X[a][c]) and not F.is_zero(y[i * n + a]):
                            acc = F.add(acc, F.mul(X[a][c], y[i * n + a]))
                    out.append(acc)
        return out


    def apply_V(self, z):
        F, m, n = self.F, self.m, self.n
        out = [F.zero()] * (m * n)
        for a in range(self.r2):
            out[self.S[a]] = z[a]
        if self.s:
            Y, s = self.Y, self.s
            for i in range(m):
                zi = z[self.r2 + i * s: self.r2 + (i + 1) * s]
                for a in range(n):
                    acc = out[i * n + a]
                    for c in range(s):
                        if not F.is_zero(Y[a][c]) and not F.is_zero(zi[c]):
                            acc = F.add(acc, F.mul(Y[a][c], zi[c]))
                    out[i * n + a] = acc
        return out


def merged_correction(F, sym, m, X=None, Y=None, mu=None):
    """Build the ONE merged correction.  Returns a MergedCorrection, which
    unpacks as the contract's (apply_U, applyT_V, r).  Two successive steps are
    not offered by this API at all, which is the intended way to make them
    impossible to write by accident."""
    US, S = wrap_columns(F, sym, m, mu)
    return MergedCorrection(F, m, sym.n, sym.k, US, S, X, Y)


def inner_matrix(F, G, m, n, k, corr, corners_SS, projections=None, stats=None):
    """M = I_r - V'^T C^{-1} U'.

    DEVIATION FROM THE CONTRACT SIGNATURE, deliberate and for ledger fidelity.
    The contract writes `inner_matrix(F, Cinv_apply, Uprime, Vprime, r)`.
    Applying `C^{-1}` to each of the r columns would cost O(r m n^2) = O(k m n^3)
    and would be a NEW DOMINANT TERM that the ledger of T05 (d) does not carry.
    Instead M is assembled from the blocks G_t, which is what line L4/L5 prices:
      - main branch, from the four corners only (`-02` (b)):   M = I - (C^{-1})_{S,S} U_S;
      - repair branch, the I_m (x) Y columns touch EVERY block row (`-02` Scope),
        but only the projections R_t=G_t X and L_t=Y^T G_t are retained.  This
        is algorithm's projected-block streaming assembly.
    Returns a dense r x r matrix.  The bound r <= 4kn is guaranteed on the
    nonsingular-B solve route by `-03` (f).  A singular diagnostic/certificate
    object can be larger and is not a Woodbury solve branch."""
    r2, s, r = corr.r2, corr.s, corr.r
    from .circulant import block_index_set
    S_blk = block_index_set(m, k)
    M = zeros(F, r, r)


    TL = mat_mul(F, corners_SS, corr.US)
    for a in range(r2):
        for b in range(r2):
            M[a][b] = F.sub(F.one() if a == b else F.zero(), TL[a][b])
    if not s:
        if stats is not None:
            stats["southeast_lag_product_slots"] = 0
            stats["southeast_copies"] = 0
        return M

    X, Y = corr.X, corr.Y
    if projections is not None:
        R, L = projections["R"], projections["L"]


        for p, i in enumerate(S_blk):
            for j in range(m):
                Rt = R[(i - j) % m]
                for a in range(n):
                    for c in range(s):
                        M[p * n + a][r2 + j * s + c] = F.neg(Rt[a][c])


        for i in range(m):
            for c in range(s):
                outrow = M[r2 + i * s + c]
                for qq, j in enumerate(S_blk):
                    Lt = L[(i - j) % m]
                    for a in range(n):
                        coeff = Lt[c][a]
                        if F.is_zero(coeff):
                            continue
                        usrow = corr.US[qq * n + a]
                        for b in range(r2):
                            if not F.is_zero(usrow[b]):
                                outrow[b] = F.sub(
                                    outrow[b], F.mul(coeff, usrow[b]))


        

        YtR = {}
        for t in range(m):
            Rt = R[t]
            block = zeros(F, s, s)
            for c in range(s):
                for c2 in range(s):
                    value = F.zero()
                    for a in range(n):
                        if not F.is_zero(Y[a][c]) and not F.is_zero(Rt[a][c2]):
                            value = F.add(value, F.mul(Y[a][c], Rt[a][c2]))
                    block[c][c2] = value
            YtR[t] = block
        if stats is not None:
            stats["southeast_lag_product_slots"] = m * n * s * s
            stats["southeast_copies"] = m * m * s * s
        for i in range(m):
            for j in range(m):
                block = YtR[(i - j) % m]
                for c in range(s):
                    for c2 in range(s):
                        value = F.neg(block[c][c2])
                        if i == j and c == c2:
                            value = F.add(F.one(), value)
                        M[r2 + i * s + c][r2 + j * s + c2] = value
        return M


    
    for p, i in enumerate(S_blk):
        for j in range(m):
            Gt = G[(i - j) % m]
            GX = mat_mul(F, Gt, X)                  
            for a in range(n):
                for c in range(s):
                    M[p * n + a][r2 + j * s + c] = F.neg(GX[a][c])

    P = zeros(F, m * s, r2)
    for i in range(m):
        for qq, j in enumerate(S_blk):
            Gt = G[(i - j) % m]
            for c in range(s):
                for b in range(n):
                    acc = F.zero()
                    for a in range(n):
                        if not F.is_zero(Y[a][c]) and not F.is_zero(Gt[a][b]):
                            acc = F.add(acc, F.mul(Y[a][c], Gt[a][b]))
                    P[i * s + c][qq * n + b] = acc
    BL = mat_mul(F, P, corr.US)
    for a in range(m * s):
        for b in range(r2):
            M[r2 + a][b] = F.neg(BL[a][b])

    for i in range(m):
        for j in range(m):
            Gt = G[(i - j) % m]
            YtG = zeros(F, s, n)
            for c in range(s):
                for b in range(n):
                    acc = F.zero()
                    for a in range(n):
                        if not F.is_zero(Y[a][c]) and not F.is_zero(Gt[a][b]):
                            acc = F.add(acc, F.mul(Y[a][c], Gt[a][b]))
                    YtG[c][b] = acc
            YGX = mat_mul(F, YtG, X)
            for c in range(s):
                for c2 in range(s):
                    v = F.neg(YGX[c][c2])
                    if i == j and c == c2:
                        v = F.add(F.one(), v)
                    M[r2 + i * s + c][r2 + j * s + c2] = v
    return M


class Factorisation(object):
    """Exact Gauss-Jordan with FULL pivoting over K: determinant plus both
    solves.  (The contract says 'LU with full pivoting'; Gauss-Jordan with full
    pivoting is the same O(r^3) elimination and additionally hands back the
    inverse, which makes solve and solve_transpose O(r^2) each.  Stated, not
    hidden.)"""

    def __init__(self, F, Minv, det, r):
        self.F, self.Minv, self.det, self.r = F, Minv, det, r

    def solve(self, b):
        from .toeplitz import mat_vec
        return mat_vec(self.F, self.Minv, b)

    def solve_transpose(self, b):
        from .toeplitz import mat_vec, mat_transpose
        return mat_vec(self.F, mat_transpose(self.Minv), b)


def factor(F, M):
    """raises SingularInnerMatrix: by `-02` (b), det M = det B / det C, so a
    singular M with C nonsingular CERTIFIES det B = 0.  That is a RESULT, and
    solver.solve() turns it into a clean report rather than a crash."""
    r = len(M)
    A = [list(row) + [F.one() if i == j else F.zero() for j in range(r)]
         for i, row in enumerate(M)]
    colperm = list(range(r))
    det = F.one()
    for c in range(r):
        piv = None
        for i in range(c, r):
            for j in range(c, r):
                if not F.is_zero(A[i][j]):
                    piv = (i, j)
                    break
            if piv:
                break
        if piv is None:
            raise SingularInnerMatrix(det_B_is_zero=True, r=r)
        pi, pj = piv
        if pi != c:
            A[c], A[pi] = A[pi], A[c]
            det = F.neg(det)
        if pj != c:
            for row in A:
                row[c], row[pj] = row[pj], row[c]
            colperm[c], colperm[pj] = colperm[pj], colperm[c]
            det = F.neg(det)
        det = F.mul(det, A[c][c])
        iv = F.inv(A[c][c])
        A[c] = [F.mul(x, iv) for x in A[c]]
        for i in range(r):
            if i != c and not F.is_zero(A[i][c]):
                f = A[i][c]
                A[i] = [F.sub(A[i][j], F.mul(f, A[c][j])) for j in range(2 * r)]

    Minv = [None] * r
    for c in range(r):
        Minv[colperm[c]] = A[c][r:]
    return Factorisation(F, Minv, det, r)
