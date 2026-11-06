# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .errors import SingularToeplitz


def zeros(F, r, c):
    z = F.zero()
    return [[z] * c for _ in range(r)]


def identity(F, n):
    z, o = F.zero(), F.one()
    I = [[z] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = o
    return I


def mat_mul(F, A, B):
    n, p, q = len(A), len(B), len(B[0])
    C = zeros(F, n, q)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for t in range(p):
            a = Ai[t]
            if F.is_zero(a):
                continue
            Bt = B[t]
            for j in range(q):
                b = Bt[j]
                if F.is_zero(b):
                    continue
                Ci[j] = F.add(Ci[j], F.mul(a, b))
    return C


def mat_vec(F, A, v):
    out = []
    for row in A:
        acc = F.zero()
        for a, x in zip(row, v):
            if not F.is_zero(a) and not F.is_zero(x):
                acc = F.add(acc, F.mul(a, x))
        out.append(acc)
    return out


def mat_transpose(A):
    return [list(col) for col in zip(*A)]


def gauss_solve(F, A, b):
    """Solve A x = b by exact elimination with partial pivoting.  Returns the
    solution or None if A is singular.  All-branch: no minor is assumed."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if not F.is_zero(M[r][c]):
                piv = r
                break
        if piv is None:
            return None
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        iv = F.inv(M[c][c])
        M[c] = [F.mul(x, iv) for x in M[c]]
        for r in range(n):
            if r != c and not F.is_zero(M[r][c]):
                f = M[r][c]
                M[r] = [F.sub(M[r][j], F.mul(f, M[c][j])) for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def gauss_det(F, A):
    n = len(A)
    M = [list(r) for r in A]
    d = F.one()
    for c in range(n):
        piv = None
        for r in range(c, n):
            if not F.is_zero(M[r][c]):
                piv = r
                break
        if piv is None:
            return F.zero()
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            d = F.neg(d)
        d = F.mul(d, M[c][c])
        iv = F.inv(M[c][c])
        for r in range(c + 1, n):
            if not F.is_zero(M[r][c]):
                f = F.mul(M[r][c], iv)
                M[r] = [F.sub(M[r][j], F.mul(f, M[c][j])) for j in range(n)]
    return d


def gauss_rank(F, A):
    if not A:
        return 0
    M = [list(r) for r in A]
    rows, cols = len(M), len(M[0])
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if not F.is_zero(M[i][c]):
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        iv = F.inv(M[r][c])
        M[r] = [F.mul(x, iv) for x in M[r]]
        for i in range(rows):
            if i != r and not F.is_zero(M[i][c]):
                f = M[i][c]
                M[i] = [F.sub(M[i][j], F.mul(f, M[r][j])) for j in range(cols)]
        r += 1
        if r == rows:
            break
    return r


def toeplitz_dense(F, lag, n):
    """The n x n Toeplitz matrix T[i][j] = lag[(i-j) + (n-1)].  Copies only."""
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected, got %d" % (2 * n - 1, len(lag)))
    return [[lag[(i - j) + (n - 1)] for j in range(n)] for i in range(n)]


def lag_zero(F, n):
    return tuple([F.zero()] * (2 * n - 1))


def lag_add(F, u, v):
    return tuple(F.add(a, b) for a, b in zip(u, v))


def lag_scale(F, c, u):
    return tuple(F.mul(c, a) for a in u)


def _ordbasis(F, lag, n, b):
    """algorithm ORDBASIS for ``(alpha, -1, -z^(n-1)b)``.

    Polynomial coefficient arrays are low-index first.  They are deliberately
    fixed-width: every update is a scalar axpy or a one-place shift, so the
    implementation mirrors the eight-line proof algorithm and uses O(n) space.
    """
    sigma = 2 * n - 1
    width = sigma + 1
    z = F.zero()


    p = [[[z] * width for _ in range(3)] for _ in range(3)]
    for i in range(3):
        p[i][i][0] = F.one()


    R = [[z] * sigma for _ in range(3)]
    for j, a in enumerate(lag):
        R[0][j] = a
    R[1][0] = F.neg(F.one())
    for j, a in enumerate(b):
        R[2][n - 1 + j] = F.neg(a)


    defects = [-(n - 1), -(n - 2), 0]
    pivots = []
    pivot_values = []
    for order in range(sigma):
        residual = [R[i][order] for i in range(3)]
        nz = [i for i in range(3) if not F.is_zero(residual[i])]
        if not nz:
            continue
        pivot = min(nz, key=lambda i: (defects[i], i))
        rp = residual[pivot]
        inv_rp = F.inv(rp)
        pivots.append(pivot)
        pivot_values.append(rp)


        for i in nz:
            if i == pivot:
                continue
            c = F.mul(residual[i], inv_rp)
            for component in range(3):
                pi, pp = p[i][component], p[pivot][component]
                for degree in range(width):
                    if not F.is_zero(pp[degree]):
                        pi[degree] = F.sub(pi[degree], F.mul(c, pp[degree]))
            for degree in range(order, sigma):
                if not F.is_zero(R[pivot][degree]):
                    R[i][degree] = F.sub(
                        R[i][degree], F.mul(c, R[pivot][degree]))


        for component in range(3):
            old = p[pivot][component]
            p[pivot][component] = [z] + old[:-1]
        R[pivot] = [z] + R[pivot][:-1]
        defects[pivot] += 1
    return p, defects, pivots, pivot_values


def _ordbasis_solution(F, lag, n, b):
    p, defects, _pivots, _values = _ordbasis(F, lag, n, b)
    for i in range(3):
        if defects[i] > 0:
            continue
        p3 = p[i][2]
        if all(F.is_zero(x) for x in p3):
            continue
        w = p3[0]
        if F.is_zero(w):
            continue
        iw = F.inv(w)
        return [F.mul(p[i][0][j], iw) for j in range(n)]
    return None


def _poly_trim(F, a):
    i = len(a)
    while i and F.is_zero(a[i - 1]):
        i -= 1
    return list(a[:i])


def _poly_remainder(F, u, v):
    """Schoolbook ``u mod v`` with the leading cancellation omitted."""
    u = _poly_trim(F, u)
    v = _poly_trim(F, v)
    if not v:
        raise ZeroDivisionError("polynomial division by zero")
    dv = len(v) - 1
    iv = F.inv(v[-1])
    for degree in range(len(u) - 1, dv - 1, -1):
        if F.is_zero(u[degree]):
            continue
        c = F.mul(u[degree], iv)
        shift = degree - dv

        u[degree] = F.zero()
        for j in range(dv):
            if not F.is_zero(v[j]):
                u[shift + j] = F.sub(u[shift + j], F.mul(c, v[j]))
    return _poly_trim(F, u)


def toeplitz_solve(F, lag, n, b, transpose=False):
    """Solve T x = b (or T^T x = b), T the n x n Toeplitz matrix of `lag`.

    ALL-BRANCH: no leading, central or trailing minor is assumed nonsingular.
    raises SingularToeplitz iff det T == 0.

    Cost contract: O(n^2) measured field operations by algorithm R-1'.  The
    historical model charge is retained as a separate checked column and now
    agrees asymptotically with the executed primitive."""
    if len(b) != n:
        raise ValueError("right-hand side of length %d expected" % n)
    work_lag = tuple(reversed(lag)) if transpose else tuple(lag)
    F.cnt.charge_solve(n)

    
    if F.is_zero(toeplitz_det(F, work_lag, n)):
        raise SingularToeplitz(lag=tuple(lag), n=n)
    x = _ordbasis_solution(F, work_lag, n, list(b))
    if x is None:
        raise SingularToeplitz(lag=tuple(lag), n=n)
    return x


def _cyc_shift(F, v, n):
    """Z_1 v, Z_1 the CYCLIC down shift: (Z_1 v)[i] = v[(i-1) mod n]."""
    return [v[(i - 1) % n] for i in range(n)]


def toeplitz_inverse(F, lag, n):
    """All n^2 entries of T^{-1} by cyclic-displacement.  Branch-free, division-free expansion.

    EXACTLY FOUR toeplitz_solve calls on every branch (two with T, two with
    T^T), then the O(n^2) reconstruction of X from D(X) = X - Z_1 X Z^T.
    raises SingularToeplitz.  Post-condition T X == I is checked in tests, never
    on the hot path."""
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))

    def a(l):
        return lag[l + (n - 1)]


    R = zeros(F, n, n)
    for i in range(n):
        for j in range(n):
            if i == n - 1:
                v = F.zero()
            elif j == n - 1:
                v = a((i + 1) - 0)           
            else:
                v = a((i + 1) - (j + 1))
            R[i][j] = F.sub(a(i - j), v)
    r = list(R[n - 1])
    c = [R[i][n - 1] for i in range(n)]
    c[n - 1] = F.zero()


    e0 = [F.zero()] * n
    e0[0] = F.one()
    en = [F.zero()] * n
    en[n - 1] = F.one()
    al = toeplitz_solve(F, lag, n, en)                                  
    ga = toeplitz_solve(F, lag, n, c)                                   
    be = toeplitz_solve(F, lag, n, _cyc_shift(F, r, n), transpose=True)  
    de = toeplitz_solve(F, lag, n, e0, transpose=True)                   

    g1, h1 = _cyc_shift(F, al, n), be
    g2, h2 = _cyc_shift(F, ga, n), de
    D = zeros(F, n, n)
    for i in range(n):
        for j in range(n):
            D[i][j] = F.add(F.mul(g1[i], h1[j]), F.mul(g2[i], h2[j]))


    X = zeros(F, n, n)
    for i in range(n):
        X[i][0] = D[i][0]
    for j in range(1, n):
        X[0][j] = F.add(X[n - 1][j - 1], D[0][j])
        for i in range(1, n):
            X[i][j] = F.add(X[i - 1][j - 1], D[i][j])
    return X


def toeplitz_det(F, lag, n):
    """det T, exactly; may be zero.

    algorithm R-2', route A: defective Euclidean remainder chain, including the
    exact leading-coefficient exponents and sign.  O(n^2), O(n) workspace."""
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    sigma, j = 2 * n - 1, n - 1
    alpha = _poly_trim(F, lag)
    if not alpha:
        return F.zero()
    n1 = len(alpha) - 1
    if n1 < j:
        return F.zero()

    fprev = [F.zero()] * sigma + [F.one()]  
    fcur = alpha
    degrees = [sigma, n1]
    leads = [F.one(), fcur[-1]]
    while degrees[-1] > j:
        rem = _poly_remainder(F, fprev, fcur)
        degree = len(rem) - 1
        degrees.append(degree)
        leads.append(rem[-1] if rem else F.zero())
        fprev, fcur = fcur, rem
        if degree < j:
            break

    mu = len(degrees) - 1 if degrees[-1] == j else len(degrees) - 2
    if mu < 1 or degrees[mu] != j:
        return F.zero()

    out = F.one()
    parity = 0
    for i in range(1, mu):
        exponent = degrees[i - 1] - degrees[i + 1]
        for _ in range(exponent):
            out = F.mul(out, leads[i])
        parity ^= ((degrees[i - 1] - j) * (degrees[i] - j)) & 1
    exponent = degrees[mu - 1] - n + 1
    for _ in range(exponent):
        out = F.mul(out, leads[mu])
    return F.neg(out) if parity else out


def toeplitz_rank(F, lag, n):
    """algorithm RK-1: all-branch rank in O(n^2) field operations."""
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    p, defects, _pivots, _values = _ordbasis(F, tuple(lag), n, [F.zero()] * n)
    del p
    nullity = sum(1 - defects[i] for i in (0, 1) if defects[i] <= 0)
    return n - nullity
