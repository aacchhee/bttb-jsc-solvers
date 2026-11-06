# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.fast_linear import inverse_and_determinant
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata
from bttb_solvers._internal.halfstrip_core.uniform_terminal import make_multiplier, matrix_product
from bttb_solvers._internal.halfstrip_core.mosaic import run_mosaic, solve_mosaic


def _precision(P):
    if type(P) is not int or P < 1:
        raise ValueError('positive integer precision required')


def _at(F, series, d):
    return series[d] if 0 <= d < len(series) else F.zero()


def _zeros(F, rows, P):
    return [[F.zero() for _ in range(P)] for _ in range(rows)]


def _basis(F, n, i):
    out = _zeros(F, n, 1)
    out[i][0] = F.one()
    return out


class SpatialConvolution:
    """Explicit bivariate Kronecker products; uniform scalar fixed-64 leaves."""
    def __init__(self, F):
        self.F = F
        self.mul = make_multiplier(F)
        self.stats = {'convolution_calls': 0, 'largest_packed_input': 0,
                      'largest_spatial_product_cells': 0}

    def product(self, A, B, P):
        _precision(P)
        if not A or not B:
            return []
        F, stride = self.F, 2*P
        def pack(X):
            out = [F.zero() for _ in range(stride*(len(X)-1)+P)]
            for i, row in enumerate(X):
                for d in range(min(P, len(row))):
                    out[stride*i+d] = row[d]
            return out
        a, b = pack(A), pack(B)
        self.stats['convolution_calls'] += 1
        self.stats['largest_packed_input'] = max(
            self.stats['largest_packed_input'], len(a), len(b))
        result = self.mul(a, b)
        size = len(A)+len(B)-1
        self.stats['largest_spatial_product_cells'] = max(
            self.stats['largest_spatial_product_cells'], size*P)
        return [[result[stride*i+d] if stride*i+d < len(result) else F.zero()
                 for d in range(P)] for i in range(size)]

    def circular_correlation(self, A, B, P):
        if len(A) != len(B):
            raise ValueError('equal circular correlation dimensions required')
        n, F = len(A), self.F
        product = self.product(A, list(reversed(B)), P)
        return [[F.add(product[n-1+i][d],
                       product[i-1][d] if i else F.zero())
                 for d in range(P)] for i in range(n)]


class RawToeplitzLags:
    """O(nL) lag input: lags[d][n-1+i-j] = (F_d)[i,j]."""
    def __init__(self, F, lags, n, w):
        if type(n) is not int or type(w) is not int or n < 2 or not 1 <= w < n:
            raise ValueError('scope requires n >= 2 and 1 <= w < n')
        if not lags or len(lags) > n+w or any(len(row) != 2*n-1 for row in lags):
            raise ValueError('one to n+w lag vectors of length 2n-1 required')
        self.F, self.n, self.w, self.L = F, n, w, n+w
        self.lags = [list(row) for row in lags]
        self.convolution = SpatialConvolution(F)

    def lag_series(self, spatial_lag, P):
        _precision(P)
        if not -self.n < spatial_lag < self.n:
            raise ValueError('spatial lag outside Toeplitz support')
        return [self.lags[d][self.n-1+spatial_lag]
                if d < len(self.lags) else self.F.zero() for d in range(P)]

    def constant_matrix(self):
        n = self.n
        return [[self.lags[0][n-1+i-j] for j in range(n)] for i in range(n)]

    def apply(self, vector, P, *, transpose=False):
        _precision(P)
        if len(vector) != self.n:
            raise ValueError('one series per spatial coordinate required')
        spatial = [self.lag_series(i, P) for i in range(1-self.n, self.n)]
        if transpose:
            spatial.reverse()
        product = self.convolution.product(spatial, vector, P)
        return product[self.n-1:2*self.n-1]

    def cyclic_boundary(self, P):
        """NH-HC=G R^t, with columns/rows returned as two series vectors."""
        F, n = self.F, self.n
        g = _zeros(F, n, P)
        for i in range(1, n):
            a, b = self.lag_series(i-n, P), self.lag_series(i, P)
            g[i] = [F.sub(x, y) for x, y in zip(a, b)]
        r = [[F.neg(x) for x in self.lag_series(-j-1, P)]
             for j in range(n-1)]
        r.append([F.neg(x) for x in self.lag_series(0, P)])
        return [_basis(F, n, 0), g], [r, _basis(F, n, n-1)]


class CompactInverse:
    """Width-two cyclic response lift from algorithm(B), q=0 only."""
    def __init__(self, source, P):
        _precision(P)
        F, n = source.F, source.n
        self.source, self.F, self.n, self.precision = source, F, n, 1
        self.stats = {'constant_seed_order': n, 'response_width': 2,
                      'doubling_targets': [], 'selected_W_matrices': 0,
                      'selected_matrix_product': {}}
        B0, self.constant_determinant = inverse_and_determinant(
            F, source.constant_matrix())
        G, Rt = source.cyclic_boundary(P)
        self.X, self.Y = [], []
        for u in range(2):
            x, y = _zeros(F, n, 1), _zeros(F, n, 1)
            for i in range(n):
                for j in range(n):
                    x[i][0] = F.add(x[i][0], F.mul(B0[i][j], G[u][j][0]))
                    y[i][0] = F.add(y[i][0], F.mul(Rt[u][j][0], B0[j][i]))
            self.X.append(x)
            self.Y.append(y)

        del B0
        while self.precision < P:
            m, target = self.precision, min(2*self.precision, P)
            nextX, nextY = [], []
            for u in range(2):
                hx = source.apply(self.X[u], target)
                hty = source.apply(self.Y[u], target, transpose=True)
                rx, ry = _zeros(F, n, target-m), _zeros(F, n, target-m)
                for i in range(n):
                    for d in range(target):
                        a = F.sub(_at(F, G[u][i], d), hx[i][d])
                        b = F.sub(_at(F, Rt[u][i], d), hty[i][d])
                        if d < m:
                            if not F.is_zero(a) or not F.is_zero(b):
                                raise ArithmeticError('compact inverse residual lost divisibility')
                        else:
                            rx[i][d-m], ry[i][d-m] = a, b
                cx = self.apply(rx, target-m)
                cy = self.apply(ry, target-m, transpose=True)
                nextX.append([list(self.X[u][i])+list(cx[i]) for i in range(n)])
                nextY.append([list(self.Y[u][i])+list(cy[i]) for i in range(n)])
            self.X, self.Y, self.precision = nextX, nextY, target
            self.stats['doubling_targets'].append(target)

    def apply(self, vector, P, *, transpose=False):
        _precision(P)
        if P > self.precision or len(vector) != self.n:
            raise ValueError('inverse action exceeds precision or spatial dimension')
        F, n, conv = self.F, self.n, self.source.convolution
        out = _zeros(F, n, P)
        for u in range(2):
            rotated = self.X[u][1:]+self.X[u][:1]
            if transpose:
                d = conv.circular_correlation(rotated, vector, P)
                value = conv.product(self.Y[u], list(reversed(d)), P)[n-1:]
            else:
                a = conv.product(self.Y[u], list(reversed(vector)), P)[n-1:]
                value = conv.circular_correlation(rotated, a, P)
            for i in range(n):
                for d in range(P):
                    out[i][d] = F.sub(out[i][d], value[i][d])
        return out

    def selected_W(self, w):
        """Only R_c for n-w+1 <= c < n+w, via temporal flattening."""
        F, n, L = self.F, self.n, self.n+w
        if type(w) is not int or not 1 <= w < n or L > self.precision:
            raise ValueError('invalid selected window or insufficient inverse precision')
        e0, en = _basis(F, n, 0), _basis(F, n, n-1)
        g = _zeros(F, n, L)
        for i in range(1, n):
            g[i] = self.source.lag_series(i-n, L)
        h = [[F.neg(x) for x in self.source.lag_series(-j-1, L)]
             for j in range(n-1)] + [[F.zero() for _ in range(L)]]
        x, y = self.apply(e0, L), self.apply(g, L)
        row = self.apply(e0, L, transpose=True)
        v, t = self.apply(h, L, transpose=True), self.apply(en, L, transpose=True)
        left = [x[i]+y[i] for i in range(n)]
        selected = {}
        for c in range(n-w+1, L):
            right = [[_at(F, v[j], c-k) for j in range(n)] for k in range(L)]
            right += [[_at(F, t[j], c-k) for j in range(n)] for k in range(L)]
            D = matrix_product(F, left, right, stats=self.stats['selected_matrix_product'])
            coefficient = _zeros(F, n, n)
            coefficient[0] = [row[j][c] for j in range(n)]
            for i in range(1, n):
                coefficient[i][0] = x[i][c]
                for j in range(1, n):
                    coefficient[i][j] = F.add(coefficient[i-1][j-1], D[i][j-1])
            selected[c] = coefficient
        self.stats['selected_W_matrices'] = len(selected)
        return selected


def _power(F, value, exponent):
    result = F.one()
    while exponent:
        if exponent % 2:
            result = F.mul(result, value)
        exponent //= 2
        if exponent:
            value = F.mul(value, value)
    return result


def solve_q0(F, lags, n, w, rhs=None):
    """Restricted determinant and one arbitrary nonsingular RHS over supplied F.

    rhs and physical_solution are time-major n*n vectors. Singular F0 raises
    ZeroDivisionError; singular physical T returns det zero, solution None.
    No dense series source or physical n*n-by-n*n matrix is constructed.
    """
    if rhs is not None and len(rhs) != n*n:
        raise ValueError('RHS requires n*n physical time-major entries')
    source = RawToeplitzLags(F, lags, n, w)
    L = n+w

    
    inverse = CompactInverse(source, L)
    selected = inverse.selected_W(w)
    def coefficient(i, j, lag):
        c = n+lag
        if c not in selected:
            raise ValueError('terminal query outside selected W window')
        return selected[c][i][j]
    backend = run_mosaic(F, [w]*n, coefficient)
    gamma = _power(F, inverse.constant_determinant, L)
    if (n*n*n*w) % 2:
        gamma = F.neg(gamma)
    det = F.mul(gamma, backend['det'])
    singular = F.is_zero(det)
    result = {'det': det, 'singular': singular, 'physical_solution': None,
              'gamma': gamma, 'q': 0, 'inverse': inverse,
              'selected_W': selected, 'terminal': backend, 'metadata': metadata()}
    if rhs is not None and not singular:
        f = _zeros(F, n, L)
        for time in range(n):
            for i in range(n):
                f[i][time+w] = rhs[time*n+i]
        Rf = inverse.apply(f, L)

        terminal_rhs = [F.neg(Rf[i][n+t]) for i in range(n) for t in range(w)]
        affine = solve_mosaic(F, backend, terminal_rhs)
        u = affine['solution']
        if u is None:
            raise ArithmeticError('nonsingular terminal lacks affine readout')
        for i in range(n):
            for time in range(w):
                f[i][time] = F.add(f[i][time], u[i*w+time])
        answer = inverse.apply(f, L)
        if any(not F.is_zero(answer[i][time]) for i in range(n) for time in range(n, L)):
            raise ArithmeticError('q0 physical reconstruction has nonzero tail')
        result.update(physical_solution=[answer[i][time] for time in range(n) for i in range(n)],
                      terminal_rhs=terminal_rhs, affine=affine)
    return result
