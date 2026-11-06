# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
import itertools
from contextlib import contextmanager

import flint
from flint import nmod_mat, nmod_poly

from bttb_solvers._internal.banded_bttb import circulant, repair, symbol, toeplitz, woodbury
from bttb_solvers._internal.banded_bttb.errors import SingularInnerMatrix, SingularToeplitz, SymbolSingular, ZeroDivisionInField
from bttb_solvers._internal.banded_bttb.field import PrimeField


def _ok(F):
    return isinstance(F, PrimeField) and F.e == 1 and 2 <= F.p < (1 << 63)


def _toeplitz_flat(lag, n):
    """Row-major entries of T[i][j] = lag[i - j + n - 1] (slices of the reversed lag vector)."""
    rl = lag[::-1] if isinstance(lag, (tuple, list)) else tuple(reversed(lag))
    return list(itertools.chain.from_iterable(rl[n - 1 - i:2 * n - 1 - i] for i in range(n)))


def _toeplitz_mat(lag, n, p):
    return nmod_mat(n, n, _toeplitz_flat(lag, n), p)


def _mat(A, p):
    r = len(A)
    c = len(A[0]) if r else 0
    return nmod_mat(r, c, list(itertools.chain.from_iterable(A)), p)


def _lists(M):
    c = M.ncols()
    flat = [int(x) for x in M.entries()]
    return [flat[i * c:(i + 1) * c] for i in range(M.nrows())]


def _col(v, p):
    return nmod_mat(len(v), 1, list(v), p)


def _vec(M):
    return [int(x) for x in M.entries()]


def _poly(a, p):
    return nmod_poly(list(a), p)


def _pl(P):
    return [int(c) for c in P.coeffs()]


_orig = {}


def toeplitz_solve(F, lag, n, b, transpose=False):
    if not _ok(F):
        return _orig['toeplitz_solve'](F, lag, n, b, transpose)
    if len(b) != n:
        raise ValueError("right-hand side of length %d expected" % n)
    work = tuple(reversed(lag)) if transpose else tuple(lag)
    F.cnt.charge_solve(n)
    try:
        X = _toeplitz_mat(work, n, F.p).solve(_col(b, F.p))
    except ZeroDivisionError:
        raise SingularToeplitz(lag=tuple(lag), n=n)
    return _vec(X)


def toeplitz_inverse(F, lag, n):
    if not _ok(F):
        return _orig['toeplitz_inverse'](F, lag, n)
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    try:
        X = _toeplitz_mat(tuple(lag), n, F.p).inv()
    except ZeroDivisionError:
        raise SingularToeplitz(lag=tuple(lag), n=n)
    return _lists(X)


def toeplitz_det(F, lag, n):
    if not _ok(F):
        return _orig['toeplitz_det'](F, lag, n)
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    return int(_toeplitz_mat(tuple(lag), n, F.p).det())


def toeplitz_rank(F, lag, n):
    if not _ok(F):
        return _orig['toeplitz_rank'](F, lag, n)
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    return int(_toeplitz_mat(tuple(lag), n, F.p).rank())


def mat_mul(F, A, B):
    if not _ok(F) or not A or not B or not B[0]:
        return _orig['mat_mul'](F, A, B)
    return _lists(_mat(A, F.p) * _mat(B, F.p))


def gauss_det(F, A):
    if not _ok(F) or not A:
        return _orig['gauss_det'](F, A)
    return int(_mat(A, F.p).det())


def gauss_rank(F, A):
    if not _ok(F) or not A:
        return _orig['gauss_rank'](F, A)
    return int(_mat(A, F.p).rank())


_sym_cache = {}


def symbol_lag_at(F, sym, x):
    if not _ok(F):
        return _orig['symbol_lag_at'](F, sym, x)
    n, k, p = sym.n, sym.k, F.p
    key = id(sym)
    ent = _sym_cache.get(key)
    if ent is None or ent[0] is not sym:
        ent = (sym, [nmod_poly(list(v), p) for v in sym.lags])
        _sym_cache[key] = ent
    polys = ent[1]
    x = int(x) % p
    if k and x == 0:
        raise ZeroDivisionInField("inverse of 0 in F_%d" % p)
    xi = pow(x, p - 2, p) if k else 1
    acc = polys[k]
    pw = qw = 1
    for l in range(1, k + 1):
        pw = pw * x % p
        qw = qw * xi % p
        acc = acc + polys[k + l] * pw + polys[k - l] * qw
    out = _pl(acc)
    out += [0] * (2 * n - 1 - len(out))
    return tuple(out)


def poly_add(F, a, b):
    if not _ok(F):
        return _orig['poly_add'](F, a, b)
    return _pl(_poly(a, F.p) + _poly(b, F.p))


def poly_sub(F, a, b):
    if not _ok(F):
        return _orig['poly_sub'](F, a, b)
    return _pl(_poly(a, F.p) - _poly(b, F.p))


def poly_mul(F, a, b):
    if not _ok(F):
        return _orig['poly_mul'](F, a, b)
    if not a or not b:
        return []
    return _pl(_poly(a, F.p) * _poly(b, F.p))


def poly_divmod(F, a, b):
    if not _ok(F):
        return _orig['poly_divmod'](F, a, b)
    if not b:
        raise ZeroDivisionError("polynomial division by zero")
    q, r = divmod(_poly(a, F.p), _poly(b, F.p))
    return _pl(q), _pl(r)


def poly_gcd(F, a, b):
    if not _ok(F):
        return _orig['poly_gcd'](F, a, b)
    return _pl(_poly(a, F.p).gcd(_poly(b, F.p)))


def poly_powmod_x(F, e, f):
    if not _ok(F):
        return _orig['poly_powmod_x'](F, e, f)
    if len(f) <= 1:
        return []
    return _pl(nmod_poly([0, 1], F.p).pow_mod(e, _poly(f, F.p)))


def _interpolate(F, xs, ys):
    """Product-tree interpolation (the algorithm of polynomial.ProductTree.interpolate) over nmod_poly."""
    if not _ok(F):
        return _orig['_interpolate'](F, xs, ys)
    p = F.p
    N = len(xs)
    if N == 0:
        return []
    if len(ys) != N:
        raise ValueError('one value per node required')
    nodes = [nmod_poly([(-int(x)) % p, 1], p) for x in xs]

    def product(lo, hi):
        if hi - lo == 1:
            return nodes[lo]
        mid = (lo + hi) // 2
        return product(lo, mid) * product(mid, hi)

    M = product(0, N)
    dM = M.derivative()
    weights = []
    for x in xs:
        d = int(dM(int(x) % p))
        if d == 0:
            raise ValueError('interpolation nodes must be distinct')
        weights.append(pow(d, p - 2, p))

    def combine(lo, hi):
        """(sum_i c_i prod_{j != i} (z - x_j), prod (z - x_j)) over lo <= i < hi."""
        if hi - lo == 1:
            return nmod_poly([int(ys[lo]) * weights[lo] % p], p), nodes[lo]
        mid = (lo + hi) // 2
        a, pa = combine(lo, mid)
        b, pb = combine(mid, hi)
        return a * pb + b * pa, pa * pb

    return _pl(combine(0, N)[0])


_vander = {}


def _dft(F, values, root, route):
    if not _ok(F):
        return _orig['_dft'](F, values, root, route)
    p, m = F.p, len(values)
    key = (p, int(root) % p, m)
    V = _vander.get(key)
    if V is None:
        r = key[1]
        pw = [1] * ((m - 1) * (m - 1) + 1)
        for e in range(1, len(pw)):
            pw[e] = pw[e - 1] * r % p
        V = nmod_mat(m, m, [pw[i * j] for i in range(m) for j in range(m)], p)
        _vander[key] = V
    return _vec(V * _col(values, p))


def accumulate_G(F, sym, m, k, mu, omega, all_shifts=False, stats=None, X=None, Y=None, dft="direct",
                 frequency_lags=None):
    """circulant.accumulate_G, direct (streamed) route: same loop over frequencies, same outputs and stats."""
    n = sym.n
    route = circulant.transform_route(m, dft, k=k, phase="setup")
    if (X is None) != (Y is None):
        raise ValueError("X and Y projections must be supplied together")
    if route != "direct" or not _ok(F):
        return _orig['accumulate_G'](F, sym, m, k, mu, omega, all_shifts, stats, X, Y, dft, frequency_lags)
    p = F.p
    cnt = F.cnt
    if frequency_lags is None:
        with cnt.stage("setup"):
            frequency_lags = circulant.frequency_symbol_lags(F, sym, m, mu, omega)
    if len(frequency_lags) != m:
        raise ValueError("frequency_lags must contain exactly m lag vectors")
    need = list(range(m)) if all_shifts else circulant.needed_shifts(m, k)
    s = 0 if X is None else len(X[0])
    zero = nmod_mat(n, n, p)
    acc = {t: zero for t in need}
    if s:
        Xm, Ym = _mat(X, p), _mat(Y, p)
        YmT = Ym.transpose()
        zR, zL = nmod_mat(n, s, p), nmod_mat(s, n, p)
        pR = {t: zR for t in range(m)}
        pL = {t: zL for t in range(m)}
    detC = 1
    omega = int(omega) % p
    for j, lag in enumerate(frequency_lags):
        T = _toeplitz_mat(tuple(lag), n, p)
        d = int(T.det())
        if d == 0:
            raise SymbolSingular(j=j, mu=F.to_str(mu))
        detC = detC * d % p
        Xinv = T.inv()
        if s:
            HX = Xinv * Xm
            YtH = YmT * Xinv
        wj = pow(omega, j, p)           
        w = 1
        weights = {}
        for t in range(m):
            weights[t] = w
            w = w * wj % p
        for t in need:
            acc[t] = acc[t] + Xinv * weights[t]
        if s:
            for t in range(m):
                pR[t] = pR[t] + HX * weights[t]
                pL[t] = pL[t] + YtH * weights[t]
    minv = pow(m % p, p - 2, p)
    out = {t: _lists(acc[t] * minv) for t in need}
    if stats is not None:
        stats["det_C"] = detC
        stats["setup_transform_route"] = route
        stats["peak_live_matrices"] = len(need) + 1 + (2 * m + 2 if s else 0)
        stats["peak_live_matrix_entries"] = ((len(need) + 1) * n * n + (2 * m * n * s + 2 * n * s if s else 0)
                                             + m * (2 * n - 1))
        stats["projection_mode"] = "G_tX/Y^TG_t" if s else "none"
        if s:
            stats["projections"] = {"R": {t: _lists(pR[t] * minv) for t in range(m)},
                                    "L": {t: _lists(pL[t] * minv) for t in range(m)}}
    return out


class FlintFactorisation(object):
    """woodbury.Factorisation with nmod_mat storage of M^{-1}; same attributes, same results."""

    def __init__(self, F, Minv, det, r):
        self.F, self.det, self.r = F, det, r
        self._M = Minv
        self._MT = Minv.transpose()

    @property
    def Minv(self):
        return _lists(self._M)

    def solve(self, b):
        return _vec(self._M * _col(b, self.F.p))

    def solve_transpose(self, b):
        return _vec(self._MT * _col(b, self.F.p))


def factor(F, M):
    if not _ok(F):
        return _orig['factor'](F, M)
    r = len(M)
    A = _mat(M, F.p)
    d = int(A.det())
    if d == 0:
        raise SingularInnerMatrix(det_B_is_zero=True, r=r)
    return FlintFactorisation(F, A.inv(), d, r)


def _us_mat(self):
    mat = self.__dict__.get('_fk_US')
    if mat is None:
        mat = self.__dict__['_fk_US'] = _mat(self.US, self.F.p)
    return mat


def _apply_U(self, z):
    if self.s or not _ok(self.F):
        return _orig['apply_U'](self, z)
    F, r2 = self.F, self.r2
    res = _vec(_us_mat(self) * _col(z[:r2], F.p))
    out = [0] * (self.m * self.n)
    for a, v in zip(self.S, res):
        out[a] = v
    return out


def _apply_UT(self, y):
    if self.s or not _ok(self.F):
        return _orig['apply_UT'](self, y)
    F = self.F
    mat = self.__dict__.get('_fk_USt')
    if mat is None:
        mat = self.__dict__['_fk_USt'] = _us_mat(self).transpose()
    return _vec(mat * _col([y[i] for i in self.S], F.p))


def _kernel_bases(F, A):
    if not _ok(F) or not A:
        return _orig['_kernel_bases'](F, A)
    rows, cols = len(A), len(A[0])
    if rows != cols:
        raise ValueError("kernel-pairing matrix must be square")
    p = F.p
    aug = [list(A[i]) + [1 if i == j else 0 for j in range(rows)] for i in range(rows)]
    R, _rank = _mat(aug, p).rref()
    M = _lists(R)
    pivots = []
    for row in M:
        c = next((c for c in range(cols) if row[c]), None)
        if c is None:
            break
        pivots.append(c)
    free = [c for c in range(cols) if c not in pivots]
    right = []
    for fcol in free:
        v = [0] * cols
        v[fcol] = 1
        for row, pcol in reversed(list(enumerate(pivots))):
            v[pcol] = (-M[row][fcol]) % p
        right.append(v)
    left = [row[cols:] for row in M if not any(row[:cols])]
    return left, right


def make_multiplier(F, cutoff=64):
    if not _ok(F):
        return _orig['make_multiplier'](F) if cutoff == 64 else _orig['make_multiplier'](F, cutoff)
    p = F.p

    def multiply(a, b):
        if not a or not b:
            return []
        out = _pl(_poly(a, p) * _poly(b, p))
        out += [0] * (len(a) + len(b) - 1 - len(out))
        return out

    multiply.stats = {"kind": "flint-nmod_poly"}
    multiply.primitive_kind = "flint"
    return multiply


def _function_swaps():
    from bttb_solvers._internal.halfstrip import allfield_poly
    return [
        (toeplitz, 'toeplitz_solve', toeplitz_solve), (toeplitz, 'toeplitz_inverse', toeplitz_inverse),
        (toeplitz, 'toeplitz_det', toeplitz_det), (toeplitz, 'toeplitz_rank', toeplitz_rank),
        (toeplitz, 'mat_mul', mat_mul), (toeplitz, 'gauss_det', gauss_det), (toeplitz, 'gauss_rank', gauss_rank),
        (symbol, 'symbol_lag_at', symbol_lag_at), (symbol, 'poly_add', poly_add), (symbol, 'poly_sub', poly_sub),
        (symbol, 'poly_mul', poly_mul), (symbol, 'poly_divmod', poly_divmod), (symbol, 'poly_gcd', poly_gcd),
        (symbol, 'poly_powmod_x', poly_powmod_x), (symbol, '_interpolate', _interpolate),
        (circulant, '_dft', _dft), (circulant, 'accumulate_G', accumulate_G),
        (woodbury, 'factor', factor), (repair, '_kernel_bases', _kernel_bases),
        (allfield_poly, 'make_multiplier', make_multiplier),
    ]


def _solver_modules():
    import sys
    return [m for name, m in list(sys.modules.items())
            if m is not None and (name.startswith('bttb_solvers._internal.banded_bttb') or name == 'bttb_solvers._internal.cc_prepared'
                                  or name == 'bttb_solvers._internal.hb_physical_solver' or name == 'bttb_solvers._internal.halfstrip.allfield_poly')]


@contextmanager
def kernels():
    """Rebind the swapped kernels (by object identity, in every loaded solver module) for the `with` block."""
    swaps = _function_swaps()
    mods = _solver_modules()
    undo = []
    _orig.clear()
    for owner, name, repl in swaps:
        orig = getattr(owner, name)
        _orig[name] = orig
        for m in mods:
            for attr, val in list(vars(m).items()):
                if val is orig:
                    undo.append((m, attr, orig))
                    setattr(m, attr, repl)
    cls = woodbury.MergedCorrection
    methods = {'apply_U': _apply_U, 'apply_UT': _apply_UT}
    saved = {}
    for name, repl in methods.items():
        _orig[name] = saved[name] = cls.__dict__[name]
        setattr(cls, name, repl)
    try:
        yield sorted({(m.__name__, a) for m, a, _ in undo})
    finally:
        for m, attr, orig in undo:
            setattr(m, attr, orig)
        for name, orig in saved.items():
            setattr(cls, name, orig)
        _sym_cache.clear()
        _vander.clear()
