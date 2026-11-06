# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Main CC setup and queries.

Role: Shared. Scoped extension-field matrix and polynomial kernels, including regular-representation matrix arithmetic.

Entry points: kernels.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
import itertools
from contextlib import contextmanager

from flint import nmod_mat, fq_default

from bttb_solvers._internal.banded_bttb import circulant, repair, symbol, toeplitz, woodbury
from bttb_solvers._internal.banded_bttb.errors import SingularInnerMatrix, SingularToeplitz, SymbolSingular, ZeroDivisionInField
from bttb_solvers._internal import extension_field

_orig = {}
_cache = {}


def _ok(F):
    return isinstance(F, extension_field.FqField)


def _tab(F):
    t = _cache.get(('T', id(F)))
    if t is None or t[0] is not F:
        d, z = F.e, F.ctx.gen()
        pw = [[int(c) for c in (z ** e).to_list()] for e in range(2 * d - 1)]

        t = (F, [[pw[a + b] for b in range(d)] for a in range(d)])
        _cache[('T', id(F))] = t
    return t[1]


class XM(object):
    """F_q matrix as d nmod_mats over F_p."""
    __slots__ = ('F', 'c', 'r', 'k')

    def __init__(self, F, comps, r, k):
        self.F, self.c, self.r, self.k = F, comps, r, k


def _coords(F, elems):
    """list of fq elements -> d lists of ints."""
    d = F.e
    lists = [e.to_list() for e in elems]
    return [[int(l[a]) for l in lists] for a in range(d)]


def xm_from_lists(F, A):
    r = len(A)
    k = len(A[0]) if r else 0
    flat = list(itertools.chain.from_iterable(A))
    return XM(F, [nmod_mat(r, k, col, F.p) for col in _coords(F, flat)], r, k)


def xm_from_vec(F, v):
    return XM(F, [nmod_mat(len(v), 1, col, F.p) for col in _coords(F, v)], len(v), 1)


def xm_toeplitz(F, lag, n):
    out = []
    for col in _coords(F, lag):
        rl = col[::-1]
        out.append(nmod_mat(n, n, list(itertools.chain.from_iterable(rl[n - 1 - i:2 * n - 1 - i] for i in range(n))), F.p))
    return XM(F, out, n, n)


def xm_lists(X):
    F, d = X.F, X.F.e
    cols = [[int(x) for x in M.entries()] for M in X.c]
    ctx = F.ctx
    flat = [ctx(list(t)) for t in zip(*cols)]
    k = X.k
    return [flat[i * k:(i + 1) * k] for i in range(X.r)]


def xm_vec(X):
    F = X.F
    cols = [[int(x) for x in M.entries()] for M in X.c]
    return [F.ctx(list(t)) for t in zip(*cols)]


def _combine(F, X_by_a, coef):
    """sum_a coef[a] * X_a (coef: ints mod p); None if all zero."""
    acc = None
    for a, c in enumerate(coef):
        if c:
            term = X_by_a[a] * c
            acc = term if acc is None else acc + term
    return acc


def xm_mul(A, B):
    F, d, p = A.F, A.F.e, A.F.p
    T = _tab(F)
    out = [None] * d
    for a in range(d):
        for b in range(d):
            prod = None
            for c in range(d):
                if T[a][b][c]:
                    if prod is None:
                        prod = A.c[a] * B.c[b]
                    term = prod * T[a][b][c] if T[a][b][c] != 1 else prod
                    out[c] = term if out[c] is None else out[c] + term
    zero = nmod_mat(A.r, B.k, p)
    return XM(F, [o if o is not None else zero for o in out], A.r, B.k)


def xm_add(A, B):
    return XM(A.F, [x + y for x, y in zip(A.c, B.c)], A.r, A.k)


def xm_scale(A, w):
    """A * w, w an fq element (scalar)."""
    F, d = A.F, A.F.e
    T = _tab(F)
    wl = [int(c) for c in w.to_list()]
    out = [None] * d
    for a in range(d):
        for b in range(d):
            if wl[b]:
                for c in range(d):
                    if T[a][b][c]:
                        term = A.c[a] * (wl[b] * T[a][b][c] % F.p)
                        out[c] = term if out[c] is None else out[c] + term
    zero = nmod_mat(A.r, A.k, F.p)
    return XM(F, [o if o is not None else zero for o in out], A.r, A.k)


def xm_transpose(A):
    return XM(A.F, [M.transpose() for M in A.c], A.k, A.r)


def xm_zero(F, r, k):
    z = nmod_mat(r, k, F.p)
    return XM(F, [z] * F.e, r, k)


def xm_K(X):
    """regular representation: d*r x d*k nmod_mat, coordinate-major blocks (c, b) = sum_a T[a][b][c] M_a."""
    F, d, p = X.F, X.F.e, X.F.p
    T = _tab(F)
    r, k = X.r, X.k
    ent = [[None] * d for _ in range(d)]
    for c in range(d):
        for b in range(d):
            blk = _combine(F, X.c, [T[a][b][c] for a in range(d)])
            ent[c][b] = [int(x) for x in blk.entries()] if blk is not None else [0] * (r * k)
    flat = []
    for c in range(d):
        for i in range(r):
            for b in range(d):
                flat.extend(ent[c][b][i * k:(i + 1) * k])
    return nmod_mat(d * r, d * k, flat, p)


def _stacked_identity(F, r):
    """d*r x r : (I_r; 0; ...; 0)."""
    flat = [0] * (F.e * r * r)
    for i in range(r):
        flat[i * r + i] = 1
    return nmod_mat(F.e * r, r, flat, F.p)


def xm_inverse(X):
    """M^-1 as XM, or raise ZeroDivisionError if singular (one K.solve with r right-hand sides)."""
    F, r = X.F, X.r
    Z = xm_K(X).solve(_stacked_identity(F, r))
    flat = [int(x) for x in Z.entries()]
    return XM(F, [nmod_mat(r, r, flat[c * r * r:(c + 1) * r * r], F.p) for c in range(F.e)], r, r)


def _vec_from_K_solution(F, X, n):
    flat = [int(x) for x in X.entries()]
    return [F.ctx([flat[b * n + i] for b in range(F.e)]) for i in range(n)]


def _vec_col(F, b):
    n = len(b)
    cols = _coords(F, b)
    return nmod_mat(F.e * n, 1, list(itertools.chain.from_iterable(cols)), F.p)


def toeplitz_solve(F, lag, n, b, transpose=False):
    if not _ok(F):
        return _orig['toeplitz_solve'](F, lag, n, b, transpose)
    if len(b) != n:
        raise ValueError("right-hand side of length %d expected" % n)
    work = tuple(reversed(lag)) if transpose else tuple(lag)
    F.cnt.charge_solve(n)
    try:
        x = xm_K(xm_toeplitz(F, work, n)).solve(_vec_col(F, b))
    except ZeroDivisionError:
        raise SingularToeplitz(lag=tuple(lag), n=n)
    return _vec_from_K_solution(F, x, n)


def toeplitz_inverse(F, lag, n):
    if not _ok(F):
        return _orig['toeplitz_inverse'](F, lag, n)
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    try:
        return xm_lists(xm_inverse(xm_toeplitz(F, tuple(lag), n)))
    except ZeroDivisionError:
        raise SingularToeplitz(lag=tuple(lag), n=n)


def toeplitz_rank(F, lag, n):
    if not _ok(F):
        return _orig['toeplitz_rank'](F, lag, n)
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    return int(xm_K(xm_toeplitz(F, tuple(lag), n)).rank()) // F.e


def _det_d2(F, lag, n):
    """det of the n x n Toeplitz matrix over GF(p^2) (EXACT value, not the norm).  M = T0 + z T1 (coordinate matrices
    over F_p).  Pick s in F_p with N = T0 + s T1 invertible (D(t) = det(T0 + t T1) has degree <= n, so s <= n works unless
    D = 0 identically, in which case det M = D(z) = 0).  Then det M = det N * det(I + (z-s) C), C = N^-1 T1, and
    det(I + u C) = sum_i c_i (-u)^i for the characteristic polynomial chi_C(x) = sum_i c_i x^(n-i) (c_0 = 1)."""
    X = xm_toeplitz(F, tuple(lag), n)
    A, B = X.c
    for s in range(n + 2):
        N = A + B * s if s else A
        try:
            Ninv = N.inv()
        except ZeroDivisionError:
            continue
        cf = [int(c) for c in (Ninv * B).charpoly().coeffs()]      
        v = -(F.ctx.gen() - s)                                       
        acc = F.zero()
        for i in range(n, -1, -1):
            acc = acc * v + cf[n - i]
        return acc * int(N.det())
    return F.zero()


def toeplitz_det(F, lag, n):
    if not _ok(F) or F.e != 2:
        return _orig['toeplitz_det'](F, lag, n)       
    if len(lag) != 2 * n - 1:
        raise ValueError("lag vector of length %d expected" % (2 * n - 1))
    return _det_d2(F, lag, n)


def mat_mul(F, A, B):
    if not _ok(F) or not A or not B or not B[0]:
        return _orig['mat_mul'](F, A, B)
    return xm_lists(xm_mul(xm_from_lists(F, A), xm_from_lists(F, B)))


def gauss_rank(F, A):
    if not _ok(F) or not A:
        return _orig['gauss_rank'](F, A)
    return int(xm_K(xm_from_lists(F, A)).rank()) // F.e


_sym_cache = {}


def _pl(P):
    return list(P.coeffs())


def symbol_lag_at(F, sym, x):
    if not _ok(F):
        return _orig['symbol_lag_at'](F, sym, x)
    n, k = sym.n, sym.k
    key = id(sym)
    ent = _sym_cache.get(key)
    if ent is None or ent[0] is not sym:
        ent = (sym, [F.pctx(list(v)) for v in sym.lags])
        _sym_cache[key] = ent
    polys = ent[1]
    if k and F.is_zero(x):
        raise ZeroDivisionInField("inverse of 0 in %s" % F.name())
    xi = F.inv(x) if k else F.one()
    acc = polys[k]
    pw = qw = F.one()
    for l in range(1, k + 1):
        pw = pw * x
        qw = qw * xi
        acc = acc + polys[k + l] * pw + polys[k - l] * qw
    out = _pl(acc)
    out += [F.zero()] * (2 * n - 1 - len(out))
    return tuple(out)


def poly_add(F, a, b):
    if not _ok(F):
        return _orig['poly_add'](F, a, b)
    return _pl(F.pctx(list(a)) + F.pctx(list(b)))


def poly_sub(F, a, b):
    if not _ok(F):
        return _orig['poly_sub'](F, a, b)
    return _pl(F.pctx(list(a)) - F.pctx(list(b)))


def poly_mul(F, a, b):
    if not _ok(F):
        return _orig['poly_mul'](F, a, b)
    if not a or not b:
        return []
    return _pl(F.pctx(list(a)) * F.pctx(list(b)))


def poly_divmod(F, a, b):
    if not _ok(F):
        return _orig['poly_divmod'](F, a, b)
    if not b:
        raise ZeroDivisionError("polynomial division by zero")
    q, r = divmod(F.pctx(list(a)), F.pctx(list(b)))
    return _pl(q), _pl(r)


def poly_gcd(F, a, b):
    if not _ok(F):
        return _orig['poly_gcd'](F, a, b)
    return _pl(F.pctx(list(a)).gcd(F.pctx(list(b))))


def poly_powmod_x(F, e, f):
    if not _ok(F):
        return _orig['poly_powmod_x'](F, e, f)
    if len(f) <= 1:
        return []
    return _pl(F.pctx([0, 1]).pow_mod(e, F.pctx(list(f))))


def _interpolate(F, xs, ys):
    if not _ok(F):
        return _orig['_interpolate'](F, xs, ys)
    N = len(xs)
    if N == 0:
        return []
    if len(ys) != N:
        raise ValueError('one value per node required')
    nodes = [F.pctx([-x, F.one()]) for x in xs]

    def product(lo, hi):
        if hi - lo == 1:
            return nodes[lo]
        mid = (lo + hi) // 2
        return product(lo, mid) * product(mid, hi)

    dM = product(0, N).derivative()
    weights = []
    for x in xs:
        dv = dM(x)
        if F.is_zero(dv):
            raise ValueError('interpolation nodes must be distinct')
        weights.append(1 / dv)

    def combine(lo, hi):
        if hi - lo == 1:
            return F.pctx([ys[lo] * weights[lo]]), nodes[lo]
        mid = (lo + hi) // 2
        a, pa = combine(lo, mid)
        b, pb = combine(mid, hi)
        return a * pb + b * pa, pa * pb

    return _pl(combine(0, N)[0])


def _dft(F, values, root, route):
    if not _ok(F):
        return _orig['_dft'](F, values, root, route)
    m = len(values)
    key = ('V', id(F), root.to_list().__repr__(), m)
    V = _cache.get(key)
    if V is None:
        pw = [F.one()] * ((m - 1) * (m - 1) + 1)
        for e in range(1, len(pw)):
            pw[e] = pw[e - 1] * root
        V = xm_from_lists(F, [[pw[i * j] for j in range(m)] for i in range(m)])
        _cache[key] = V
    return xm_vec(xm_mul(V, xm_from_vec(F, values)))


def accumulate_G(F, sym, m, k, mu, omega, all_shifts=False, stats=None, X=None, Y=None, dft="direct",
                 frequency_lags=None):
    n = sym.n
    route = circulant.transform_route(m, dft, k=k, phase="setup")
    if (X is None) != (Y is None):
        raise ValueError("X and Y projections must be supplied together")
    if route != "direct" or not _ok(F):
        return _orig['accumulate_G'](F, sym, m, k, mu, omega, all_shifts, stats, X, Y, dft, frequency_lags)
    cnt = F.cnt
    if frequency_lags is None:
        with cnt.stage("setup"):
            frequency_lags = circulant.frequency_symbol_lags(F, sym, m, mu, omega)
    if len(frequency_lags) != m:
        raise ValueError("frequency_lags must contain exactly m lag vectors")
    need = list(range(m)) if all_shifts else circulant.needed_shifts(m, k)
    s = 0 if X is None else len(X[0])
    acc = {t: xm_zero(F, n, n) for t in need}
    if s:
        Xm, Ym = xm_from_lists(F, X), xm_from_lists(F, Y)
        YmT = xm_transpose(Ym)
        pR = {t: xm_zero(F, n, s) for t in range(m)}
        pL = {t: xm_zero(F, s, n) for t in range(m)}
    for j, lag in enumerate(frequency_lags):
        try:
            Xinv = xm_inverse(xm_toeplitz(F, tuple(lag), n))
        except ZeroDivisionError:
            raise SymbolSingular(j=j, mu=F.to_str(mu))
        if s:
            HX = xm_mul(Xinv, Xm)
            YtH = xm_mul(YmT, Xinv)
        wj = omega ** j
        w = F.one()
        weights = {}
        for t in range(m):
            weights[t] = w
            w = w * wj
        for t in need:
            acc[t] = xm_add(acc[t], xm_scale(Xinv, weights[t]))
        if s:
            for t in range(m):
                pR[t] = xm_add(pR[t], xm_scale(HX, weights[t]))
                pL[t] = xm_add(pL[t], xm_scale(YtH, weights[t]))
    minv = F.inv(extension_field_int(F, m))
    out = {t: xm_lists(xm_scale(acc[t], minv)) for t in need}
    if stats is not None:
        stats["det_C"] = None                       
        stats["setup_transform_route"] = route
        stats["projection_mode"] = "G_tX/Y^TG_t" if s else "none"
        if s:
            stats["projections"] = {"R": {t: xm_lists(xm_scale(pR[t], minv)) for t in range(m)},
                                    "L": {t: xm_lists(xm_scale(pL[t], minv)) for t in range(m)}}
    return out


def extension_field_int(F, i):
    return F.ctx(int(i))


class FlintFactorisation(object):
    def __init__(self, F, Minv, r):
        self.F, self.det, self.r = F, None, r
        self._M = Minv
        self._MT = xm_transpose(Minv)

    @property
    def Minv(self):
        return xm_lists(self._M)

    def solve(self, b):
        return xm_vec(xm_mul(self._M, xm_from_vec(self.F, b)))

    def solve_transpose(self, b):
        return xm_vec(xm_mul(self._MT, xm_from_vec(self.F, b)))


def factor(F, M):
    if not _ok(F):
        return _orig['factor'](F, M)
    r = len(M)
    try:
        inv = xm_inverse(xm_from_lists(F, M))
    except ZeroDivisionError:
        raise SingularInnerMatrix(det_B_is_zero=True, r=r)
    return FlintFactorisation(F, inv, r)


def _us_mat(self):
    mat = self.__dict__.get('_fk_US')
    if mat is None:
        mat = self.__dict__['_fk_US'] = xm_from_lists(self.F, self.US)
    return mat


def _apply_U(self, z):
    if self.s or not _ok(self.F):
        return _orig['apply_U'](self, z)
    F, r2 = self.F, self.r2
    res = xm_vec(xm_mul(_us_mat(self), xm_from_vec(F, z[:r2])))
    out = [F.zero()] * (self.m * self.n)
    for a, v in zip(self.S, res):
        out[a] = v
    return out


def _apply_UT(self, y):
    if self.s or not _ok(self.F):
        return _orig['apply_UT'](self, y)
    F = self.F
    mat = self.__dict__.get('_fk_USt')
    if mat is None:
        mat = self.__dict__['_fk_USt'] = xm_transpose(_us_mat(self))
    return xm_vec(xm_mul(mat, xm_from_vec(F, [y[i] for i in self.S])))


def _function_swaps():
    return [
        (toeplitz, 'toeplitz_solve', toeplitz_solve), (toeplitz, 'toeplitz_inverse', toeplitz_inverse),
        (toeplitz, 'toeplitz_rank', toeplitz_rank), (toeplitz, 'toeplitz_det', toeplitz_det),
        (toeplitz, 'mat_mul', mat_mul),
        (toeplitz, 'gauss_rank', gauss_rank),
        (symbol, 'symbol_lag_at', symbol_lag_at), (symbol, 'poly_add', poly_add), (symbol, 'poly_sub', poly_sub),
        (symbol, 'poly_mul', poly_mul), (symbol, 'poly_divmod', poly_divmod), (symbol, 'poly_gcd', poly_gcd),
        (symbol, 'poly_powmod_x', poly_powmod_x), (symbol, '_interpolate', _interpolate),
        (circulant, '_dft', _dft), (circulant, 'accumulate_G', accumulate_G), (woodbury, 'factor', factor),
    ]


def _solver_modules():
    import sys
    return [m for name, m in list(sys.modules.items())
            if m is not None and (name.startswith('bttb_solvers._internal.banded_bttb') or name == 'bttb_solvers._internal.cc_prepared'
                                  or name == 'bttb_solvers._internal.halfstrip.allfield_poly')]


@contextmanager
def kernels():
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
    saved = {}
    for name, repl in {'apply_U': _apply_U, 'apply_UT': _apply_UT}.items():
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
        _cache.clear()
