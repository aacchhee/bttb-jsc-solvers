# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from dataclasses import dataclass
from typing import Tuple

from .errors import NoGoodMu, SingularToeplitz, FieldTooSmall
from .field import ordered_elements
from .toeplitz import toeplitz_dense, toeplitz_det, zeros

CONVENTION = ("Chat_j = a(omega^{-j}) = sum_l omega^{-jl} A_l ; "
              "G_t = (1/m) sum_j omega^{+jt} Chat_j^{-1}")


@dataclass(frozen=True)
class BandedSymbol:
    """`2k+1` lag vectors of length `2n-1`.  `m` is deliberately NOT a field of
    this object: one symbol defines a family of matrices, one per `m`, and `m`
    enters at solver construction.

    lags[l + k][d + (n-1)] = a_l[d] for d = -(n-1)..(n-1), so A_l[s][t] = a_l[s-t]
    and B[i*n+s][j*n+t] = a_{i-j}[s-t].  (CC print `A_{j-i}`: a relabelling
    l <-> -l and no change of class.)"""

    F: object
    n: int
    k: int
    lags: Tuple[Tuple[object, ...], ...]

    def __post_init__(self):
        if len(self.lags) != 2 * self.k + 1:
            raise ValueError("expected %d lag vectors, got %d"
                             % (2 * self.k + 1, len(self.lags)))
        for v in self.lags:
            if len(v) != 2 * self.n - 1:
                raise ValueError("each lag vector has length %d exactly" % (2 * self.n - 1))

    def block(self, l):
        """Lag vector of A_l; the all-zero vector outside [-k, k]."""
        if -self.k <= l <= self.k:
            return self.lags[l + self.k]
        return tuple([self.F.zero()] * (2 * self.n - 1))

    def with_lag0_shift(self, E0):
        """A_0 -> A_0 + E0 for a CONSTANT TOEPLITZ E0 (Lemma C's repair).  Same
        class, same m, n, k; only the lag-0 block moves."""
        F, n, k = self.F, self.n, self.k
        e = [F.zero()] * (2 * n - 1)
        for d in range(-(n - 1), n):
            i, j = (d, 0) if d >= 0 else (0, -d)
            e[d + n - 1] = E0[i][j]
        new0 = tuple(F.add(a, b) for a, b in zip(self.lags[k], e))
        lags = list(self.lags)
        lags[k] = new0
        return BandedSymbol(F, n, k, tuple(lags))


def toeplitz_of(F, lag, n):
    """Dense n x n Toeplitz matrix.  For reference.py and tests ONLY."""
    return toeplitz_dense(F, lag, n)


def symbol_lag_at(F, sym, x):
    """Lag vector of a(x) = sum_{l=-k}^{k} x^l A_l.  O(k n) field operations.

    x must be nonzero when k >= 1 (negative powers occur)."""
    n, k = sym.n, sym.k
    xs = {0: F.one()}
    if k:
        xi = F.inv(x)
        p = F.one()
        q = F.one()
        for l in range(1, k + 1):
            p = F.mul(p, x)
            q = F.mul(q, xi)
            xs[l] = p
            xs[-l] = q
    out = [F.zero()] * (2 * n - 1)
    for l in range(-k, k + 1):
        c = xs[l]
        v = sym.lags[l + k]
        for d in range(2 * n - 1):
            if not F.is_zero(v[d]) and not F.is_zero(c):
                out[d] = F.add(out[d], F.mul(c, v[d]))
    return tuple(out)


def symbol_lag_at_freq(F, sym, m, j, mu, omega):
    """Lag vector of Chat_j(mu) = a(mu * omega^{-j}) in the DECLARED convention."""
    x = F.mul(mu, F.pow(omega, (-j) % m))
    return symbol_lag_at(F, sym, x)


def _ptrim(F, a):
    i = len(a)
    while i and F.is_zero(a[i - 1]):
        i -= 1
    return list(a[:i])


def poly_add(F, a, b):
    n = max(len(a), len(b))
    z = F.zero()
    return _ptrim(F, [F.add(a[i] if i < len(a) else z, b[i] if i < len(b) else z)
                      for i in range(n)])


def poly_sub(F, a, b):
    n = max(len(a), len(b))
    z = F.zero()
    return _ptrim(F, [F.sub(a[i] if i < len(a) else z, b[i] if i < len(b) else z)
                      for i in range(n)])


def poly_mul(F, a, b):
    if not a or not b:
        return []
    out = [F.zero()] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if F.is_zero(x):
            continue
        for j, y in enumerate(b):
            if F.is_zero(y):
                continue
            out[i + j] = F.add(out[i + j], F.mul(x, y))
    return _ptrim(F, out)


def poly_divmod(F, a, b):
    a = list(a)
    db = len(b) - 1
    il = F.inv(b[db])
    q = [F.zero()] * max(0, len(a) - db)
    for i in range(len(a) - 1, db - 1, -1):
        if not F.is_zero(a[i]):
            c = F.mul(a[i], il)
            q[i - db] = c
            for j in range(db + 1):
                a[i - db + j] = F.sub(a[i - db + j], F.mul(c, b[j]))
    return _ptrim(F, q), _ptrim(F, a)


def poly_gcd(F, a, b):
    a, b = _ptrim(F, a), _ptrim(F, b)
    while b:
        a, b = b, poly_divmod(F, a, b)[1]
    return a


def poly_powmod_x(F, e, f):
    """z^e mod f, by square and multiply on the polynomial z."""
    r, b = [F.one()], poly_divmod(F, [F.zero(), F.one()], f)[1] if len(f) > 1 else []
    if len(f) <= 1:
        return []
    while e:
        if e & 1:
            r = poly_divmod(F, poly_mul(F, r, b), f)[1]
        e >>= 1
        if e:
            b = poly_divmod(F, poly_mul(F, b, b), f)[1]
    return r


def _interpolate(F, xs, ys):
    """Exact reusable-tree interpolation; kernel complexity is field dependent.

    See polynomial.py: GF12289 has a bounded NTT specialization; other fields
    use schoolbook multiplication. The old numerator-by-numerator routine was
    cubic, despite its former O(N^2) docstring.
    """
    from .polynomial import ProductTree
    return ProductTree(F, xs).interpolate(ys)


def eval_points(F, count):
    """The first `count` elements of F* in the field's own element order.
    Fixed, documented, deterministic - no randomness anywhere in this library."""
    pts = []
    for x in ordered_elements(F):
        if F.is_zero(x):
            continue
        pts.append(x)
        if len(pts) == count:
            break
    if len(pts) < count:
        raise ValueError("only %d nonzero elements available, %d needed"
                         % (len(pts), count))
    return pts


def det_symbol_poly(F, sym, points=None):
    """f(z) = z^{kn} det a(z), as a coefficient list, low index first.

    deg f <= 2kn; the coefficient of z^{2kn} is det A_{+k} and f(0) is det A_{-k}
    (`-01` (a)).  Both are ASSERTED against toeplitz_det of A_{+-k}: a free and
    sharp check on the whole routine.  Returns [] when det a(z) is identically
    zero, which is exactly the branch test of T05 (c)."""
    n, k = sym.n, sym.k
    deg = 2 * k * n
    pts = points if points is not None else eval_points(F, deg + 1)
    xs, ys = [], []
    for x in pts:
        lag = symbol_lag_at(F, sym, x)
        d = toeplitz_det(F, lag, n)
        xs.append(x)
        ys.append(F.mul(d, F.pow(x, k * n)))
    f = _interpolate(F, xs, ys)
    lead = toeplitz_det(F, sym.block(k), n)
    const = toeplitz_det(F, sym.block(-k), n)
    got_lead = f[deg] if len(f) > deg else F.zero()
    got_const = f[0] if f else F.zero()
    assert F.eq(got_lead, lead), (
        "extreme-coefficient check failed: coeff z^{2kn} = %s but det A_k = %s"
        % (F.to_str(got_lead), F.to_str(lead)))
    assert F.eq(got_const, const), (
        "extreme-coefficient check failed: f(0) = %s but det A_{-k} = %s"
        % (F.to_str(got_const), F.to_str(const)))
    return f


def is_singular_symbol(F, sym):
    """det a(z) identically zero  <=>  f == 0."""
    return len(det_symbol_poly(F, sym)) == 0


def coset_representatives(F, m, count):
    """`count` elements of K* in PAIRWISE DISTINCT cosets of <omega>, in the
    fixed element order of the field.

    x and y lie in the SAME coset of <omega> iff x/y lies in <omega>, the
    subgroup of order m, iff x**m == y**m.  So the coset label is x**m - NOT
    x**((q-1)/m), which labels cosets of the subgroup of ORDER (q-1)/m and
    yields only m classes; with that exponent the candidate list silently comes
    back short and Lemma A's "at most 2kn of the (q-1)/m cosets are bad"
    guarantee is void.  That was a real bug in this file, caught by a spurious
    NoGoodMu over GF(5^3) at m = 4, and it is written down here because it is
    invisible whenever mu = 1 happens to work, which is the common case.

    No generator of K* is needed in this form.  The O(kn)-multiplication form
    that does need one is the CONDITIONAL route of T05 Open item 4."""
    seen, out = set(), []
    for x in ordered_elements(F):
        if F.is_zero(x):
            continue
        lab = F.pow(x, m)
        key = lab if isinstance(lab, tuple) else int(lab)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) == count:
            break
    if len(out) < count:

        raise FieldTooSmall(required_degree=0, q=F.q, m=m)
    return out


def _mu_is_good_naive(F, sym, m, mu, omega):
    """Test mu by the m symbol determinants.  Cost m*D, so the whole naive
    route is (2kn+1) m D exactly as `-01` (e) prices it."""
    for j in range(m):
        lag = symbol_lag_at_freq(F, sym, m, j, mu, omega)
        if F.is_zero(toeplitz_det(F, lag, sym.n)):
            return False
    return True


def find_mu(F, sym, m, omega, route="naive", f=None, stats=None):
    """mu in K* with det C(mu) != 0, i.e. all m symbols a(mu omega^{-j}) nonsingular.

    route='naive' : the fixed candidate list above, each tested by the m-fold
                    symbol determinant product.  (2kn+1) m D  -  `-01` (e).
    route='gcd'   : mu good <=> gcd(f(z), z^m - mu^m) == 1, f = z^{kn} det a(z);
                    the branch test is free inside this route.

    pre: (H1), (H2), (H3) and det a(z) not identically zero.
    raises NoGoodMu, which under (H3) CONTRADICTS `-01` (b): it is a refutation
    of a PROVED part and is reported with the full instance, never retried."""
    n, k = sym.n, sym.k
    cands = coset_representatives(F, m, 2 * k * n + 1)
    tried = 0
    if route == "gcd":
        if f is None:
            f = det_symbol_poly(F, sym)
        if not f:
            raise ValueError("gcd route needs det a(z) not identically zero")
        zm = poly_powmod_x(F, m, f)          
    for mu in cands:
        tried += 1
        if stats is not None:
            stats["tried"] = tried
        if route == "naive":
            if _mu_is_good_naive(F, sym, m, mu, omega):
                return mu
        elif route == "gcd":

            g = poly_gcd(F, f, poly_sub(F, zm, [F.pow(mu, m)]))
            if len(g) == 1:
                return mu
        else:
            raise ValueError("route must be 'naive' or 'gcd'")
    raise NoGoodMu(tried=[F.to_str(x) for x in cands],
                   witness={"field": F.name(), "m": m, "n": n, "k": k,
                            "lags": [[F.to_str(x) for x in v] for v in sym.lags]})
