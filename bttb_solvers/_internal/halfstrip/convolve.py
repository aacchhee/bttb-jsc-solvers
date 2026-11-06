# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.banded_bttb.field import int_element, root_of_unity

from .errors import NoNTTRoot


def next_pow2(x):
    n = 1
    while n < x:
        n <<= 1
    return n


def _schoolbook_linear(F, a, b):
    if not a or not b:
        return []
    out = [F.zero()] * (len(a) + len(b) - 1)

    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = F.add(out[i + j], F.mul(x, y))
    return out


class ConvolutionBackend(object):
    """One backend shared by cyclic-displacement applications and all CDQ merges."""

    def __init__(self, F, cnt=None, mode="auto", required_orders=()):
        if mode not in ("auto", "ntt", "schoolbook_oracle"):
            raise ValueError("convolution backend must be 'auto', 'ntt', or "
                             "'schoolbook_oracle'")
        self.F, self.cnt = F, cnt
        requested_orders = tuple(int(x) for x in required_orders)
        invalid = [m for m in requested_orders
                   if m < 1 or (m & (m - 1)) != 0]
        if invalid:
            raise ValueError("radix-two NTT orders required; got %r"
                             % (tuple(invalid),))
        orders = tuple(sorted(set(x for x in requested_orders if x > 1)))
        self.ntt_capable = bool(
            getattr(F, "is_finite", getattr(F, "q", None) is not None)
            and getattr(F, "p", None) is not None
            and getattr(F, "q", None) is not None)

        

        self.finite_field = self.ntt_capable
        if not self.ntt_capable:
            if mode == "ntt":
                raise NoNTTRoot(None, max(orders) if orders else 1,
                                field=F.name())
            bad = list(orders)
        else:
            bad = [m for m in orders
                   if m % F.p == 0 or (F.q - 1) % m != 0]
        if mode == "ntt" and bad:
            raise NoNTTRoot(F.q, max(bad))
        self.kind = ("ntt" if mode == "ntt" or
                     (mode == "auto" and self.ntt_capable and not bad)
                     else "schoolbook_oracle")
        self.requested_mode = mode
        self.required_orders = orders
        self._roots = {}
        self._inv_roots = {}
        self._inv_lengths = {}
        self._kernel_spectra = {}
        self.flint_products = 0
        self.flint_merges = 0
        if self.kind == "ntt" and orders:
            top = max(orders)
            omega = root_of_unity(F, top)
            self._roots[top] = omega
            self._inv_roots[top] = F.inv(omega)
            self._inv_lengths[top] = F.inv(int_element(F, top))

    @property
    def achieved(self):
        return self.kind == "ntt" and not (self.flint_products or self.flint_merges)

    def summary(self):
        if self.flint_products or self.flint_merges:
            return {'kind':'flint-nmod_poly', 'requested':self.requested_mode,
                    'required_orders':self.required_orders,
                    'cubic_ledger_achieved':False,
                    'reason':'compiled polynomial products; no inherited NTT or uniform exponent claim',
                    'compiled_products':self.flint_products,'compiled_2d_merges':self.flint_merges,
                    'operation_counts_valid':False,
                    'legacy_counter_scope':'historical NTT model charges only, not executed transforms',
                    'kernel_transform_cache':False,'zero_skipping':False}
        return {
            "kind": self.kind,
            "requested": self.requested_mode,
            "required_orders": self.required_orders,
            "cubic_ledger_achieved": self.achieved,
            "reason": ("exact radix-two NTTs execute every convolution"
                       if self.achieved else
                       ("exact rational schoolbook correctness oracle; "
                        "cubic_ledger_achieved=False and no Q bit-complexity "
                        "or cubic claim"
                        if not self.finite_field else
                        "exact schoolbook oracle fallback; correctness only, "
                        "not an achieved cubic implementation")),
            "zero_skipping": False,
        }

    def _root(self, n):
        if n <= 0 or n & (n - 1):
            raise ValueError("radix-two NTT length required")
        if n in self._roots:
            return self._roots[n]
        if not self.finite_field:
            raise NoNTTRoot(None, n, field=self.F.name())
        if n % self.F.p == 0 or (self.F.q - 1) % n != 0:
            raise NoNTTRoot(self.F.q, n)
        larger = [m for m in self._roots if m % n == 0]
        if larger:
            m = min(larger)
            omega = self.F.pow(self._roots[m], m // n)
        else:
            omega = root_of_unity(self.F, n)
        self._roots[n] = omega
        self._inv_roots[n] = self.F.inv(omega)
        self._inv_lengths[n] = self.F.inv(int_element(self.F, n))
        return omega

    def _ntt(self, values, inverse=False):
        F = self.F
        a = list(values)
        n = len(a)
        self._root(n)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        root = self._inv_roots[n] if inverse else self._roots[n]
        length = 2
        while length <= n:
            wlen = F.pow(root, n // length)
            half = length >> 1
            for base in range(0, n, length):
                w = F.one()
                for k in range(half):
                    u = a[base + k]
                    v = F.mul(a[base + k + half], w)
                    a[base + k] = F.add(u, v)
                    a[base + k + half] = F.sub(u, v)
                    w = F.mul(w, wlen)
            length <<= 1
        if inverse:
            invn = self._inv_lengths[n]
            a = [F.mul(x, invn) for x in a]
        return a

    def linear(self, a, b):
        if not a or not b:
            return []
        from bttb_solvers._internal.halfstrip_core.flint_backend import product
        accelerated = product(self.F, a, b)
        if accelerated is not None:
            self.flint_products += 1
            return accelerated
        needed = len(a) + len(b) - 1
        padded = next_pow2(needed)
        if self.kind == "ntt":
            z = self.F.zero()
            aa = self._ntt(list(a) + [z] * (padded - len(a)))
            bb = self._ntt(list(b) + [z] * (padded - len(b)))
            cc = [self.F.mul(x, y) for x, y in zip(aa, bb)]
            out = self._ntt(cc, inverse=True)[:needed]
        else:
            out = _schoolbook_linear(self.F, a, b)
        if self.cnt is not None:
            self.cnt.charge_conv(padded, engine=self.kind, transforms=3)
        return out

    def cyclic(self, a, b):
        if len(a) != len(b):
            raise ValueError("cyclic convolution needs equal lengths")
        n = len(a)
        raw = self.linear(a, b)
        out = [self.F.zero()] * n
        for k, x in enumerate(raw):
            out[k % n] = self.F.add(out[k % n], x)
        return out

    def linear_corr(self, h, v):
        if len(h) != len(v):
            raise ValueError("linear correlation needs equal lengths")
        n = len(h)
        raw = self.linear(list(reversed(h)), v)
        return raw[n - 1:2 * n - 1]

    def cyclic_corr(self, a, v):
        if len(a) != len(v):
            raise ValueError("cyclic correlation needs equal lengths")
        n = len(a)
        reflected = [a[(-i) % n] for i in range(n)]
        return self.cyclic(reflected, v)

    def _ntt2(self, matrix, inverse=False):
        rows = [self._ntt(row, inverse=inverse) for row in matrix]
        nr, nc = len(rows), len(rows[0])
        cols = [[rows[i][j] for i in range(nr)] for j in range(nc)]
        cols = [self._ntt(col, inverse=inverse) for col in cols]
        return [[cols[j][i] for j in range(nc)] for i in range(nr)]

    def merge_window(self, sources, kernel, n, first, last, left):
        """Return (window, kernel_transform_was_fresh)."""
        length = len(kernel)
        from bttb_solvers._internal.halfstrip_core.flint_backend import enabled_prime, product
        if enabled_prime(self.F) is not None:

            

            ns = next_pow2(3*n-2)
            aa = [0]*(left*ns)
            bb = [0]*(length*ns)
            for a in range(left):
                aa[a*ns:a*ns+n] = sources[a]
            for d in range(1,length):
                if kernel[d] is not None:
                    bb[d*ns:d*ns+2*n-1] = kernel[d]
            self.flint_merges += 1
            cc = product(self.F,aa,bb)
            return ({tau: cc[tau*ns+n-1:tau*ns+2*n-1]
                     for tau in range(first,last)},False)
        if self.kind != "ntt":
            out = {}
            for tau in range(first, last):
                acc = [self.F.zero()] * n
                for a in range(left):
                    d = tau - a
                    if d < 1 or d >= length or kernel[d] is None:
                        continue
                    for q in range(n):
                        for s in range(n):
                            acc[s] = self.F.add(
                                acc[s], self.F.mul(
                                    sources[a][q],
                                    kernel[d][(s + n - 1) - q]))
                out[tau] = acc
            return out, False

        nt = next_pow2(left + length - 1)
        ns = next_pow2(3 * n - 2)
        z = self.F.zero()
        key = (length, nt, ns, tuple(tuple(row) if row is not None else ()
                                     for row in kernel))
        fresh = key not in self._kernel_spectra
        if fresh:
            K = [[z] * ns for _ in range(nt)]
            for d in range(1, length):
                if kernel[d] is not None:
                    K[d][:2 * n - 1] = list(kernel[d])
            self._kernel_spectra[key] = self._ntt2(K)
        Khat = self._kernel_spectra[key]
        A = [[z] * ns for _ in range(nt)]
        for a in range(left):
            A[a][:n] = list(sources[a])
        Ahat = self._ntt2(A)
        Chat = [[self.F.mul(Ahat[i][j], Khat[i][j])
                 for j in range(ns)] for i in range(nt)]
        C = self._ntt2(Chat, inverse=True)
        return ({tau: C[tau][n - 1:2 * n - 1]
                 for tau in range(first, last)}, fresh)


def _oneoff(F, cnt, backend, needed):
    return backend or ConvolutionBackend(
        F, cnt, "auto", (next_pow2(needed),))


def linear_conv(F, a, b, cnt=None, backend=None):
    return _oneoff(F, cnt, backend, max(1, len(a) + len(b) - 1)).linear(a, b)


def cyclic_conv(F, a, b, cnt=None, backend=None):
    return _oneoff(F, cnt, backend, max(1, 2 * len(a) - 1)).cyclic(a, b)


def cyclic_corr(F, a, v, cnt=None, backend=None):
    return _oneoff(F, cnt, backend, max(1, 2 * len(a) - 1)).cyclic_corr(a, v)


def linear_corr(F, h, v, cnt=None, backend=None):
    return _oneoff(F, cnt, backend, max(1, 2 * len(h) - 1)).linear_corr(h, v)


def merge_window(F, sources, kernel, n, first, last, left, backend=None):
    be = backend or ConvolutionBackend(
        F, None, "auto", (next_pow2(left + len(kernel) - 1),
                           next_pow2(3 * n - 2)))
    out, _ = be.merge_window(sources, kernel, n, first, last, left)
    return out
