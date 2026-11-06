# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
import hashlib
import time
from operator import mul as _mul

from .exact import (EtaleField, Restart, PrimeField, RationalField,
                    base_factory, inverse)
from .field_setup import setup_effective_field, descend, raw_scalar, BaseFieldView
from .poly import sa, stats as _poly_stats
from .flint_backend import enabled_prime as _flint_prime
from .solver import solve as _solve, _apply, _series, _validate_input
from .mosaic import solve_mosaic, _select_backend, _make_backend_multiplier


class SingularSystem(ArithmeticError):
    """The physical matrix is singular (determinant zero); no solve exists."""


class ReuseCertificateError(ArithmeticError):
    """A precondition of the R-div lemma failed at run time (no fallback)."""


class CountingField:
    """Exact-field proxy counting scalar operations.

    Deliberately exposes only the guarded scalar interface, so native packed
    products of the wrapped adapter are not used (counts are then honest field
    operation counts, at the price of speed).
    """
    def __init__(self, field):
        self._f = field
        self.ops = {'add': 0, 'sub': 0, 'neg': 0, 'mul': 0, 'inv': 0,
                    'is_zero': 0}
        self.p = getattr(field, 'p', 0)
        self.inverse_guards = []

    def zero(self): return self._f.zero()
    def one(self): return self._f.one()
    def from_int(self, x): return self._f.from_int(x)
    def add(self, a, b): self.ops['add'] += 1; return self._f.add(a, b)
    def sub(self, a, b): self.ops['sub'] += 1; return self._f.sub(a, b)
    def neg(self, a): self.ops['neg'] += 1; return self._f.neg(a)
    def mul(self, a, b): self.ops['mul'] += 1; return self._f.mul(a, b)
    def is_zero(self, a): self.ops['is_zero'] += 1; return self._f.is_zero(a)
    def inv(self, a):
        self.ops['inv'] += 1
        out = self._f.inv(a)
        self.inverse_guards.append({'denominator': a, 'unit': True})
        return out
    def eq(self, a, b): return self.is_zero(self.sub(a, b))
    def info(self, include_inversions=True):
        return {'kind': 'counting', 'ops': dict(self.ops)}
    def total(self):
        return sum(v for k, v in self.ops.items() if k != 'is_zero')


def _blocks_from_lags(lags, n, w):
    """lags[d][n-1+i-j] = F_d[i][j]; fewer than n+w rows allowed (rest zero)."""
    if type(n) is not int or n < 2 or not lags or len(lags) > n+w:
        raise ValueError('raw halfstrip lag dimensions')
    if any(len(row) != 2*n-1 for row in lags):
        raise ValueError('every lag row must have 2n-1 entries')
    return [[[row[n-1+i-j] for j in range(n)] for i in range(n)] for row in lags]


def _toeplitz_blocks(base, blocks, n):
    """True iff every block is constant along its diagonals."""
    for block in blocks:
        for i in range(1, n):
            for j in range(1, n):
                if not base.is_zero(base.sub(block[i][j], block[i-1][j-1])):
                    return False
    return True


def _mat_vec_physical(F, blocks, n, w, x):
    """Independent O(L n^3)-ish physical product T x (tests / residual checks)."""
    out = [F.zero() for _ in range(n*n)]
    for t in range(n):
        for s in range(n):
            d = t-s+w
            if not 0 <= d < len(blocks):
                continue
            blk = blocks[d]
            for i in range(n):
                acc = out[t*n+i]
                for j in range(n):
                    acc = F.add(acc, F.mul(blk[i][j], x[s*n+j]))
                out[t*n+i] = acc
    return out


def physical_matvec(F, blocks, n, w, x, transpose=False):
    """T x or T^t x over the field F for blocks in the halfstrip_core convention."""
    if not transpose:
        return _mat_vec_physical(F, blocks, n, w, x)
    out = [F.zero() for _ in range(n*n)]
    for t in range(n):
        for s in range(n):
            d = t-s+w          
            if not 0 <= d < len(blocks):
                continue
            blk = blocks[d]
            for i in range(n):
                for j in range(n):
                    out[s*n+j] = F.add(out[s*n+j], F.mul(blk[i][j], x[t*n+i]))
    return out


def field_view_info(F):
    """What the algorithms see of the field: wrapper class and native packed hooks.

    Every benchmark arm that is compared with another must show the same record
    (review-reuse-code.md finding 1: the old ``classical`` arm hid the hooks).
    """
    base = getattr(F, '_base', None)
    return {'wrapper': type(F).__name__,
            'base_adapter': None if base is None else type(base).__name__,
            'has_polynomial_native': hasattr(F, 'polynomial_native'),
            'has_polynomial_leaf': hasattr(F, 'polynomial_leaf')}


def _int_prime(F):
    """p when F is a plain BaseFieldView over an exact PrimeField (native int
    elements), else None.  CountingField, EtaleField and other adapters give None."""
    base = getattr(F, '_base', None)
    if isinstance(F, BaseFieldView) and type(base) is PrimeField and base.p > 1:
        return base.p
    return None


def _count_elements(obj):
    if isinstance(obj, dict):
        return sum(_count_elements(v) for v in obj.values())
    if isinstance(obj, (list,)):
        return sum(_count_elements(v) for v in obj)
    return 1


class _Core:
    """Prepared state over one working algebra F; raises Restart to its owner."""

    def __init__(self, F, blocks, n, w, labels, terminal_backend, terminal_query,
                 verify_certificate):
        self.F, self.n, self.w = F, n, w
        self.L = n+w
        self.terminal_backend, self.terminal_query = terminal_backend, terminal_query
        out = _solve(F, blocks, n, w, None, labels, terminal_backend=terminal_backend)
        j = out['joint']
        self.q = out['carrier']['q']
        self.overcap = bool(j['overcap'])
        self.delta = list(j.get('delta', ()))
        self.det = out['det']
        self.singular = bool(out['singular'])
        self.report = {'q': self.q, 'delta': tuple(self.delta), 'overcap': self.overcap,
                       'terminal_backend': terminal_backend,
                       'terminal_query': terminal_query, 'k': sum(self.delta)}
        self.joint = None
        self.shuffle = None
        self.backend_app = None
        self._div = None
        self._lu = None
        if self.overcap or self.singular:
            return
        self.joint = {key: j[key] for key in ('W', 'G', 'B', 'delta')}
        self.shuffle = out['terminal']['physical_shuffle']
        backend = out['terminal']['backend']
        self.report.update(m=backend['order'], r=backend['r'], b_c=backend['chunk_bound'],
                           A_c=len(backend['chunks']),
                           B_c=len(backend['column_chunks']),
                           terminal_order=backend['approximant_order'])
        if terminal_query == 'lu':
            self._lu = self._prepare_lu(backend)
            self.report['lu_kernel'] = self._lu['kernel']
            self.report['baseline_note'] = ('terminal-lu: dense m x m LU of the terminal system; '
                                           'a baseline, not the lemma route')
        elif terminal_query == 'appended':
            keep = ('order', 'approximant_order', 'chunks', 'chunk_bound', 'offsets',
                    'delta', 'rho', 'gamma', 'profile_matches', 'terminal_backend',
                    'coefficients')
            self.backend_app = {key: backend[key] for key in keep}
        else:
            self._div = self._prepare_division(backend, verify_certificate)

        self.report['retained_elements'] = self.retained_elements()


    def _prepare_lu(self, backend):
        """Dense m x m terminal matrix and its guarded LU, retained for all queries.

        M[(i,a),(j,b)] = coefficient(i,j,a-b) from the blocks retained by
        ``run_mosaic``.  Every zero test and inversion goes through the guarded
        field.  When the view is a plain prime field the row updates use native
        ints (one reduction per entry); otherwise the guarded scalar interface.
        """
        F = self.F
        if not backend['profile_matches']:
            raise ReuseCertificateError('nonsingular input with profile != Delta')
        rows, cols = backend['chain_lengths'], backend['column_lengths']
        blocks, m = backend['coefficient_blocks'], backend['order']
        M = []
        for i, li in enumerate(rows):
            for a in range(li):
                row = []
                for j, cj in enumerate(cols):
                    blk = blocks[i][j]
                    row.extend(blk[a-b+cj-1] for b in range(cj))
                M.append(row)
        if len(M) != m or any(len(r) != m for r in M):
            raise ReuseCertificateError('terminal matrix shape mismatch')
        zero, is_zero, mul, sub = F.zero(), F.is_zero, F.mul, F.sub
        p = _int_prime(F)
        perm = list(range(m))
        dinv = []
        for k in range(m):
            piv = None
            for r in range(k, m):
                if not is_zero(M[r][k]):
                    piv = r
                    break
            if piv is None:
                raise ReuseCertificateError(
                    'terminal matrix is singular although the profile equals Delta')
            if piv != k:
                M[k], M[piv] = M[piv], M[k]
                perm[k], perm[piv] = perm[piv], perm[k]
            inv_k = F.inv(M[k][k])
            dinv.append(inv_k)
            from .flint_backend import lu_update
            if lu_update(F,M,k,inv_k):
                continue
            tail = M[k][k+1:]
            for r in range(k+1, m):
                x = M[r][k]
                if is_zero(x):
                    M[r][k] = zero
                    continue
                f = mul(x, inv_k)
                M[r][k] = f
                if p is not None:
                    M[r][k+1:] = [(u - f*v) % p for u, v in zip(M[r][k+1:], tail)]
                else:
                    row = M[r]
                    for c, v in enumerate(tail, k+1):
                        if not is_zero(v):
                            row[c] = sub(row[c], mul(f, v))
        Lrows = [M[i][:i] for i in range(m)]
        Urows = [M[i][i+1:] for i in range(m)]
        return {'perm': perm, 'L': Lrows, 'U': Urows, 'dinv': dinv, 'm': m,
                'kernel': ('flint-nmod_mat rows' if _flint_prime(F) is not None else
                           'native-int rows' if p is not None else 'guarded scalar rows'), 'p': p}

    def _terminal_lu(self, dchain):
        """y = M^-1 d from the retained LU: forward then backward substitution, O(m^2)."""
        F = self.F
        d = self._lu
        m, perm, L, U, dinv, p = d['m'], d['perm'], d['L'], d['U'], d['dinv'], d['p']
        if len(dchain) != m:
            raise ValueError('terminal right-hand side length')
        z = []
        from .flint_backend import enabled_prime, dot
        if enabled_prime(F) is not None:
            for i in range(m):
                z.append(F.sub(dchain[perm[i]],dot(F,L[i],z)))
            y=[0]*m
            for i in range(m-1,-1,-1):
                y[i]=F.mul(F.sub(z[i],dot(F,U[i],y[i+1:])),dinv[i])
            return y
        if p is not None:
            for i in range(m):
                li = L[i]
                acc = dchain[perm[i]] - (sum(map(_mul, li, z)) if li else 0)
                z.append(acc % p)
            y = [0]*m
            for i in range(m-1, -1, -1):
                ui = U[i]
                acc = z[i] - (sum(map(_mul, ui, y[i+1:])) if ui else 0)
                y[i] = (acc * dinv[i]) % p
            return y
        sub, mul, add = F.sub, F.mul, F.add
        for i in range(m):
            acc = dchain[perm[i]]
            for l, v in zip(L[i], z):
                acc = sub(acc, mul(l, v))
            z.append(acc)
        y = [None]*m
        for i in range(m-1, -1, -1):
            acc = z[i]
            for u, v in zip(U[i], y[i+1:]):
                acc = sub(acc, mul(u, v))
            y[i] = mul(acc, dinv[i])
        return y


    def _prepare_division(self, backend, verify):
        F = self.F
        engine, product, _ = _select_backend(self.terminal_backend)
        mul = _make_backend_multiplier(F, self.terminal_backend)
        if not backend['profile_matches']:
            raise ReuseCertificateError('nonsingular input with profile != Delta')
        basis = backend['basis']['basis']
        Delta = list(backend['delta'])
        comps = len(Delta)
        b = backend['chunk_bound']
        B = len(backend['column_chunks'])
        A = len(backend['chunks'])
        if comps != A+B or len(basis[0]) != comps:
            raise ReuseCertificateError('unexpected root-basis shape')

        for k, C in enumerate(basis):
            for i in range(comps):
                for jj in range(comps):
                    if k > Delta[jj] and not F.is_zero(C[i][jj]):
                        raise ReuseCertificateError(
                            'column-degree bound violated: deg (P_M)[%d][%d] >= %d > Delta=%d'
                            % (i, jj, k, Delta[jj]))
        zero = F.zero()
        lc = [[(basis[Delta[jj]][i][jj] if Delta[jj] < len(basis) else zero)
               for jj in range(comps)] for i in range(comps)]
        try:
            lc_inv = inverse(F, lc)
        except ZeroDivisionError:
            raise ReuseCertificateError(
                'leading-coefficient matrix of the root basis is singular '
                '(contradicts lemma B1 for nonsingular M)') from None

        qlen = min(max(Delta)+1, b)
        Q = []
        for k in range(qlen):
            Q.append([[(basis[Delta[jj]-k][i][jj] if 0 <= Delta[jj]-k < len(basis) else zero)
                       for jj in range(comps)] for i in range(comps)])

        S = [lc_inv]
        t = 1
        two = F.add(F.one(), F.one())
        while t < b:
            t = min(2*t, b)
            QS = product(F, Q[:t], S, mul, limit=t, skip_zeros=True)
            U = [[[F.neg(x) for x in row] for row in C] for C in QS]
            U[0] = [[F.add(U[0][i][jj], two if i == jj else zero)
                     for jj in range(comps)] for i in range(comps)]
            S = product(F, S, U, mul, limit=t, skip_zeros=True)
        S = S[:b]
        if verify:
            self._check_inverse(F, product, mul, Q, S, b, comps)

        Sent = [[None]*comps for _ in range(comps)]
        for jj in range(comps):
            for i in range(comps):
                poly = [C[jj][i] for C in S]
                while poly and F.is_zero(poly[-1]):
                    poly.pop()
                Sent[jj][i] = poly

        Pcols = []
        for c in range(B):
            col = []
            for i in range(comps):
                poly = [basis[k][i][c] for k in range(min(len(basis), Delta[c]+1))]
                while poly and F.is_zero(poly[-1]):
                    poly.pop()
                col.append(poly)
            Pcols.append(col)
        return {'mul': mul, 'S': Sent, 'Pcols': Pcols, 'Q': Q, 'comps': comps,
                'A': A, 'B': B, 'b': b, 'Delta': Delta,
                'gamma': list(backend['gamma']), 'chunks': list(backend['chunks']),
                'lc_inverse_ok': True}

    @staticmethod
    def _check_inverse(F, product, mul, Q, S, b, comps):
        QS = product(F, Q[:b], S, mul, limit=b, skip_zeros=True)
        for k, C in enumerate(QS):
            for i in range(comps):
                for jj in range(comps):
                    want = F.one() if (k == 0 and i == jj) else F.zero()
                    if not F.is_zero(F.sub(C[i][jj], want)):
                        raise ReuseCertificateError('Q*S != I mod x^b_c')

    def certificate(self):
        """Re-verify Q*S == I (mod x^b_c); raises ReuseCertificateError."""
        d = self._div
        if d is None:
            raise ValueError('no division state (singular or appended mode)')
        _, product, _ = _select_backend(self.terminal_backend)
        self._check_inverse(self.F, product, d['mul'], d['Q'], self._S_as_matrices(),
                            d['b'], d['comps'])
        return True

    def _S_as_matrices(self):
        d = self._div
        F = self.F
        comps, b = d['comps'], d['b']
        S = []
        for k in range(b):
            S.append([[(d['S'][jj][i][k] if k < len(d['S'][jj][i]) else F.zero())
                       for i in range(comps)] for jj in range(comps)])
        return S


    def retained_elements(self):
        n = 0
        if self.joint is not None:
            n += sum(_count_elements(self.joint[k]) for k in ('W', 'G', 'B'))
        if self._div is not None:
            d = self._div
            n += _count_elements(d['S']) + _count_elements(d['Pcols']) + _count_elements(d['Q'])
        if self.backend_app is not None:
            n += _count_elements(self.backend_app['coefficients'])
        if self._lu is not None:
            n += _count_elements(self._lu['L']) + _count_elements(self._lu['U']) + len(self._lu['dinv'])
        return n

    def state_hash(self):
        """sha256 of the retained numeric state (reuse-integrity test)."""
        h = hashlib.sha256()
        def feed(o):
            if isinstance(o, dict):
                for k in sorted(o, key=repr):
                    h.update(repr(k).encode()); feed(o[k])
            elif isinstance(o, (list, tuple)):
                h.update(b'[')
                for v in o:
                    feed(v)
                h.update(b']')
            else:
                h.update(repr(o).encode())
        feed(self.joint)
        feed(self.shuffle)
        feed(self.det)
        if self._div is not None:
            d = self._div
            for key in ('S', 'Pcols', 'Q', 'gamma', 'chunks', 'Delta', 'b', 'A', 'B', 'comps'):
                h.update(key.encode()); feed(d[key])
        if self.backend_app is not None:
            feed(self.backend_app)
        if self._lu is not None:
            for key in ('perm', 'L', 'U', 'dinv', 'm'):
                h.update(key.encode()); feed(self._lu[key])
        return h.hexdigest()

    def _terminal_divide(self, dchain):
        """Part (B): y-components of rho = f0 - q P_M, q = rev(phi S mod x^b)."""
        F = self.F
        d = self._div
        mul, S, Pcols = d['mul'], d['S'], d['Pcols']
        A, B, b, comps = d['A'], d['B'], d['b'], d['comps']
        zero = F.zero()

        ghat = []
        for jj, chunk in enumerate(d['chunks']):
            poly = [zero]*b
            for mm in range(chunk['length']):
                poly[b-1-mm] = F.neg(dchain[chunk['start']+mm])
            ghat.append(poly)

        q = []
        for i in range(comps):
            acc = [zero]*b
            for jj in range(A):
                s = S[B+jj][i]
                if not s:
                    continue
                prod = mul(ghat[jj], s)
                for k in range(min(b, len(prod))):
                    acc[k] = F.add(acc[k], prod[k])
            q.append([acc[b-1-e] for e in range(b)])      

        solution = []
        for c in range(B):
            gam = d['gamma'][c]
            total = []
            for i in range(comps):
                p = Pcols[c][i]
                if not p:
                    continue
                prod = mul(q[i], p)
                if len(prod) > len(total):
                    total += [zero]*(len(prod)-len(total))
                for k, x in enumerate(prod):
                    total[k] = F.add(total[k], x)
            for k in range(gam, len(total)):
                if not F.is_zero(total[k]):
                    raise ReuseCertificateError(
                        'division remainder violates deg y_%d < gamma=%d (coefficient %d)'
                        % (c, gam, k))
            for k in range(gam):
                solution.append(F.neg(total[k]) if k < len(total) else zero)
        return solution

    def _terminal_appended(self, dchain):
        affine = solve_mosaic(self.F, self.backend_app, dchain)
        if affine['solution'] is None:
            raise ArithmeticError('nonsingular terminal affine basis lacks readout')
        return affine['solution']

    def query(self, rhs):
        """Lean reconstruct: physical solution (working-algebra elements)."""
        F, n, w, L = self.F, self.n, self.w, self.L
        joint, delta = self.joint, self.delta
        dmax = max(delta, default=0)
        f = [F.zero()]*(n*w) + list(rhs)
        fs = _series(F, f, n, L)
        Wf = _apply(F, joint['W'], fs, -dmax, L)
        Gf = _apply(F, joint['G'], fs, 0, L+dmax)
        dtime = [F.neg(Wf[r].get(t, F.zero())) for t in range(n, L) for r in range(n)]
        dtime += [Gf[i].get(t, F.zero()) for i, de in enumerate(delta) for t in range(de)]
        shuffle = self.shuffle
        dchain = [dtime[i] for i in shuffle]
        if self.terminal_query == 'division':
            solution = self._terminal_divide(dchain)
        elif self.terminal_query == 'lu':
            solution = self._terminal_lu(dchain)
        else:
            solution = self._terminal_appended(dchain)
        time_solution = [F.zero()]*len(solution)
        for i, target in enumerate(shuffle):
            time_solution[target] = solution[i]
        u, sigma = time_solution[:n*w], time_solution[n*w:]
        y = list(f)
        for i, value in enumerate(u):
            y[i] = F.add(y[i], value)
        ys = _series(F, y, n, L)
        Wy = _apply(F, joint['W'], ys, -dmax, L)
        kernels = [{} for _ in delta]
        at = 0
        for i, de in enumerate(delta):
            for bb in range(de):
                value = sigma[at]
                at += 1
                if not F.is_zero(value):
                    kernels[i][L-de+bb] = value
        BK = _apply(F, joint['B'], kernels, 0, L)
        xseries = [sa(F, Wy[i], BK[i], -dmax, L) for i in range(n)]
        for series in xseries:
            if any(k < 0 and not F.is_zero(v) for k, v in series.items()):
                raise ArithmeticError('physical reconstruction has negative support')
        xfull = [xseries[r].get(t, F.zero()) for t in range(L) for r in range(n)]
        if any(not F.is_zero(v) for v in xfull[n*n:]):
            raise ArithmeticError('physical reconstruction has nonzero tail')
        return xfull[:n*n]


class FailedH0Reuse:
    """Prepared failed-H0 half-strip handle: det, solve, solve_transpose.

    Parameters are those of ``prepare_blocks``.  The handle owns the restart
    loop: a Restart in preparation or in a query discards every field-dependent
    object and re-prepares from the stored original base-field blocks with the
    restricted modulus (algorithm Step 0 / lemma part E).
    """

    def __init__(self, base, blocks, w, *, terminal_backend='classical',
                 terminal_query='division', verify=False):
        n = len(blocks[0]) if blocks else 0
        _validate_input(blocks, n, w)
        if terminal_backend not in ('classical', 'uniform_fast'):
            raise ValueError('terminal_backend must be classical or uniform_fast')
        if terminal_query not in ('division', 'appended', 'lu'):
            raise ValueError("terminal_query must be 'division', 'appended' or 'lu'")
        self.base, self.n, self.w = base, n, w
        self._blocks = [[list(r) for r in blk] for blk in blocks]
        self.terminal_backend, self.terminal_query = terminal_backend, terminal_query
        self._verify = verify
        self.toeplitz = _toeplitz_blocks(base, self._blocks, n)
        self.restarts = []
        self._initial_degree = None
        self._restriction = None
        self.prepare_seconds = None
        self.query_log = []
        self._establish()


    def _establish(self):
        started = time.perf_counter()
        while True:
            F, emb, labels, meta, embed = setup_effective_field(
                self.base, self._blocks, self.n, self._restriction)
            self.F, self._embed, self.field_setup = F, embed, meta
            if self._initial_degree is None:
                self._initial_degree = getattr(F, 'degree', 1)
            try:
                self._core = _Core(F, emb, self.n, self.w, labels, self.terminal_backend,
                                   self.terminal_query, self._verify)
                break
            except Restart as event:
                self._note_restart(event, 'prepare')
        elapsed = time.perf_counter()-started
        self.prepare_seconds = (self.prepare_seconds or 0.0)+elapsed

    def _note_restart(self, event, where):
        old = self.F.degree
        new = len(event.factor)-1
        if not 1 <= new < old:
            raise ArithmeticError('restriction did not strictly lower the current degree')
        if len(self.restarts)+1 > self._initial_degree-1:
            raise ArithmeticError('restriction exceeded the global degree-drop bound')
        self.restarts.append({'where': where, 'old_degree': old, 'new_degree': new,
                              'guard_index': event.guard_index,
                              'restart_from': 'stored original base-field blocks'})
        self._restriction = event.factor


    @property
    def det(self):
        return descend(self.F, self._core.det)

    @property
    def singular(self):
        return self._core.singular

    @property
    def report(self):
        r = dict(self._core.report)
        r.update(restarts=len(self.restarts), restart_events=list(self.restarts),
                 n=self.n, w=self.w, prepare_seconds=self.prepare_seconds,
                 queries=len(self.query_log), field_view=field_view_info(self.F))
        return r

    def counts(self):
        """Cheap counters: guards/inversions and structural polynomial stats."""
        F = self.F
        out = {'guards': getattr(F, 'guards', None),
               'inverse_guards': len(getattr(F, 'inverse_guards', ())),
               'poly': _poly_stats(F)}
        ops = getattr(self.base, 'ops', None)      
        if ops is not None:
            out['field_ops'] = dict(ops)
        return out

    def state_hash(self):
        return self._core.state_hash()

    def certificate(self):
        return self._core.certificate()


    def _query(self, rhs, kind):
        if len(rhs) != self.n*self.n:
            raise ValueError('RHS must have n*n physical time-major entries')
        if self.singular:
            raise SingularSystem('physical matrix is singular (determinant zero)')
        started = time.perf_counter()
        original = list(rhs)
        while True:
            try:
                embedded = [self._embed(v) for v in original]
                physical = self._core.query(embedded)
                result = [descend(self.F, v) for v in physical]
                break
            except Restart as event:
                self._note_restart(event, 'query')
                self._establish()
                if self.singular:
                    raise SingularSystem('physical matrix is singular (determinant zero)')
        self.query_log.append({'kind': kind, 'seconds': time.perf_counter()-started})
        return result

    def solve(self, rhs):
        """x with T x = rhs (base-field elements, length n*n, time-major)."""
        return self._query(rhs, 'forward')

    def solve_transpose(self, rhs):
        """x with T^t x = rhs, via T^t = J T J (needs Toeplitz blocks)."""
        if not self.toeplitz:
            raise ValueError('transpose identity T^t = J T J requires Toeplitz blocks')
        return list(reversed(self._query(list(reversed(rhs)), 'transpose')))


def prepare_blocks(base, blocks, w, *, terminal_backend='classical',
                   terminal_query='division', verify=False):
    """Prepare from halfstrip_core-convention blocks over an effective exact field."""
    return FailedH0Reuse(base, blocks, w, terminal_backend=terminal_backend,
                         terminal_query=terminal_query, verify=verify)


def prepare_failed_h0_halfstrip(base, lags, n, w, *, terminal_backend='classical',
                                terminal_query='division', verify=False):
    """Prepare from raw lags, lags[d][n-1+i-j] = F_d[i][j] (design D1 API)."""
    return FailedH0Reuse(base, _blocks_from_lags(lags, n, w), w,
                         terminal_backend=terminal_backend,
                         terminal_query=terminal_query, verify=verify)


def oneshot_solve(base, blocks, w, rhs, *, terminal_backend='classical'):
    """HSC reference: the full pipeline per RHS (solve_field with a chosen engine).

    Returns the result dict of ``solver.solve`` (``det_base``, ``solution``,
    ``singular``) plus ``field_view`` (``field_view_info`` of the working field).
    Same restart loop as ``solver.solve_field``.  This sees exactly the field
    view that ``FailedH0Reuse`` sees (``setup_effective_field``), so it is the HSC
    baseline for every reuse arm; the benchmark's old ``classical`` arm hides the
    native packed hooks and must not be mixed with it.
    """
    n = len(blocks[0])
    _validate_input(blocks, n, w, rhs)
    restriction = None
    while True:
        F, emb, labels, meta, embed = setup_effective_field(base, blocks, n, restriction)
        try:
            e_rhs = None if rhs is None else [embed(v) for v in rhs]
            res = _solve(F, emb, n, w, e_rhs, labels, terminal_backend=terminal_backend)
            res['field_view'] = field_view_info(F)
            return res
        except Restart as event:
            restriction = event.factor


def _field_spec(field):
    """Return (name, base, rand) for 'F<p>', 'Q', 'F4' or an int prime."""
    if isinstance(field, int):
        field = 'F%d' % field
    base = base_factory(field)
    return field, base


def _rank_of(family, n):
    if family == 'rank-nm1':
        k = n-1
    elif family == 'rank-half':
        k = n//2
    elif family.startswith('rank-'):
        k = int(family[5:])
    else:
        return None

    
    if not 0 <= k < n:
        raise ValueError('family %s: rank must satisfy 0 <= k < n = %d (F0 must fail H0)' % (family, n))
    return k


def random_failed_h0_lags(n, w, family, seed, field='F998244353'):
    """Seeded raw lags (lags[d][n-1+i-j]) over named field; F0 per family.

    Families: ``rank-<k>`` / ``rank-nm1`` / ``rank-half`` (F0 = a single
    diagonal of ones, rank k, 0 <= k < n; ``rank-0`` is F0 = 0), ``higher`` (one
    diagonal at offset 2), ``shift`` (F0 = the single corner entry F0[n-1][0],
    rank 1; F_w = identity diagonal; every other block zero), ``generic``
    (random F0), ``zero`` (all blocks zero).  Entries are integers:
    mod p for F<p>, 0..3 for F4, small signed integers for Q.
    """
    import random
    name, base = _field_spec(field)
    rng = random.Random(seed)
    if name == 'Q':
        draw = lambda: rng.randrange(-5, 6)
    elif name == 'F4':
        draw = lambda: rng.randrange(4)
    else:
        p = int(name[1:])
        draw = lambda: rng.randrange(p)
    lags = [[draw() for _ in range(2*n-1)] for _ in range(n+w)]
    rank = _rank_of(family, n)
    if rank is not None:
        lags[0] = [0]*(2*n-1)
        if rank:
            lags[0][2*n-rank-1] = 1
    elif family == 'higher':
        lags[0] = [0]*(2*n-1)
        lags[0][n-1+min(2, n-1)] = 1
    elif family == 'shift':
        lags = [[0]*(2*n-1) for _ in range(n+w)]
        lags[0][2*n-2] = 1          
        lags[w][n-1] = 1
    elif family == 'zero':
        lags = [[0]*(2*n-1) for _ in range(n+w)]
    elif family != 'generic':
        raise ValueError('unknown family '+family)
    return lags


def lags_to_base_blocks(name, base, lags, n):
    """Convert integer lags to base-field block elements (halfstrip_core convention)."""
    return [[[raw_scalar(base, name, row[n-1+i-j]) for j in range(n)] for i in range(n)]
            for row in lags]


def _rhs_vectors(name, base, n, count, seed):
    import random
    rng = random.Random(seed*7919+13)
    N = n*n
    out = []
    for _ in range(count):
        if name == 'Q':
            out.append([raw_scalar(base, name, rng.randrange(-9, 10)) for _ in range(N)])
        elif name == 'F4':
            out.append([raw_scalar(base, name, rng.randrange(4)) for _ in range(N)])
        else:
            p = int(name[1:])
            out.append([base.from_int(rng.randrange(p)) for _ in range(N)])
    return out


def _sha_json(value):
    import hashlib as _h
    import json as _json
    return _h.sha256(_json.dumps(value, default=str).encode()).hexdigest()


MODES = ('rdiv', 'rapp', 'terminal-lu', 'oneshot')
_QUERY_OF_MODE = {'rdiv': 'division', 'rapp': 'appended', 'terminal-lu': 'lu'}
BASELINE_NOTE = {'oneshot': 'HSC baseline: full pipeline per right-hand side (one-shot)',
                 'terminal-lu': 'BASELINE (not the lemma route): prepared once, retained dense m x m terminal LU, '
                                'O(m^2) per query'}


def run_cell(n, w, field, seed, R, mode='rdiv', *, family='higher',
             terminal_backend='classical', transpose=True, verify=True,
             count_ops=False):
    """One benchmark cell; returns a JSON-able dict (prepare/query times split).

    mode: ``'rdiv'`` (prepare once, division), ``'rapp'`` (prepare once,
    appended affine PM-Basis per RHS), ``'terminal-lu'`` (prepare once, retained
    dense terminal LU: a baseline), ``'oneshot'`` (full HSC pipeline per RHS).

    All four modes get the SAME base field object and go through the same
    ``setup_effective_field`` view (native packed hooks visible); ``field_view``
    in the result records the wrapper class and the hook set actually seen, so a
    benchmark can assert that compared arms saw the same view.  The old benchmark
    ``classical`` arm hides the hooks (6-7x slower on identical input) and is NOT
    comparable with these modes.

    Phase names follow the benchmark worker: prepare / determinant /
    forward-i / transpose-i (oneshot: combined-forward-i / combined-transpose-i).
    Residual checks (T x = b, T^t x = b) are done outside the timers when
    ``verify``.  ``count_ops`` wraps the base in CountingField: exact scalar
    operation counts on a DIFFERENT code path (no native hooks), so the times of
    such a run are never comparable with a timed run (``count_path_differs_from_timed``).

    Per-RHS cost of ``oneshot`` is independent of R by construction (the whole
    pipeline repeats), so a benchmark runs it at R = 1 and scales per RHS in the
    analysis.  ``pair0_sha256`` hashes the first forward/transpose answers, which
    do not depend on R (same seeded RHS prefix), to tie R = 1 blocks to larger-R
    blocks.  A ``ReuseCertificateError`` in a timed phase is returned as
    ``status == 'certificate_failure'`` (partial phases kept), not raised.
    """
    from statistics import median
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    name, base0 = _field_spec(field)
    counting = CountingField(base0) if count_ops else None
    base = counting if counting is not None else base0
    lags = random_failed_h0_lags(n, w, family, seed, name)
    blocks = lags_to_base_blocks(name, base0, lags, n)
    fwd = _rhs_vectors(name, base0, n, R, seed)
    bwd = _rhs_vectors(name, base0, n, R if transpose else 0, seed+1)
    phases = []
    def snap():
        return dict(counting.ops) if counting is not None else {}
    def phase(label, fn):
        before = snap()
        wall, cpu = time.perf_counter(), time.process_time()
        value = fn()
        row = {'phase': label, 'wall_seconds': time.perf_counter()-wall,
               'cpu_seconds': time.process_time()-cpu}
        if counting is not None:
            after = snap()
            row['field_ops'] = {k: after[k]-before[k] for k in after}
        phases.append(row)
        return value
    solutions, transposes = [], []
    det = None
    singular = False
    report = None
    view = None
    failure = None
    L = n+w
    try:
        if mode != 'oneshot':
            handle = phase('prepare', lambda: FailedH0Reuse(
                base, blocks, w, terminal_backend=terminal_backend,
                terminal_query=_QUERY_OF_MODE[mode]))
            det = phase('determinant', lambda: handle.det)
            singular = handle.singular
            report = handle.report
            view = report['field_view']
            if not singular:
                for i, b in enumerate(fwd):
                    solutions.append(phase('forward-%d' % i, lambda b=b: handle.solve(b)))
                for i, b in enumerate(bwd):
                    transposes.append(phase('transpose-%d' % i,
                                            lambda b=b: handle.solve_transpose(b)))
        else:
            for i, b in enumerate(fwd):
                res = phase('combined-forward-%d' % i, lambda b=b: oneshot_solve(
                    base, blocks, w, b, terminal_backend=terminal_backend))
                det, singular, view = res['det_base'], res['singular'], res['field_view']
                if res['solution'] is not None:
                    solutions.append(res['solution'])
            for i, b in enumerate(bwd):
                res = phase('combined-transpose-%d' % i, lambda b=b: oneshot_solve(
                    base, blocks, w, list(reversed(b)), terminal_backend=terminal_backend))
                view = view or res['field_view']
                if res['solution'] is not None:
                    transposes.append(list(reversed(res['solution'])))
            if R == 0 or not fwd:
                res = phase('determinant', lambda: oneshot_solve(
                    base, blocks, w, None, terminal_backend=terminal_backend))
                det, singular, view = res['det_base'], res['singular'], res['field_view']
    except ReuseCertificateError as exc:
        failure = {'type': 'ReuseCertificateError', 'message': str(exc),
                   'after_phases': [p['phase'] for p in phases]}
    verified = None
    if failure is None and not singular:

        if len(solutions) != len(fwd) or len(transposes) != len(bwd):
            raise ArithmeticError('missing solutions: %d/%d forward, %d/%d transpose'
                                  % (len(solutions), len(fwd), len(transposes), len(bwd)))
        if verify:
            verified = True
            for x, b in zip(solutions, fwd):
                assert len(x) == n*n and len(b) == n*n
                verified &= all(base0.is_zero(base0.sub(u, v)) for u, v in zip(
                    physical_matvec(base0, blocks, n, w, x), b))
            for x, b in zip(transposes, bwd):
                assert len(x) == n*n and len(b) == n*n
                verified &= all(base0.is_zero(base0.sub(u, v)) for u, v in zip(
                    physical_matvec(base0, blocks, n, w, x, transpose=True), b))
    out_hash = _sha_json([solutions, transposes])
    pair0 = _sha_json([solutions[0], transposes[0]]) if solutions and transposes else None
    prep = sum(p['wall_seconds'] for p in phases if p['phase'] == 'prepare')
    fq = [p['wall_seconds'] for p in phases if 'forward-' in p['phase']]
    tq = [p['wall_seconds'] for p in phases if 'transpose-' in p['phase']]
    def med(v):
        return median(v) if v else None
    def tjson(x):
        return x if isinstance(x, (int, float, str, bool, type(None))) else str(x)
    Pcarrier = L+3*n*w
    out = {'n': n, 'w': w, 'field': name, 'seed': seed, 'R': R, 'mode': mode,
           'status': ('certificate_failure' if failure else ('singular' if singular else 'ok')),
           'family': family, 'terminal_backend': terminal_backend,
           'baseline': BASELINE_NOTE.get(mode),
           'L': L, 'P_carrier': Pcarrier, 'P_over_native_cap_512': Pcarrier > 512,
           'field_view': view,
           'det': tjson(det), 'singular': bool(singular),
           'prepare_seconds': prep,
           'forward_query_seconds': fq, 'transpose_query_seconds': tq,
           'median_forward_seconds': med(fq), 'median_transpose_seconds': med(tq),
           'total_seconds': sum(p['wall_seconds'] for p in phases),
           'phases': phases, 'verified': verified,
           'solutions_sha256': out_hash, 'pair0_sha256': pair0,
           'report': None if report is None else {k: tjson(v) if not isinstance(v, (list, tuple, dict)) else str(v)
                                                  for k, v in report.items()}}
    if failure:
        out['failure'] = failure
    if count_ops:
        out['count_path_differs_from_timed'] = True
        out['warnings'] = ['count_ops=True: CountingField hides the native packed polynomial hooks, so the algorithm '
                           'runs its generic scalar path (synthetic FFT above 64, no Kronecker products); operation '
                           'counts are honest for that path only and the times of this run must never be mixed with '
                           'timed runs (review-reuse-code.md finding 4).']
    return out
