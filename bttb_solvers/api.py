# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Public solver interfaces, named as in the accompanying paper.

Lags are ordered by outer diagonal, and each row by inner diagonal. For
banded input, lags[d+k][b+n-1] is a_{d,b}. For half-strip input the offset
is w instead of k and there are n+w rows. All vectors have n*n entries.
"""
from contextlib import ExitStack, contextmanager

PRIME = 998244353


def _validate(lags, n, width, banded=True):
    if type(n) is not int or type(width) is not int or not 1 <= width < n:
        raise ValueError('require integer n >= 2 and 1 <= width < n')
    expected = 2 * width + 1 if banded else n + width
    rows = [list(row) for row in lags]
    if len(rows) != expected or any(len(row) != 2*n-1 for row in rows):
        raise ValueError(f'expected {expected} lag rows of length {2*n-1}')
    return rows


@contextmanager
def _kernels(backend='flint', circulant=False, extension=False):
    if backend not in ('python', 'flint'):
        raise ValueError("backend must be 'python' or 'flint'")
    from ._internal.halfstrip_core.flint_backend import kernel_scope
    with ExitStack() as stack:
        if not extension:
            stack.enter_context(kernel_scope(backend))
        if circulant and (backend == 'flint' or extension):
            from ._internal import cc_kernels, extension_kernels
            stack.enter_context(extension_kernels.kernels() if extension else cc_kernels.kernels())
        yield


class PreparedSolver:
    """Prepared state with forward and literal transpose queries.

    Use as a context manager when working with Dense-LU, which owns native
    allocations. Solver/kernel contexts are process-global: use sequentially.
    """
    def __init__(self, state, name, backend='flint', *, circulant=False, extension=False,
                 forward=None, transpose=None, determinant=None, cleanup=None,
                 forward_many=None, transpose_many=None):
        self.state, self.name, self.backend = state, name, backend
        self._circulant, self._extension = circulant, extension
        self._forward = forward or state.solve
        self._transpose = transpose or state.solve_transpose
        self._determinant, self._cleanup = determinant, cleanup
        self._forward_many, self._transpose_many = forward_many, transpose_many
        self.closed = False

    def _call(self, fn, *args):
        if self.closed:
            raise RuntimeError('prepared solver is closed')
        with _kernels(self.backend, self._circulant, self._extension):
            return fn(*args)

    def solve(self, rhs):
        """Return x satisfying T*x = rhs."""
        return self._call(self._forward, list(rhs))

    def solve_transpose(self, rhs):
        """Return x satisfying transpose(T)*x = rhs, without conjugation."""
        return self._call(self._transpose, list(rhs))

    def solve_many(self, vectors, *, transpose=False):
        """Use a native batch where exposed; otherwise reuse state per vector."""
        fn = self._transpose_many if transpose else self._forward_many
        if fn is not None:
            return self._call(fn, [list(v) for v in vectors])
        method = self.solve_transpose if transpose else self.solve
        return [method(v) for v in vectors]

    def determinant(self):
        """Return the determinant only for routes exposing that service."""
        if self._determinant is None:
            raise NotImplementedError(f'{self.name} exposes solves only')
        return self._call(self._determinant)

    def close(self):
        if not self.closed:
            if self._cleanup is not None:
                self._cleanup()
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def root_of_unity(field, n):
    """Deterministically supply an exact-order root; acquisition is separate.

    Works over a supplied field with n dividing q-1. The search and field
    construction are not part of CC-E/CC-E-R timing in the paper.
    """
    if n < 1 or (field.q - 1) % n:
        raise ValueError('transform order must divide field size minus one')
    factors, d, m = [], 2, n
    while d*d <= m:
        if m % d == 0:
            factors.append(d)
            while m % d == 0: m //= d
        d += 1
    if m > 1: factors.append(m)
    if n == 1: return field.one()
    candidate = 1
    while candidate < field.q:
        candidate += 1
        if field.e == 1:
            x = field.from_int(candidate)
        else:
            x = field.from_int(sum((candidate+t)*field.p**t for t in range(field.e)))
        if field.is_zero(x): continue
        value = field.pow(x, (field.q-1)//n)
        if all(not field.eq(field.pow(value, n//f), field.one()) for f in factors):
            return value
    raise ValueError('no primitive root found in the supplied field')


def extension_field(p, degree):
    """Construct the supplied native GF(p**degree) adapter for CC-E/CC-E-R."""
    from ._internal.extension_field import FqFieldNative
    from ._internal.banded_bttb.counter import OpCounter
    return FqFieldNative(p, degree, OpCounter(), count_ops=False)


def _prepare_cc(lags, n, k, *, p, root, backend, retained, field=None):
    from ._internal.cc_prepared import prepare_cc as prepare
    from ._internal.banded_bttb.field import PrimeField
    from ._internal.banded_bttb.counter import OpCounter
    rows = _validate(lags, n, k)
    extension = field is not None
    if extension and backend != 'flint':
        raise ValueError('CC-E and CC-E-R use the native extension backend')
    with _kernels(backend, circulant=True, extension=extension):
        F = field if extension else PrimeField(p, OpCounter(), count_ops=False)
        if root is None:
            root = root_of_unity(F, n)
        raw = {(a, b): (F.ctx(int(rows[a+k][b+n-1])) if extension else rows[a+k][b+n-1])
               for a in range(-k, k+1) for b in range(1-n, n)}
        state = prepare(F, n, k, raw, root, arm='retained' if retained else 'original')
    def query(rhs, transpose=False):
        values = [F.ctx(int(v)) for v in rhs] if extension else rhs
        result = state.query(values, transpose=transpose)
        if not extension: return result
        answer = []
        for value in result:
            poly = value.polynomial()
            coefficients = [int(v) for v in poly]
            if any(coefficients[1:]):
                raise ArithmeticError('extension solution does not lie in the prime subfield')
            answer.append(coefficients[0] if coefficients else 0)
        return answer
    name = ('CC-E-R' if retained else 'CC-E') if extension else ('CC-R' if retained else 'CC')
    return PreparedSolver(state, name, backend, circulant=True, extension=extension,
                          forward=query, transpose=lambda b: query(b, True))


def prepare_cc(lags, n, k, *, p=PRIME, root=None, backend='flint'):
    """Prepare CC: prime-field circulant correction, with the paper's HB guards."""
    return _prepare_cc(lags,n,k,p=p,root=root,backend=backend,retained=False)


def prepare_cc_r(lags, n, k, *, p=PRIME, root=None, backend='flint'):
    """Prepare CC-R: CC retaining the four-vector frequency factors."""
    return _prepare_cc(lags,n,k,p=p,root=root,backend=backend,retained=True)


def prepare_cc_e(lags, n, k, *, field, root, backend='flint'):
    """Prepare CC-E over a supplied extension field/root; queries descend to F_p."""
    return _prepare_cc(lags,n,k,p=field.p,root=root,backend=backend,retained=False,field=field)


def prepare_cc_e_r(lags, n, k, *, field, root, backend='flint'):
    """Prepare CC-E-R, retaining factors over the supplied extension field."""
    return _prepare_cc(lags,n,k,p=field.p,root=root,backend=backend,retained=True,field=field)


def prepare_bfm(lags, n, k, *, p=PRIME, backend='flint', product='classical', strassen_leaf=16):
    """Prepare BFM boundary-front merge; product is classical or Strassen."""
    from ._internal.boundary_front.api import prepare_bfm as prepare
    from ._internal.boundary_front.fields import PrimeNative
    rows = _validate(lags,n,k)
    with _kernels(backend):
        state = prepare(PrimeNative(p),rows,n,dense_backend=product,
                        strassen_leaf=strassen_leaf if product=='strassen' else 1,residual_check='classical')
    return PreparedSolver(state,'BFM' if product=='classical' else 'BFM-Strassen',backend)


def prepare_gbe(lags, n, k, *, p=PRIME, backend='python'):
    """Prepare GBE guarded boundary elimination (the paper's comparator)."""
    from ._internal.guarded_boundary.unified import prepare_unified
    from ._internal.boundary_front.fields import PrimeNative, LeanAttrField
    rows = _validate(lags,n,k)
    with _kernels(backend):
        state = prepare_unified(LeanAttrField(PrimeNative(p)),rows,n,strategy='rootfree')
    return PreparedSolver(state,'GBE',backend)


def _halfstrip_rows(lags,n,w): return _validate(lags,n,w,banded=False)


def solve_hsc(lags,n,w,rhs=None,*,p=PRIME,backend='flint'):
    """HSC: determinant at every rank and one nonsingular forward solve.

    Repeats the full construction per call. The returned dictionary contains
    determinant, solution (or None) and singular. No singular solve is offered.
    """
    from ._internal.halfstrip_core.reuse import oneshot_solve, lags_to_base_blocks
    from ._internal.halfstrip_core.exact import PrimeField
    rows = _halfstrip_rows(lags,n,w)
    with _kernels(backend):
        base = PrimeField(p)
        blocks = lags_to_base_blocks('F'+str(p),base,rows,n)
        result = oneshot_solve(base,blocks,w,rhs)
    return {'determinant':result['det_base'],'solution':result['solution'],
            'singular':result['singular']}


def solve_hsc_q0(lags,n,w,rhs=None,*,p=PRIME,backend='flint'):
    """HSC-q0: guarded zero-corank precision reduction; refuses nonzero corank."""
    from ._internal.hsc_solver import solve_hsc as solve
    from ._internal.halfstrip_core.exact import PrimeField
    rows = _halfstrip_rows(lags,n,w)
    with _kernels(backend):
        result = solve(PrimeField(p),rows,n,w,rhs,precision_mode='q0_L')
    return {'determinant':result['det'],'solution':result['physical_solution'],
            'singular':result['singular']}


def prepare_hir(lags,n,w,*,p=PRIME,backend='flint'):
    """Prepare HIR impulse responses; requires an invertible outermost block."""
    from ._internal.hir_prepared import prepare_hir as prepare
    rows = _halfstrip_rows(lags,n,w)
    with _kernels(backend):
        state = prepare(p,rows,n,w,terminal_backend='dense',response_layout='n_impulse',count_ops=False)
    return PreparedSolver(state,'HIR',backend,determinant=lambda:state.det())


def prepare_hpr(lags,n,w,*,p=PRIME,backend='flint',terminal_query='lu'):
    """Prepare HPR without the outermost-block assumption.

    The paper's timed implementation uses terminal_query='lu'. 'division'
    and 'appended' are the separately checked terminal query comparisons.
    Their correctness/complexity correspondence is not upgraded by this API.
    """
    from ._internal.halfstrip_core.reuse import prepare_failed_h0_halfstrip
    from ._internal.halfstrip_core.exact import PrimeField
    rows = _halfstrip_rows(lags,n,w)
    with _kernels(backend):
        state = prepare_failed_h0_halfstrip(PrimeField(p),rows,n,w,terminal_query=terminal_query)
    return PreparedSolver(state,'HPR',backend,determinant=lambda:state.det)


def prepare_dense_lu(lags,n,width,*,kind='banded',p=PRIME):
    """One native FLINT factorization, batched solves and determinant.

    This is the exact paper baseline: pinned Linux x86-64 LP64 ABI and
    python-flint 0.9.0 / FLINT 3.6.0. It refuses unsupported platforms.
    """
    from ._internal.dense_lu import NativeLU
    rows = _validate(lags,n,width,banded=kind=='banded')
    problem={'field':{'kind':'prime','p':p},'dimensions':{'n':n,'k' if kind=='banded' else 'w':width},
             'raw':{'kind':kind,'rows':rows}}
    state = NativeLU.from_problem(problem)
    try:state.factorize()
    except BaseException:
        state.close()
        raise
    return PreparedSolver(state,'Dense-LU',forward=lambda b:state.solve([b])[0],
                          transpose=lambda b:state.solve([b],transpose=True)[0],
                          determinant=state.determinant,cleanup=state.close,
                          forward_many=state.solve,
                          transpose_many=lambda bs:state.solve(bs,transpose=True))


def solve_dedicated(lags,n,k,forward_rhs,transpose_rhs,*,p=PRIME,
                    backend='flint',product='strassen',strassen_leaf=8):
    """Dedicated banded proof-of-concept, including the Strassen route.

    Pays the complete construction for one forward/transpose pair. This is
    the bounded comparison from the exponent discussion, not an executable
    realization of the general fast-exponent theorem. The statistics include
    the explicit dense-kernel ledger.
    """
    from ._internal.dedicated_banded_arithmetic import solve_dedicated_banded
    from ._internal.halfstrip_core.exact import PrimeField
    rows=_validate(lags,n,k)
    raw={(a,b):rows[a+k][b+n-1] for a in range(-k,k+1) for b in range(1-n,n)}
    with _kernels(backend):
        return solve_dedicated_banded(PrimeField(p),n,k,raw,list(forward_rhs),list(transpose_rhs),
            dense_backend=product,strassen_leaf=strassen_leaf if product=='strassen' else 1)
