# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from contextlib import contextmanager

from bttb_solvers._internal.halfstrip_core import dense_backend
from bttb_solvers._internal.halfstrip_core.uniform_terminal import _rectangular, _classical, _square, _count


@contextmanager
def backend_scope(dense='classical', strassen_leaf=1):
    """Open a dense_scope; yields its ledger.  Nested scopes restore the outer one."""
    with dense_backend.dense_scope(dense, strassen_leaf, polynomial_entry_policy="off") as ledger:
        yield ledger


def merge_ledger(into, ledger):
    for key, value in ledger.items():
        into[key] = into.get(key, 0) + value


def shape(A):
    return len(A), (len(A[0]) if A else 0)


def zeros(F, rows, cols):
    z = F.zero
    return [[z() for _ in range(cols)] for _ in range(rows)]


def padded_side(b, leaf):
    """Least c * 2**j >= b with c = ceil(b / 2**j) <= leaf (j minimal)."""
    j = 0
    while -(-b // (1 << j)) > leaf:
        j += 1
    return -(-b // (1 << j)) << j


def _slice(X, i, nr, k, nk):
    return [row[k:k + nk] for row in X[i:i + nr]]


def _strassen_product(F, A, B, rows, inner, cols, leaf, stats):
    block = min(rows, inner, cols)
    dense_backend.record("dense_product_calls")
    dense_backend.record("constant_matrix_product_calls")
    _count(stats, "product_calls")
    if block <= leaf:

        dense_backend.record("strassen_small_dim_classical_calls")
        from bttb_solvers._internal.halfstrip_core.flint_backend import matmul as flint_matmul
        accelerated = flint_matmul(F,A,B)
        if accelerated is not None:
            dense_backend.record("flint_nmod_mat_product_calls")
            dense_backend.record("flint_compiled_scalar_counts_unavailable")
            return accelerated
        return _classical(A, B, rows, inner, cols, F.zero, F.add, F.mul, stats)
    padded = padded_side(block, leaf)
    zero, add = F.zero, F.add
    out = [[None] * cols for _ in range(rows)]
    for i in range(0, rows, block):
        nr = min(block, rows - i)
        for j in range(0, cols, block):
            nc = min(block, cols - j)
            for k in range(0, inner, block):
                nk = min(block, inner - k)
                if nr == nk == nc == block:
                    if rows == inner == padded and i == k == 0:
                        X = A
                    else:
                        _count(stats, "copy_cells", padded * padded)
                        X = [[A[i + u][k + v] if (u < block and v < block) else zero()
                              for v in range(padded)] for u in range(padded)]
                    if inner == cols == padded and k == j == 0:
                        Y = B
                    else:
                        _count(stats, "copy_cells", padded * padded)
                        Y = [[B[k + u][j + v] if (u < block and v < block) else zero()
                              for v in range(padded)] for u in range(padded)]
                    from bttb_solvers._internal.halfstrip_core.flint_backend import enabled_prime
                    Z = (_flint_square(F,X,Y,leaf) if enabled_prime(F) is not None
                         else _square(X, Y, add, F.sub, F.mul, stats, zero))
                else:
                    Z = matmul(F, _slice(A, i, nr, k, nk), _slice(B, k, nk, j, nc), stats=stats)
                for u in range(nr):
                    row, zrow = out[i + u], Z[u]
                    if k == 0:
                        for v in range(nc):
                            row[j + v] = zrow[v]
                    else:
                        for v in range(nc):
                            row[j + v] = add(row[j + v], zrow[v])
    return out


def matmul(F, A, B, *, stats=None):
    """A (r x i) times B (i x c), lists of rows; dimensions checked by the caller's shapes."""
    rows = len(A)
    inner = len(B)
    cols = len(B[0]) if B else 0
    if rows and len(A[0]) != inner:
        raise ValueError("incompatible matmul shapes")
    if rows == 0 or cols == 0:
        return [[F.zero() for _ in range(cols)] for _ in range(rows)]
    if inner == 0:
        return zeros(F, rows, cols)
    from bttb_solvers._internal.halfstrip_core.flint_backend import matmul as flint_matmul
    backend, leaf = dense_backend.selection()
    from bttb_solvers._internal.halfstrip_core.flint_backend import enabled_prime
    if stats is not None and enabled_prime(F) is not None:
        stats["flint_compiled_scalar_counts_unavailable"] = True
    if backend == "classical":
        accelerated = flint_matmul(F, A, B)
        if accelerated is not None:
            dense_backend.record("flint_nmod_mat_product_calls")
            dense_backend.record("flint_compiled_scalar_counts_unavailable")
            return accelerated
    if backend == "classical":
        return _rectangular(A, B, rows, inner, cols, F.zero, F.add, F.sub, F.mul, stats)
    return _strassen_product(F, A, B, rows, inner, cols, leaf, stats)


def matvec(F, A, x):
    """Classical matrix-vector product (root query and the per-occurrence reference path)."""
    out = []
    for row in A:
        total = F.zero()
        for u, v in zip(row, x):
            total = F.add(total, F.mul(u, v))
        out.append(total)
    return out


def add_matrix(F, X, Y):
    return [[F.add(u, v) for u, v in zip(a, b)] for a, b in zip(X, Y)]


def sub_matrix(F, X, Y):
    return [[F.sub(u, v) for u, v in zip(a, b)] for a, b in zip(X, Y)]


def neg_matrix(F, X):
    return [[F.neg(u) for u in a] for a in X]


def _flint_square(F,A,B,leaf):
    """Same seven-product division-free Strassen schedule, compiled leaves."""
    n=len(A)
    if n<=leaf:
        from bttb_solvers._internal.halfstrip_core.flint_backend import matmul as flint_matmul
        dense_backend.record("flint_nmod_mat_product_calls")
        dense_backend.record("flint_compiled_scalar_counts_unavailable")
        return flint_matmul(F,A,B)
    dense_backend.record("strassen_nodes")
    h=n//2
    a,b,c,d=(_slice(A,0,h,0,h),_slice(A,0,h,h,h),
             _slice(A,h,h,0,h),_slice(A,h,h,h,h))
    e,f,g,i=(_slice(B,0,h,0,h),_slice(B,0,h,h,h),
             _slice(B,h,h,0,h),_slice(B,h,h,h,h))
    add=lambda x,y:add_matrix(F,x,y)
    sub=lambda x,y:sub_matrix(F,x,y)
    mul=lambda x,y:_flint_square(F,x,y,leaf)
    p=mul(add(a,d),add(e,i));q=mul(add(c,d),e)
    r=mul(a,sub(f,i));s=mul(d,sub(g,e))
    t=mul(add(a,b),i);u=mul(sub(c,a),add(e,f));v=mul(sub(b,d),add(g,i))
    aa=add(sub(add(p,s),t),v);bb=add(r,t)
    cc=add(q,s);dd=add(sub(add(p,r),q),u)
    return [aa[j]+bb[j] for j in range(h)]+[cc[j]+dd[j] for j in range(h)]
