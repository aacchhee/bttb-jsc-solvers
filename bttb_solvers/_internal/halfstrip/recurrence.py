# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .convolve import ConvolutionBackend, next_pow2


def _kernel(sym, length):
    """kernel[d] = A[:, d-w] for d = 1 .. length-1, the raw common lag array with
    d = 0 excluded, exactly as algorithm (d) specifies."""
    return [None] + [sym.lag(d - sym.w) for d in range(1, length)]


def run(F, sym, cnt, leading, init, R, route="cdq", stats=None, backend=None):
    """Generate the extended rows for a batch of responses.

    `init[t][c]` for t = 0 .. w-1 : the free initial rows, one n-vector per
        response c.
    `R[i][c]` for i = 0 .. n-1    : the physical forcing, one n-vector per
        response c (pass zero vectors for a homogeneous response).
    Returns `X[t][c]`, t = 0 .. L-1.

    Stages: every merge is charged to `merge`; every application of
    `B_{-w}^{-1}` to a completed row is charged to `apply_inv`.  Nothing else in
    this function performs field arithmetic.
    """
    n, w = sym.n, sym.w
    L = n + w
    nresp = len(init[0]) if w else len(R[0])
    zero = [F.zero()] * n

    X = [[None] * nresp for _ in range(L)]
    C = [[list(zero) for _ in range(nresp)] for _ in range(L)]
    for t in range(w):
        for c in range(nresp):
            X[t][c] = list(init[t][c])
    for i in range(n):
        for c in range(nresp):
            C[i + w][c] = list(R[i][c])

    st = stats if stats is not None else {}
    st.setdefault("merges", 0)
    st.setdefault("leaf_applies", 0)
    st.setdefault("kernel_shapes", set())

    kernel_full = _kernel(sym, L)
    if backend is None:
        backend = ConvolutionBackend(
            F, cnt, "auto", (next_pow2(2 * L), next_pow2(3 * n - 2)))

    def leaf(t):
        if t < w:
            return
        with cnt.stage("apply_inv"):
            for c in range(nresp):
                X[t][c] = leading.apply(C[t][c])
        st["leaf_applies"] += nresp

    def merge(lo, mid, hi):
        length = hi - lo
        left = mid - lo
        first = max(mid, w)
        if first >= hi:
            return
        with cnt.stage("merge"):
            kern = kernel_full[:length]
            kernel_fresh = False
            for c in range(nresp):
                sources = [X[lo + a][c] for a in range(left)]
                out, fresh_now = backend.merge_window(
                    sources, kern, n, first - lo, hi - lo, left)
                kernel_fresh = kernel_fresh or fresh_now
                for tau, vec in out.items():
                    tgt = C[lo + tau][c]
                    for s in range(n):
                        tgt[s] = F.sub(tgt[s], vec[s])
            nt = next_pow2(left + length - 1)
            ns = next_pow2(3 * n - 2)
            shape = (length, nt, ns)
            fresh = shape not in st["kernel_shapes"]
            st["kernel_shapes"].add(shape)

            

            
            model_kernel_fresh = (kernel_fresh if backend.kind == "ntt"
                                  else fresh)
            cnt.charge_conv2d(nresp, nt, ns,
                              kernel_cells=(nt * ns
                                            if model_kernel_fresh else 0),
                              raw=(left + length - 1, 3 * n - 2),
                              engine=backend.kind,
                              kernel_fresh=kernel_fresh)
        st["merges"] += 1

    if route == "cdq":
        def cdq(lo, hi):
            if hi - lo == 1:
                leaf(lo)
                return
            mid = (lo + hi) // 2
            cdq(lo, mid)
            merge(lo, mid, hi)
            cdq(mid, hi)
        cdq(0, L)
    elif route == "direct":

        
        for t in range(L):
            if t >= w:
                with cnt.stage("merge"):
                    for c in range(nresp):
                        tgt = C[t][c]
                        for d in range(1, t + 1):
                            lag = sym.lag(d - w)
                            src = X[t - d][c]
                            for q in range(n):
                                x = src[q]
                                for s in range(n):
                                    y = lag[(s - q) + (n - 1)]
                                    tgt[s] = F.sub(tgt[s], F.mul(x, y))
            leaf(t)
    else:
        raise ValueError("route must be 'cdq' or 'direct', got %r" % (route,))

    st["kernel_shapes"] = len(st["kernel_shapes"])
    return X
