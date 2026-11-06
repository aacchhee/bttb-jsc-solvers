# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Trial/shared. Block-Cauchy representation supporting alternate structured recurrence terminals.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.banded_bttb.field import int_element
from .block_levinson import add, matmul, zeros, solve_matrix, materialize
from .dense import LU
from .convolve import ConvolutionBackend


class CauchyChartBreakdown(ArithmeticError):
    def __init__(self, start, width):
        self.section = (start, width)
        super().__init__("singular recursive Fourier-Cauchy pivot at %r" % (self.section,))


class UnsupportedCauchyWidth(ValueError):
    pass


def _transpose(A):
    return [list(r) for r in zip(*A)]


def _scale(F, A, c):
    return [[F.mul(c, v) for v in row] for row in A]


class _Representation:
    """Block Cauchy generators on two contiguous intervals of shared grids."""
    def __init__(self, owner, G, H, rs, cs, a, b):
        self.o, self.G, self.H = owner, G, H
        self.rs, self.cs, self.a, self.b = rs, cs, a, b

    def apply(self, X):
        o, F = self.o, self.o.F
        nr, nc, rank, q = len(self.G), len(self.H), len(self.H[0]), len(X[0][0])
        if len(X) != nc:
            raise ValueError("wrong structured right-hand-side block count")
        V = [matmul(F, h, x) for h, x in zip(self.H, X)]

        ratio = F.div(self.b, self.a)
        offset = self.cs-self.rs
        cyclic = nr == nc == o.width
        if cyclic:
            h = [F.inv(F.sub(F.one(), F.mul(ratio, o.powers[(-k) % o.width])))
                 for k in range(o.width)]
        else:

            h = [F.inv(F.sub(F.one(), F.mul(ratio,
                       o.powers[(offset+nc-1-k) % o.width])))
                 for k in range(nr+nc-1)]
        Y = [zeros(F, rank, q) for _ in range(nr)]
        for r in range(rank):
            for c in range(q):
                v = [x[r][c] for x in V]
                if cyclic:
                    be = o.backend
                    hv, vv = be._ntt(h), be._ntt(v)
                    z = be._ntt([F.mul(x,y) for x,y in zip(hv,vv)], inverse=True)
                    if be.cnt is not None:
                        be.cnt.charge_conv(o.width, engine="ntt", transforms=3)
                else:
                    z = o.backend.linear(h, v)[nc-1:nc-1+nr]
                o.stats["scalar_convolutions"] += 1
                o.stats["max_convolution_order"] = max(o.stats["max_convolution_order"],
                    o.width if cyclic else 1 << (len(h)+len(v)-2).bit_length())
                for i in range(nr):
                    Y[i][r][c] = z[i]
        out = [matmul(F, g, y) for g,y in zip(self.G,Y)]
        for i in range(nr):
            out[i] = _scale(F, out[i], F.inv(F.mul(self.a, o.powers[(self.rs+i) % o.width])))
        o.stats["structured_applications"] += 1
        return out

    def left_apply(self, X):

        F = self.o.F
        trans = _Representation(self.o,
            [_scale(F, _transpose(h), F.neg(F.one())) for h in self.H],
            [_transpose(g) for g in self.G], self.cs,self.rs,self.b,self.a)
        return [_transpose(y) for y in trans.apply([_transpose(x) for x in X])]


class BlockCauchy:
    """LU-compatible terminal factor with an explicit recursive pivot chart."""
    def __init__(self, F, lags, width, backend=None, fallback="raise"):
        if fallback not in ("raise", "dense"):
            raise ValueError("fallback must be 'raise' or 'dense'")
        if width < 1 or set(lags) != set(range(1-width,width)):
            raise ValueError("all 2w-1 lag blocks are required")
        self.F,self.lags,self.width = F,lags,width
        self.block_size = len(lags[0])
        n = self.block_size
        if not n or any(len(A)!=n or any(len(row)!=n for row in A) for A in lags.values()):
            raise ValueError("lag blocks must be nonempty square matrices")
        self.n = width*n
        self.dense = None
        self.breakdown_section = None
        self.stats = {"recursive_nodes":0,"leaf_blocks":0,"structured_applications":0,
                      "scalar_convolutions":0,"max_convolution_order":0,
                      "outer_scalar_transforms":0,"outer_transform_order":width}
        if width & (width-1):
            if fallback == "raise":
                raise UnsupportedCauchyWidth("Fourier-Cauchy backend requires a dyadic width")
            self._dense_fallback("unsupported non-dyadic width")
            return

        cnt = getattr(backend,"cnt",getattr(F,"cnt",None))
        self.backend = ConvolutionBackend(F,cnt,"ntt",(2*width,))
        self.eta = self.backend._root(2*width)
        omega = F.mul(self.eta,self.eta)
        self.powers = [F.one()]
        for i in range(1,width):
            self.powers.append(F.mul(self.powers[-1],omega))
        self.eta_powers = [F.one()]
        inveta = F.inv(self.eta)
        for i in range(1,width):
            self.eta_powers.append(F.mul(self.eta_powers[-1],inveta))
        self.invw = F.inv(int_element(F,width))
        I = [[F.one() if i==j else F.zero() for j in range(n)] for i in range(n)]
        Graw,Hraw = [],[]
        for i in range(width):
            C = zeros(F,n) if i==0 else add(F,lags[i-width],lags[i])
            Graw.append([(I[r] if i==0 else [F.zero()]*n)+C[r] for r in range(n)])
            R = (add(F,lags[width-1-i],lags[-i-1],True) if i<width-1
                 else _scale(F,lags[0],int_element(F,2)))
            Hraw.append(R+(I if i==width-1 else zeros(F,n)))
        G = self._transform(Graw, inverse=False, normalize=True)
        H = self._transform([_scale(F,h,p) for h,p in zip(Hraw,self.eta_powers)],
                            inverse=True, normalize=False)
        try:
            self.inverse,self.det = self._invert(G,H,0,F.one(),self.eta)
        except CauchyChartBreakdown as exc:
            self.breakdown_section = exc.section
            if fallback == "raise":
                raise
            self._dense_fallback("recursive pivot breakdown")
            return
        correction = F.pow(self.eta,n*width*(width-1)//2)
        self.det = F.mul(self.det,correction)
        self.singular,self.rank = False,self.n

    def _transform(self, X, inverse, normalize):
        F,w = self.F,self.width
        nr,nc = len(X[0]),len(X[0][0])
        self.stats["outer_scalar_transforms"] += nr*nc
        Y = [zeros(F,nr,nc) for _ in range(w)]
        scale = (F.one() if inverse and normalize else
                 int_element(F,w) if inverse else self.invw if normalize else F.one())
        for r in range(nr):
            for c in range(nc):
                v = self.backend._ntt([x[r][c] for x in X],inverse=inverse)
                for i in range(w):
                    Y[i][r][c] = F.mul(scale,v[i])
        return Y

    def _invert(self,G,H,start,a,b):
        F,m = self.F,len(G)
        self.stats["recursive_nodes"] += 1
        if m==1:
            denom = F.sub(F.mul(a,self.powers[start]),F.mul(b,self.powers[start]))
            A = _scale(F,matmul(F,G[0],H[0]),F.inv(denom))
            lu = LU(F,A)
            if lu.singular:
                raise CauchyChartBreakdown(start,1)
            self.stats["leaf_blocks"] += 1
            P = _scale(F,solve_matrix(lu,G[0]),F.neg(F.one()))
            U = [lu.solve_transpose(list(row)) for row in H[0]]
            return _Representation(self,[P],[U],start,start,b,a),lu.det
        h = m//2
        G1,G2,H1,H2 = G[:h],G[h:],H[:h],H[h:]
        Ai,da = self._invert(G1,H1,start,a,b)
        B = _Representation(self,G1,H2,start,start+h,a,b)
        C = _Representation(self,G2,H1,start+h,start,a,b)
        Gs = [add(F,g,x) for g,x in zip(G2,C.apply(Ai.G))]
        Hs = [add(F,hv,x,True) for hv,x in zip(H2,B.left_apply(Ai.H))]
        Si,ds = self._invert(Gs,Hs,start+h,a,b)
        top = [add(F,p,x,True) for p,x in zip(Ai.G,Ai.apply(B.apply(Si.G)))]
        left = [add(F,u,x,True) for u,x in zip(Ai.H,Ai.left_apply(C.left_apply(Si.H)))]
        return _Representation(self,top+Si.G,left+Si.H,start,start,b,a),F.mul(da,ds)

    def _dense_fallback(self,reason):
        self.fallback_reason = reason
        self.dense = LU(self.F,materialize(self.lags,self.width))
        self.det,self.singular,self.rank = self.dense.det,self.dense.singular,self.dense.rank

    def solve(self,b):
        if len(b)!=self.n:
            raise ValueError("wrong terminal right-hand-side length")
        if self.dense is not None:
            return self.dense.solve(b)
        n,F = self.block_size,self.F
        X = [[[v] for v in b[i*n:(i+1)*n]] for i in range(self.width)]
        Y = self.inverse.apply(self._transform(X,inverse=False,normalize=True))
        Z = self._transform(Y,inverse=True,normalize=False)
        return [F.mul(p,row[0]) for p,x in zip(self.eta_powers,Z) for row in x]

    def nullity(self):
        return self.n-self.rank

    def summary(self):
        dense = self.dense is not None
        return {"kind":"dense_fallback" if dense else "block_fourier_cauchy",
                "terminal_rate":"O(w^3 n^3)" if dense else "O~(w n^3)",
                "published_rate_achieved":not dense,
                "normal_chart_rate_achieved":not dense,
                "all_input_fast_rate_achieved":False,
                "scope":"finite-field dyadic Fourier-Cauchy normal chart",
                "full_terminal_materialized":dense,
                "stored_scalar_cells":self.n**2 if dense else
                    ((2*self.width-1)+4*self.width)*self.block_size**2+2*self.width,
                "chart":"dyadic width, primitive 2w-th root, nonsingular recursive Fourier-Cauchy block pivots",
                "breakdown_section":self.breakdown_section,
                "fallback_reason":getattr(self,"fallback_reason",None),
                "ntt_order_required":2*self.width,
                "displacement_rank_bound":2*self.block_size,
                "chart_counters":dict(self.stats),"structured_zero_skipping":False}
