# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .errors import (HalfStripError, SingularLeadingBlock, SingularTerminalMatrix,
                     NotPrepared, NoNTTRoot, WidthOutOfRange)
from .leading import LeadingInverse
from .dense import LU
from .symbol import CONVENTION
from .convolve import ConvolutionBackend, next_pow2
from . import recurrence


class HalfStripSolver(object):
    """Reusable half-strip operator with explicit response layout selection.

    wn_homogeneous retains the historical baseline. n_impulse prepares n
    length-(n+w) causal inverse impulses, reuses temporal shifts for K, and
    applies initial-data reconstruction with one response per query. The
    dense terminal is a comparison arm; structured_classical avoids dense K.
    """
    def __init__(self, sym, cnt, route="cdq", require_ntt_field=False,
                 chart_note=None, convolution_backend="auto",terminal_backend="dense",
                 response_layout="wn_homogeneous"):
        """Stores inputs.  Does NO arithmetic - all work is in prepare()."""
        self.sym = sym
        self.F = sym.F
        self.n = sym.n
        self.w = sym.w
        self.N = sym.n * sym.n
        self.cnt = cnt
        self.route = route
        self.require_ntt_field = require_ntt_field
        self.chart_note = chart_note
        self.convolution_backend = convolution_backend
        self.backend = None
        self.prepared = False
        self.report = None
        self.leading = None
        self.terminal = None
        self.H = None
        if terminal_backend not in ('dense','structured_classical'):
            raise ValueError('terminal_backend must be dense or structured_classical')
        self.terminal_backend=terminal_backend
        if response_layout not in ('wn_homogeneous', 'n_impulse'):
            raise ValueError('response_layout must be wn_homogeneous or n_impulse')
        if response_layout == 'n_impulse' and route != 'cdq':
            raise ValueError('n_impulse currently requires cdq')
        self.response_layout = response_layout


    def prepare(self):
        if self.prepared:
            return self.report
        F, n, w, cnt = self.F, self.n, self.w, self.cnt
        L = n + w
        rep = {"theorem": "algorithm (a)-(e)", "field": F.name(),
               "q": getattr(F, "q", None),
               "n": n, "w": w, "N": self.N, "L": L, "route": self.route,
               "convention": CONVENTION,
               "leading_backend": "cyclic-displacement cyclic-displacement compact representation "
                                  "(4 vectors, 4 convolutions per application)",
               "H0_det_B_minus_w_nonzero": None,
               "H1_gs_chart": "NOT A HYPOTHESIS OF THIS BUILD -- deleted per "
                              "oc1_reledger.md sections 2 and 5; there is no "
                              "exception for it and no code path consults it",
               "H2_terminal_nonsingular": None,
               "H3_cost_model": ("algebraic RAM over a finite field carrying the "
                                 "required radix-two roots of unity; only an actual "
                                 "NTT run is labelled achieved. Q uses only the "
                                 "schoolbook correctness oracle and carries no "
                                 "cubic or bit-complexity claim"),
               "seeds": {"library_randomness": "none - this package contains no "
                                               "randomness at all"},
               "chart_note": self.chart_note}

        with cnt.stage("setup"):
            temporal_order = next_pow2(2 * L)
            spatial_order = next_pow2(3 * n - 2)
            requested = self.convolution_backend
            if self.require_ntt_field:
                if requested == "schoolbook_oracle":
                    raise ValueError("require_ntt_field=True conflicts with the "
                                     "schoolbook_oracle backend")
                requested = "ntt"
            self.backend = ConvolutionBackend(
                F, cnt, requested, (temporal_order, spatial_order))
        rep["ntt_orders_required"] = {
            "temporal": temporal_order,
            "spatial": spatial_order,
            "combined": max(temporal_order, spatial_order),
        }
        rep["convolution_backend"] = self.backend.summary()
        rep["cubic_ledger_achieved"] = (
            self.route == "cdq" and self.backend.achieved)
        if self.route != "cdq":
            rep["convolution_backend"]["cubic_ledger_achieved"] = False
            rep["convolution_backend"]["reason"] = (
                "direct recurrence oracle selected; not a cubic implementation")


        with cnt.stage("leading"):
            self.leading = LeadingInverse(
                F, self.sym.leading_lag(), n, cnt, backend=self.backend)
        rep["H0_det_B_minus_w_nonzero"] = True
        rep["det_B_minus_w"] = F.to_str(self.leading.det)
        rep["leading_setup"] = {
            "solves": 4,
            "note": "four Toeplitz solves (two with T, two with T^T), one-time on "
                    "the single matrix B_{-w}; MEASURED O(n^3) by dense elimination, "
                    "MODEL 4n^2 by the R-1 charge. det B_{-w} comes from the same "
                    "dense LU, NOT from a Levinson pivot product (which is the "
                    "chart-dependent object -- oc1_reledger.md section 6)."}


        nh = w * n
        zero = [F.zero()] * n
        stats = {}
        rep['response_layout'] = self.response_layout
        if self.response_layout == 'n_impulse':
            from bttb_solvers._internal.halfstrip_structured_terminal import causal_solve, DenseImpulseTerminal
            forcing = [[[F.one() if t == 0 and i == c else F.zero()
                         for i in range(n)] for c in range(n)] for t in range(L)]
            impulse = causal_solve(F, self.sym, self.leading, self.backend, cnt,
                                   forcing, stats=stats)

            
            self.impulse = impulse
            with cnt.stage('terminal'):
                if self.terminal_backend == 'structured_classical':
                    from bttb_solvers._internal.halfstrip_structured_terminal import StructuredTerminal
                    self.terminal = StructuredTerminal(F,self.sym,self.leading,
                        self.backend,cnt,None,inverse_responses=impulse)
                    self.Bterm = None
                else:
                    self.terminal = DenseImpulseTerminal(F,self.sym,self.leading,
                        self.backend,cnt,impulse)
                    self.Bterm = self.terminal.matrix
            self.H = None
            rep.update(homogeneous_responses=0, impulse_responses=n,
                       response_horizon=L, response_entries=L*n*n,
                       merges=stats.get('merges',0), kernel_shapes=None,
                       leaf_inverse_applications_prepare=stats['leaf_applications'],
                       terminal_matrix_kind='K' if self.Bterm is not None else None,
                       terminal_matrix_materialized=self.Bterm is not None)
        else:
            init = [[None] * nh for _ in range(w)]
            for t in range(w):
                for c in range(nh):
                    u, v = divmod(c, n)
                    e = [F.zero()] * n
                    if u == t:
                        e[v] = F.one()
                    init[t][c] = e
            R = [[list(zero) for _ in range(nh)] for _ in range(n)]
            Xh = recurrence.run(F, self.sym, cnt, self.leading, init, R,
                                route=self.route, stats=stats, backend=self.backend)
            rep['homogeneous_responses'] = nh
            rep['merges'] = stats['merges']
            rep['kernel_shapes'] = stats['kernel_shapes']
            rep['leaf_inverse_applications_prepare'] = stats['leaf_applies']
            with cnt.stage('terminal'):
                if self.terminal_backend=='structured_classical':
                    from bttb_solvers._internal.halfstrip_structured_terminal import StructuredTerminal
                    self.terminal=StructuredTerminal(F,self.sym,self.leading,self.backend,cnt,Xh)
                    Bt=None
                    rep['terminal_backend']=self.terminal.summary()
                    rep['terminal_matrix_materialized']=False
                else:
                    Bt = [[Xh[n + a][c][s] for c in range(nh)]
                          for a in range(w) for s in range(n)]
                    self.terminal = LU(F, Bt)
            self.Bterm = Bt
            self.H = [[Xh[i][c][s] for c in range(nh)] for i in range(n) for s in range(n)]
        rep["terminal_order"] = nh
        rep["H2_terminal_nonsingular"] = not self.terminal.singular
        if self.terminal.singular:
            rep["terminal_nullity"] = self.terminal.nullity()
            rep["certificate"] = ("terminal matrix singular: by algorithm (a), "
                                  "nullity(T) = nullity(B), so det T = 0 with this "
                                  "nullity. solve() raises SingularTerminalMatrix; "
                                  "det() returns zero.")
        rep["ops"] = cnt.report(n=n, w=w)
        self.report = rep
        self.prepared = True
        return rep

    def _require(self):
        if not self.prepared:
            raise NotPrepared("call prepare() first")
        if self.terminal.singular:
            raise SingularTerminalMatrix(order=self.w * self.n,
                                         nullity=self.terminal.nullity())


    def _particular(self, b):
        """The single particular response: the CDQ at q = 1, then the terminal
        elimination and the reconstruction."""
        F, n, w = self.F, self.n, self.w
        zero = [F.zero()] * n
        init = [[list(zero)] for _ in range(w)]
        R = [[[b[i * n + s] for s in range(n)]] for i in range(n)]
        Xp = recurrence.run(F, self.sym, self.cnt, self.leading, init, R,
                            route=self.route, backend=self.backend)
        r = [Xp[n + a][0][s] for a in range(w) for s in range(n)]
        with self.cnt.stage("terminal"):
            alpha = self.terminal.solve([F.neg(x) for x in r])
        if self.response_layout == 'n_impulse':

            
            init = [[list(alpha[t*n:(t+1)*n])] for t in range(w)]
            zero_forcing = [[list(zero)] for _ in range(n)]
            Xh = recurrence.run(F,self.sym,self.cnt,self.leading,init,zero_forcing,
                               route=self.route,backend=self.backend)
            with self.cnt.stage('reconstruct'):
                out = [F.add(Xp[i][0][s],Xh[i][0][s]) for i in range(n) for s in range(n)]
        else:
            with self.cnt.stage("reconstruct"):
                out = []
                for i in range(n):
                    for s in range(n):
                        acc = Xp[i][0][s]
                        row = self.H[i * n + s]
                        for c, av in enumerate(alpha):
                            acc = F.add(acc, F.mul(row[c], av))
                        out.append(acc)
        return out, r, alpha

    def solve(self, b):
        """x with T x = b.  raises SingularTerminalMatrix if det T == 0 - that IS
        the certificate, not a crash."""
        self._require()
        if len(b) != self.N:
            raise ValueError("right-hand side of length %d expected" % self.N)
        x, _, _ = self._particular(b)
        return x

    def _reverse(self, v):
        """Pi v, Pi the reversal (i, s) -> (n-1-i, n-1-s) of algorithm (c)."""
        n = self.n
        return [v[(n - 1 - i) * n + (n - 1 - s)] for i in range(n) for s in range(n)]

    def solve_transpose(self, c):
        """y with T^T y = c.

        algorithm (c), THE CHEAP ROUTE: T is persymmetric in the composite index,
        T^T = Pi T Pi, hence T^{-T} = Pi T^{-1} Pi.  Reverse the right-hand side
        in both axes, run the SAME solver on the SAME operator and the SAME
        factorisation, reverse the output.  NO new setup, NO second backend, NO
        new hypothesis - and, since (H1) is gone, no hypothesis to transfer
        either (the H1-transfer sentence of (c) is vacuous here:
        oc1_reledger.md section 6)."""
        self._require()
        if len(c) != self.N:
            raise ValueError("right-hand side of length %d expected" % self.N)
        with self.cnt.stage("transpose"):
            rc = self._reverse(c)
        y, _, _ = self._particular(rc)
        with self.cnt.stage("transpose"):
            return self._reverse(y)


    def det(self):
        """det T = (-1)^{w n^3} det(B_{-w})^n det B, algorithm (b), exactly.

        Returns zero when the terminal matrix is singular - which, by (a), is
        correct and not an evasion."""
        if not self.prepared:
            raise NotPrepared("call prepare() first")
        F, n, w = self.F, self.n, self.w
        with self.cnt.stage("det"):
            d = F.mul(F.pow(self.leading.det, n), self.terminal.det)
            if (w * n ** 3) % 2:
                d = F.neg(d)
            return d

    def mark(self, b, u, eps):
        """(det(T + eps b u^T), u^T T^{-1} b) = (det T * (1 + eps t), t).

        THE eps IS CARRIED and is a FIELD ELEMENT, not an infinitesimal: the
        u-gradient of the marked determinant is eps * det T * T^{-1} b, while
        det T * T^{-1} b is the u-gradient of the eps-COEFFICIENT.  Returning
        only the second and calling it 'the mark' is the algorithm (c) ambiguity
        and is forbidden here (benchmark.md)."""
        self._require()
        if len(b) != self.N:
            raise ValueError("right-hand side of length %d expected" % self.N)
        if len(u) != self.N:
            raise ValueError("mark vector u has length %d; expected %d"
                             % (len(u), self.N))
        F = self.F
        x = self.solve(b)
        d = self.det()
        with self.cnt.stage("mark"):
            t = F.zero()
            for ui, xi in zip(u, x):
                t = F.add(t, F.mul(ui, xi))
            marked = F.mul(d, F.add(F.one(), F.mul(eps, t)))
        return marked, t


    def terminal_matrix(self):
        """Dense diagnostic: B for wn_homogeneous, K for n_impulse; None if structured."""
        if not self.prepared:
            raise NotPrepared("call prepare() first")
        return self.Bterm

    def terminal_nullity(self):
        if not self.prepared:
            raise NotPrepared("call prepare() first")
        return self.terminal.nullity()

    def counts(self):
        """Current measured/model/backend counts, including post-prepare queries."""
        out = self.cnt.report(n=self.n, w=self.w)
        out["convolution_backend"] = self.backend.summary() if self.backend else None
        out["cubic_ledger_achieved"] = bool(
            self.backend and self.backend.achieved and self.route == "cdq")
        if self.backend and self.route != "cdq":
            out["convolution_backend"]["cubic_ledger_achieved"] = False
            out["convolution_backend"]["reason"] = (
                "direct recurrence oracle selected; not a cubic implementation")
        return out

    stagewise_counts = counts

