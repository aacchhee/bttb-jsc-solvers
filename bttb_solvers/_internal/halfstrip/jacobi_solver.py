# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from time import perf_counter

from .convolve import ConvolutionBackend, next_pow2
from .leading import LeadingInverse
from .dense import LU
from .block_levinson import BlockLevinson, materialize
from .errors import NotPrepared, SingularTerminalMatrix
from .symbol import CONVENTION
from .jacobi_series import build_series, apply_window


class JacobiHalfStripSolver:
    def __init__(self, sym, cnt, terminal_backend="cauchy", terminal_fallback="raise",
                 convolution_backend="auto"):
        if terminal_backend not in ("cauchy", "levinson", "dense"):
            raise ValueError("terminal_backend must be 'cauchy', 'levinson', or 'dense'")
        if terminal_fallback not in ("raise", "dense"):
            raise ValueError("terminal_fallback must be 'raise' or 'dense'")
        self.sym, self.cnt, self.F = sym, cnt, sym.F
        self.n, self.w, self.N = sym.n, sym.w, sym.n**2
        self.terminal_backend, self.terminal_fallback = terminal_backend, terminal_fallback
        self.convolution_backend = convolution_backend
        self.prepared, self.report, self.C, self.terminal = False, None, None, None
        self.phase_seconds = {}

    def prepare(self):
        if self.prepared:
            return self.report
        n, w, F, cnt = self.n, self.w, self.F, self.cnt
        L = n+w
        orders = (next_pow2(2*L), next_pow2(3*n-2))
        with cnt.stage("setup"):
            self.backend = ConvolutionBackend(F, cnt, self.convolution_backend, orders)
        start = perf_counter()
        with cnt.stage("leading"):
            self.leading = LeadingInverse(F, self.sym.leading_lag(), n, cnt, backend=self.backend)
        self.phase_seconds["leading"] = perf_counter()-start
        start = perf_counter()
        stats = {}
        self.C = build_series(self.sym, cnt, self.leading, self.backend, stats=stats)
        self.phase_seconds["series"] = perf_counter()-start
        lags = {k: self.C[n+k] for k in range(1-w, w)}
        start = perf_counter()
        with cnt.stage("terminal"):
            if self.terminal_backend == "levinson":
                self.terminal = BlockLevinson(F, lags, w, self.terminal_fallback)
            elif self.terminal_backend == "cauchy":
                from .block_cauchy import BlockCauchy
                self.terminal = BlockCauchy(F, lags, w, self.backend,
                                          fallback=self.terminal_fallback)
            else:
                self.terminal = LU(F, materialize(lags, w))
        self.phase_seconds["terminal_factor"] = perf_counter()-start
        terminal_report = (self.terminal.summary() if hasattr(self.terminal, "summary") else
                           {"kind": "dense_oracle", "terminal_rate": "O(w^3 n^3)",
                            "published_rate_achieved": False,
                            "full_terminal_materialized": True,
                            "stored_scalar_cells": (w*n)**2})
        self.report = {"theorem": "algorithm identities (A)-(C)", "n": n, "w": w,
                       "N": self.N, "L": L, "field": F.name(), "convention": CONVENTION,
                       "series_coefficients": L, "series_responses": n,
                       "series_stored_scalar_cells": L*n*n,
                       "series_construction": stats,
                       "convolution_backend": self.backend.summary(),
                       "terminal_backend": terminal_report,
                       "target_rate_achieved_on_supported_chart": bool(self.backend.achieved and
                            terminal_report["published_rate_achieved"]),
                       "rate_scope": "uniform implementation on the reported roots and pivot chart; "
                                     "not an all-input implementation of the algorithm computation trees",
                       "H0_leading_nonsingular": True,
                       "terminal_nonsingular": not self.terminal.singular,
                       "terminal_nullity": self.terminal.nullity(),
                       "phase_seconds": dict(self.phase_seconds),
                       "ops": cnt.report(n=n, w=w)}
        self.prepared = True
        return self.report

    def _require(self, solve=False):
        if not self.prepared:
            raise NotPrepared("call prepare() first")
        if solve and self.terminal.singular:
            raise SingularTerminalMatrix(order=self.w*self.n, nullity=self.terminal.nullity())

    def solve(self, b):
        self._require(solve=True)
        if len(b) != self.N:
            raise ValueError("right-hand side of length %d expected" % self.N)
        F, n, w = self.F, self.n, self.w
        start = perf_counter()
        with self.cnt.stage("reconstruct"):

            r = apply_window(F, self.C, b, n, w, n-w, self.backend)
            pb = apply_window(F, self.C, b, n, n, -w, self.backend)
        self.phase_seconds["rhs_readouts"] = perf_counter()-start
        start = perf_counter()
        with self.cnt.stage("terminal"):
            y = self.terminal.solve(r)
        self.phase_seconds["terminal_solve"] = perf_counter()-start
        start = perf_counter()
        with self.cnt.stage("reconstruct"):
            qy = apply_window(F, self.C, y, w, n, 0, self.backend)
            x = [F.sub(a, b) for a, b in zip(pb, qy)]
        self.phase_seconds["reconstruct"] = perf_counter()-start
        return x

    def solve_transpose(self, b):

        return list(reversed(self.solve(list(reversed(b)))))

    def det(self):
        self._require()
        with self.cnt.stage("det"):
            d = self.F.mul(self.F.pow(self.leading.det, self.n+self.w), self.terminal.det)
            return self.F.neg(d) if (self.w*self.n**3) % 2 else d

    def mark(self, b, u, eps):
        """Nonsingular rank-one mark via the determinant lemma.

        Singular inputs raise SingularTerminalMatrix. The polynomial identity
        algorithm (A4) holds there, but this API does not implement its singular
        marked-terminal determinant backend.
        """
        self._require(solve=True)
        if len(u) != self.N:
            raise ValueError("wrong mark-vector length")
        x = self.solve(b)
        with self.cnt.stage("mark"):
            t = self.F.zero()
            for ui, xi in zip(u, x):
                t = self.F.add(t, self.F.mul(ui, xi))
            return self.F.mul(self.det(), self.F.add(self.F.one(), self.F.mul(eps, t))), t

    def terminal_matrix(self):
        self._require()
        return materialize({k: self.C[self.n+k] for k in range(1-self.w, self.w)}, self.w)

    def terminal_nullity(self):
        self._require()
        return self.terminal.nullity()

    def counts(self):
        self._require()
        return {"ops": self.cnt.report(n=self.n, w=self.w),
                "phase_seconds": dict(self.phase_seconds),
                "terminal_backend": self.report["terminal_backend"],
                "target_rate_achieved_on_supported_chart":
                    self.report["target_rate_achieved_on_supported_chart"]}

    stagewise_counts = counts
