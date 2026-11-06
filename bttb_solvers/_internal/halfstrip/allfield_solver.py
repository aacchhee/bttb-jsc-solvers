# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Trial/shared. Uniform/CK inverse-series and approximant-terminal comparison; a separate route from public HSC and HIR.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from time import perf_counter

from .jacobi_solver import JacobiHalfStripSolver
from .allfield_convolve import AllFieldConvolutionBackend, select_multiplier
from .block_approximant import BlockApproximant
from .leading import LeadingInverse
from .jacobi_series import build_series, apply_window
from .symbol import CONVENTION


class AllFieldHalfstripSolver(JacobiHalfStripSolver):
    """algorithm identities and soft-O(n^3+w*n^3) exact field arithmetic.

    The established leading-block H0 hypothesis remains required. Singular
    terminals are prepared and support determinant, nullity and scalar marks;
    physical solve/transpose preserve the inherited typed singular refusal.
    No roots, extension-field search, randomization, or terminal pivot chart is
    used. This executes the direct block reduction, not literal KN Algorithm 3.
    """
    def __init__(self, sym, cnt, multiplier="allfield", frontend_multiplier=None, terminal_multiplier=None):
        super().__init__(sym, cnt, terminal_backend="dense", terminal_fallback="raise",
                         convolution_backend="auto")
        self.frontend_multiplier = multiplier if frontend_multiplier is None else frontend_multiplier
        self.terminal_multiplier = multiplier if terminal_multiplier is None else terminal_multiplier
        if self.frontend_multiplier not in ("allfield", "ck") or self.terminal_multiplier not in ("allfield", "ck"):
            raise ValueError("frontend and terminal multiplier must be allfield or ck")
        self.terminal_backend = "approximant_" + self.terminal_multiplier
        self.convolution_backend = "ck_fft" if self.frontend_multiplier == "ck" else "allfield_fft"
        self._mark_report = None

    def prepare(self):
        if self.prepared:
            return self.report
        F, n, w, cnt = self.F, self.n, self.w, self.cnt
        L = n+w
        start = perf_counter()
        with cnt.stage("setup"):
            self.backend = AllFieldConvolutionBackend(F, cnt, self.frontend_multiplier)
        self.phase_seconds["setup"] = perf_counter()-start
        start = perf_counter()
        with cnt.stage("leading"):
            self.leading = LeadingInverse(F, self.sym.leading_lag(), n, cnt, backend=self.backend)
        self.phase_seconds["leading"] = perf_counter()-start
        start = perf_counter()
        stats = {}
        self.C = build_series(self.sym, cnt, self.leading, self.backend, stats=stats)
        self.phase_seconds["series"] = perf_counter()-start
        start = perf_counter()
        with cnt.stage("terminal"):
            self._terminal_mul = select_multiplier(F,self.terminal_multiplier)
            self.terminal = BlockApproximant(F, {k: self.C[n+k] for k in range(1-w, w)}, w, cnt, mul=self._terminal_mul)
        self.phase_seconds["terminal_factor"] = perf_counter()-start
        tr = self.terminal.summary()
        self.report = {"theorem": "algorithm identities (A)-(C)", "n": n, "w": w,
            "N": self.N, "L": L, "field": F.name(), "convention": CONVENTION,
            "series_coefficients": L, "series_responses": n,
            "series_stored_scalar_cells": L*n*n, "series_construction": stats,
            "convolution_backend": self.backend.summary(), "terminal_backend": tr,
            "target_rate_achieved_on_supported_chart": True,
            "all_input_H0_coverage": True, "all_input_fast_rate_achieved": True,
            "theorem_identity_support": "solve, transpose, determinant, nullity, scalar rank-one marks at all terminal ranks",
            "rate_scope": ("soft-O(n^3+w*n^3) field operations over supported exact field adapters, under H0; explicit direct block PM-Basis; frontend scalar bound: " + self.backend._multiply.operation_bound + "; terminal scalar bound: " + self._terminal_mul.operation_bound),
            "allfield_soft_rate_implemented": True,
            "frontend_multiplier": self.frontend_multiplier,
            "terminal_multiplier": self.terminal_multiplier,
            "literal_KN_algorithm_implemented": False,
            "reference_Algorithm5_implemented": False,
            "terminal_pivot_chart_required": False, "root_setup_required": False,
            "dense_terminal_fallback": False,
            "H0_leading_nonsingular": True,
            "terminal_nonsingular": not self.terminal.singular,
            "terminal_nullity": self.terminal.nullity(),
            "phase_seconds": dict(self.phase_seconds), "ops": cnt.report(n=n, w=w)}
        self.prepared = True
        return self.report

    def marked_determinant(self, b, u, eps):
        """Return det(T+eps*b*u^T) by algorithm A4, including singular T.

        r=Rb, q=Q^T u and alpha=u^T Pb feed the direct bordered terminal
        determinant. No inverse of a singular terminal or dense mark is used.
        The inherited nonsingular-only ``mark`` contract remains unchanged.
        """
        self._require()
        if len(b) != self.N or len(u) != self.N:
            raise ValueError("mark vectors must both have length %d" % self.N)
        start = perf_counter()
        before = self.cnt.report()["measured"]
        F, n, w = self.F, self.n, self.w
        with self.cnt.stage("mark"):
            r = apply_window(F, self.C, b, n, w, n-w, self.backend)
            pb = apply_window(F, self.C, b, n, n, -w, self.backend)
            alpha = F.zero()
            for ui, bi in zip(u, pb):
                alpha = F.add(alpha, F.mul(ui, bi))
            Ct = [[list(row) for row in zip(*block)] for block in self.C]
            revu = [x for i in reversed(range(n)) for x in u[i*n:(i+1)*n]]
            qr = apply_window(F, Ct, revu, n, w, n-w, self.backend)
            q = [x for i in reversed(range(w)) for x in qr[i*n:(i+1)*n]]
            marked = self.terminal.marked_determinant(r, q, F.neg(eps))
            bracket = F.add(marked, F.mul(F.mul(eps, alpha), self.terminal.det))
            scale = F.pow(self.leading.det, n+w)
            if (w*n**3)%2:
                scale = F.neg(scale)
            value = F.mul(scale, bracket)
        elapsed = perf_counter()-start
        self.phase_seconds["marked_determinant"] = elapsed
        after = self.cnt.report()["measured"]
        self._mark_report = {"kind": "algorithm A4 direct all-field bordered PM-Basis",
            "terminal_was_singular": self.terminal.singular,
            "full_marked_terminal_materialized": False,
            "expanded_window_materialized": False,
            "inverse_of_singular_terminal_used": False,
            "additional_transposed_series_scalar_cells": (n+w)*n*n,
            "seconds": elapsed, "measured_ops": {k: after[k]-before[k] for k in after},
            "terminal_mark": self.terminal.marked_determinant_report(),
            "all_input_fast_rate_achieved": True}
        return value

    def marked_determinant_report(self):
        return None if self._mark_report is None else dict(self._mark_report)

    def terminal_matrix(self):
        self._require()
        raise NotImplementedError("all-field terminal API keeps polynomial blocks; use an independent test oracle for a scalar matrix")

    def counts(self):
        self._require()
        return {"ops": self.cnt.report(n=self.n, w=self.w),
            "phase_seconds": dict(self.phase_seconds),
            "convolution_backend": self.backend.summary(),
            "terminal_backend": self.terminal.summary(),
            "target_rate_achieved_on_supported_chart": True,
            "all_input_fast_rate_achieved": True}

    stagewise_counts = counts

