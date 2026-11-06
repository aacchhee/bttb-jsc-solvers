# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured half-strip comparison.

Role: Trial. Read source coefficients, form the terminal problem and recover a physical solution in the structured half-strip comparison.

Entry points: FullSource; solve_prepared.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.uniform_terminal import matrix_product
from bttb_solvers._internal.halfstrip_core.mosaic import run_mosaic, solve_mosaic
from bttb_solvers._internal.halfstrip_core.exact import power
from bttb_solvers._internal.joint_arrays import SeriesOps, trim_factor
from bttb_solvers._internal.joint_blocks import materialize_physical
from bttb_solvers._internal.joint_compiler import compile_joint


class FullSource:
    def __init__(self, initial, compiled=None):
        self.initial = initial
        self.compiled = compile_joint(initial) if compiled is None else compiled
        if self.compiled.initial is not initial or self.compiled.overcap:
            raise ValueError("matching successful compact compiler prefix required")
        self.setup, self.F = initial.setup, initial.F
        self.n, self.w, self.L, self.h, self.q = initial.n, initial.w, initial.L, initial.h, initial.q
        self.delta = tuple(self.compiled.delta)
        if len(self.delta) != self.q or sum(self.delta) > self.h:
            raise ArithmeticError("compiled channel/cap mismatch")
        self.ops, self.precision = SeriesOps(self.F), 2*self.L
        self.stats = {"selected_matrix_products": {}, "selected_W_count": 0,
                      "residual_physical_cells": 0, "term_count": 0}
        self.positive_channels = tuple(i for i, delay in enumerate(self.delta) if delay > 0)
        self.chain_lengths = [self.w]*self.n + [self.delta[i] for i in self.positive_channels]
        self.selected_W = self._acquire_selected_W()

    def _shape(self, vector, rows, P):
        if type(P) is not int or not 1 <= P <= self.precision:
            raise ValueError("requested source-action precision unavailable")
        if len(vector) != rows or any(len(poly) != P for poly in vector):
            raise ValueError("source-action vector shape mismatch")

    def G_action(self, vector, P):
        """G y=U_s V^t R y, same analytic histories through precision2L."""
        self._shape(vector, self.n, P)
        value = self.setup.R.apply(vector, P)
        moments = self.initial.moment_transform.evaluate(value, P)
        return self.compiled.histories.left(moments, P)

    def B_action(self, vector, P):
        """B eta=R J V V_s eta; weighted moments avoid an n*q*P loop."""
        self._shape(vector, self.q, P)
        value = self.compiled.histories.right(vector, P)
        physical = self.initial.moment_transform.moments(value, P, self.n)
        return self.setup.R.apply(list(reversed(physical)), P)

    def W_action(self, vector):
        """Exact W y on [-L,L), for a polynomial y of degree<L.

        Return n rows of 2L coefficients with explicit exponent offset -L.
        Auxiliary channels are formed through2L, retaining the lift/shear tail.
        """
        F, L, P = self.F, self.L, self.precision
        if len(vector) != self.n or any(len(poly) != L for poly in vector):
            raise ValueError("W input must be n polynomial rows of lengthL")
        padded = [list(poly)+self.ops.zero(L) for poly in vector]
        Gvalue = self.G_action(padded, P)
        eta = [self.ops.delay(poly, L-delay, P) for poly, delay in zip(Gvalue, self.delta)]
        Bvalue, Rvalue = self.B_action(eta, P), self.setup.R.apply(padded, P)
        out = []
        for i in range(self.n):
            row = list(Bvalue[i])
            for degree in range(L): row[L+degree] = F.add(row[L+degree], Rvalue[i][degree])
            out.append(row)
        return out

    def _residual_terms(self):
        state, ids, P = self.compiled.state, self.compiled.residual_ids, self.precision
        F, n, r = self.F, self.n, state.width
        if not ids:
            return [], None
        if len(ids)*self.L > self.h:
            raise ArithmeticError("residual acquisition precedes a valid cap")
        C = materialize_physical(state.upper(ids, P), P)
        Dt = materialize_physical(state.lower(ids, P).upper, P)

        
        self.stats["residual_physical_cells"] = 2*n*len(ids)*P
        positions = {i: a for a, i in enumerate(state.active_ids)}
        g, j = [state.g[positions[i]] for i in ids], [state.j[positions[i]] for i in ids]
        left, right = [[self.ops.zero(P) for _ in range(r)] for _ in range(n)], [[self.ops.zero(P) for _ in range(r)] for _ in range(n)]
        row = [self.ops.zero(P) for _ in range(n)]
        for i in range(n):
            for channel in range(len(ids)):
                row[i] = self.ops.add(row[i], self.ops.mul(C[0][channel], Dt[i][channel], P))
                for u in range(r):
                    left[i][u] = self.ops.add(left[i][u], self.ops.mul(C[i][channel], g[channel][u][:P], P))
                    right[i][u] = self.ops.add(right[i][u], self.ops.mul(j[channel][u][:P], Dt[i][channel], P))
        E, Fs = trim_factor(state.E[:n], P), trim_factor(state.Fscaled[:n], P)
        del C, Dt
        return [(E, right, -self.L), (left, Fs, -self.L)], row

    def _coefficient_product(self, left, right, shift, coefficient):
        F, n, P = self.F, self.n, self.precision
        r = len(left[0]) if left else 0
        target = coefficient-shift
        if not 0 <= target < P:
            raise ArithmeticError("selected displacement coefficient lacks precision")
        A = [[left[i][u][t] for u in range(r) for t in range(P)] for i in range(n)]
        B = []
        for u in range(r):
            for t in range(P):
                index = target-t
                B.append([right[j][u][index] if index >= 0 else F.zero() for j in range(n)])
        return matrix_product(F, A, B, stats=self.stats["selected_matrix_products"])

    def _acquire_selected_W(self):
        if self.q == 0:

            
            selected = self.setup.R.selected_W(self.w)
            self.stats["selected_W_count"] = len(selected)
            return selected
        F, n, P = self.F, self.n, self.precision
        state = self.compiled.state
        if state.precision < P:
            raise ArithmeticError("final source precision is below2L")
        terms = [(trim_factor(state.E[:n], P), trim_factor(state.Fscaled[:n], P), -state.common_depth)]
        residual, residual_row = self._residual_terms()
        terms.extend(residual)
        self.stats["term_count"] = len(terms)
        selected = {}
        for c in range(n-self.w+1, self.L):
            delta = [[F.zero() for _ in range(n)] for _ in range(n)]
            for left, right, shift in terms:
                product = self._coefficient_product(left, right, shift, c)
                delta = [[F.add(x, y) for x, y in zip(a, b)] for a, b in zip(delta, product)]
            X = [[F.zero() for _ in range(n)] for _ in range(n)]
            X[0] = [state.X_first_row[j][c+state.common_depth] for j in range(n)]
            if residual_row is not None:
                X[0] = [F.add(x, residual_row[j][c+self.L]) for j, x in enumerate(X[0])]
            for i in range(1, n):
                for j in range(n):
                    X[i][j] = F.add(X[i-1][j+1] if j+1 < n else F.zero(), delta[i-1][j])
            selected[c] = [list(reversed(row)) for row in X]
        self.stats["selected_W_count"] = len(selected)
        return selected

    def coefficient(self, i, j, lag):
        lengths, n, F = self.chain_lengths, self.n, self.F
        if not 0 <= i < len(lengths) or not 0 <= j < len(lengths):
            raise ValueError("terminal chain index outside actual chain list")
        if not 1-lengths[j] <= lag < lengths[i]:
            raise ValueError("terminal lag outside actual coefficient window")
        if i < n and j < n: return self.selected_W[n+lag][i][j]
        if i < n:
            channel = self.positive_channels[j-n]
            index = self.delta[channel]-self.w+lag
            return self.compiled.B_low[i][channel][index] if index >= 0 else F.zero()
        if j < n:
            channel = self.positive_channels[i-n]
            return F.neg(self.compiled.G_low[channel][j][lag]) if lag >= 0 else F.zero()
        return F.zero()

    def determinant_scale(self):
        F = self.F
        unit = F.mul(self.setup.detH0, F.inv(F.mul(self.compiled.dU, self.compiled.dV)))
        value = power(F, unit, self.L)
        parity = self.n*self.n*self.h + sum(delay*(self.L-delay) for delay in self.delta)
        return F.neg(value) if parity % 2 else value

    def recover(self, backend, rhs):
        """algorithm (XV)–(XVI): one arbitrary physical time-major RHS."""
        F, n, w, L, P = self.F, self.n, self.w, self.L, self.precision
        if len(rhs) != n*n: raise ValueError("physical RHS requires n^2 native entries")
        f = [self.ops.zero(L) for _ in range(n)]
        for time in range(n):
            for i in range(n): f[i][time+w] = rhs[time*n+i]
        Wf = self.W_action(f)
        Gf = self.G_action([poly+self.ops.zero(L) for poly in f], P)
        terminal_rhs = [F.neg(Wf[i][L+n+t]) for i in range(n) for t in range(w)]
        terminal_rhs += [Gf[channel][degree] for channel in self.positive_channels for degree in range(self.delta[channel])]
        affine = solve_mosaic(F, backend, terminal_rhs)
        solution = affine["solution"]
        if solution is None:
            raise ArithmeticError("nonsingular terminal lacks affine readout")
        y = [list(poly) for poly in f]
        for i in range(n):
            for t in range(w): y[i][t] = F.add(y[i][t], solution[i*w+t])
        kernels = [self.ops.zero(L) for _ in range(self.q)]
        at = n*w
        for channel in self.positive_channels:
            delay = self.delta[channel]
            kernels[channel][L-delay:L] = solution[at:at+delay]
            at += delay
        Wy, BK = self.W_action(y), self.B_action(kernels, L)
        answer = [list(poly) for poly in Wy]
        for i in range(n):
            for degree in range(L): answer[i][L+degree] = F.add(answer[i][L+degree], BK[i][degree])
        if any(not F.is_zero(value) for poly in answer for value in poly[:L]):
            raise ArithmeticError("physical reconstruction has negative support")
        if any(not F.is_zero(answer[i][L+degree]) for i in range(n) for degree in range(n, L)):
            raise ArithmeticError("physical reconstruction has nonzero tail")
        return {"physical_solution": [answer[i][L+time] for time in range(n) for i in range(n)],
                "terminal_rhs": terminal_rhs, "terminal_solution": solution}


def solve_prepared(initial, rhs=None):
    """One guarded attempt: all-rank det and one RHS when nonsingular.

    Does not catch Restart or descend. The outer owner must discard this entire
    attempt, including source/history/terminal/cache state, after any restriction.
    """
    F = initial.F
    if rhs is not None and len(rhs) != initial.n*initial.n:
        raise ValueError("physical RHS requires n^2 native entries")
    compiled = compile_joint(initial)
    if compiled.overcap:
        return {"det": F.zero(), "singular": True, "physical_solution": None,
                "q": initial.q, "overcap": True, "compiler": compiled,
                "source": None, "terminal": None}
    source = FullSource(initial, compiled)
    backend = run_mosaic(F, source.chain_lengths, source.coefficient)
    gamma = source.determinant_scale()
    det = F.mul(gamma, backend["det"])
    singular = F.is_zero(det)
    result = {"det": det, "singular": singular, "physical_solution": None,
              "q": initial.q, "delta": source.delta, "k": sum(source.delta),
              "gamma": gamma, "overcap": False, "source": source, "terminal": backend}
    if rhs is not None and not singular: result.update(source.recover(backend, rhs))
    return result
