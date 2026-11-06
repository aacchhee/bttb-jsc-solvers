# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from time import perf_counter
from copy import deepcopy

from .approximant import pm_basis
from .allfield_poly import make_multiplier


def _zeros(F, m, n):
    return [[F.zero() for _ in range(n)] for _ in range(m)]


def _branch_summary(result, target=None):
    """Observe retained integer metadata; never perform field arithmetic.

    Traversal/formatting time belongs to diagnostic reporting, outside the
    measured field-operation ledger. Field pivot values and matrices are omitted.
    """
    out = {"nodes":0, "leaves":0, "max_depth":0, "order_histogram":{},
           "leaf_rank_histogram":{}, "leaf_pivot_columns_histogram":{},
           "pivot_degree_histogram":{}, "onto_true_nodes":0, "onto_false_nodes":0,
           "observed_profile":list(result["pivot_degree"]),
           "shifted_row_degrees":list(result["row_degree"]),
           "metadata_only":True, "field_operations":0}
    def bump(key, value):
        value = str(value)
        out[key][value] = out[key].get(value,0)+1
    def walk(node, depth):
        out["nodes"] += 1
        out["max_depth"] = max(out["max_depth"], depth)
        bump("order_histogram",node["order"])
        bump("pivot_degree_histogram",tuple(node["profile"]))
        out["onto_true_nodes" if node["all_onto"] else "onto_false_nodes"] += 1
        if node["order"] == 1:
            out["leaves"] += 1
            bump("leaf_rank_histogram",len(node["row_rank_profile"]))
            bump("leaf_pivot_columns_histogram",tuple(node["pivot_columns"]))
        else:
            walk(node["left"],depth+1)
            walk(node["right"],depth+1)
    walk(result["ledger"],0)
    if target is not None:
        out["expected_profile"] = list(target)
        out["profile_guard"] = "accepted" if result["pivot_degree"] == target else "rejected"
    return out


class BlockApproximant:
    """LU-compatible determinant/rank/solve terminal, including all ranks.

    Polynomial coefficients are degree first. The multiplier must compute
    exact full scalar products in the same field. Default multiplication is
    the uniform synthetic cyclotomic-ring FFT; a supplied multiplier retains
    its own rate and is not silently labeled as the default fast primitive.
    """
    def __init__(self, F, lags, width, cnt=None, mul=None):
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("positive integer terminal width required")
        if set(lags) != set(range(1-width, width)):
            raise ValueError("all 2w-1 lag blocks are required")
        n = len(lags[0])
        if not n or any(len(B) != n or any(len(row) != n for row in B) for B in lags.values()):
            raise ValueError("lag blocks must be nonempty square matrices")
        self.F, self.cnt, self.width, self.block_size = F, cnt, width, n
        self.n, self.order = width*n, 2*width-1
        self.lags = {k: [list(row) for row in B] for k, B in lags.items()}
        self.default_multiplier = mul is None
        self.mul = make_multiplier(F) if mul is None else mul
        if not callable(self.mul):
            raise ValueError("scalar polynomial multiplier required")
        self.multiplier_kind = getattr(self.mul, "primitive_kind", "caller supplied")
        self.recognized_multiplier = self.multiplier_kind in ("allfield", "ck")
        self.calls = {"determinant": 0, "bounded_solve": 0, "marked_determinant": 0}
        self.seconds = {}
        self._last_mark_report = None
        self._last_query_branches = None
        start = perf_counter()
        constraints = self._plain_constraints()
        target = [width]*n + [width-1]*n
        self.factor = pm_basis(F, constraints, self.order, [-x for x in target], self.mul)
        self.calls["determinant"] += 1
        self.seconds["prepare"] = perf_counter()-start
        if not self.factor["ledger"]["all_onto"]:
            raise ArithmeticError("plain terminal constraint module unexpectedly not onto")
        self._prepare_branches = _branch_summary(self.factor,target)
        self._nullity = sum(max(0, -x) for x in self.factor["row_degree"])
        self.rank, self.singular = self.n-self._nullity, self._nullity != 0
        self.det = self._target_determinant(self.factor, target)
        if F.is_zero(self.det) != self.singular:
            raise ArithmeticError("terminal profile determinant and nullity disagree")

    def _plain_constraints(self, extra=0, extra_constraints=0):
        F, n, w = self.F, self.block_size, self.width
        out = [_zeros(F, 2*n+extra, n+extra_constraints) for _ in range(self.order)]
        for k in range(self.order):
            B = self.lags[k-w+1]
            for i in range(n):
                for j in range(n):
                    out[k][i][j] = B[j][i]  
        for i in range(n):
            out[0][n+i][i] = F.neg(F.one())
        return out

    def _target_determinant(self, result, target):
        if not result["ledger"]["all_onto"]:
            raise ArithmeticError("direct determinant constraint module unexpectedly not onto")
        if result["pivot_degree"] != target:
            return self.F.zero()
        value = result["ledger"]["window_determinant"]
        return self.F.neg(value) if self.block_size*(self.width-1)%2 else value

    def nullity(self):
        return self._nullity

    def _vector(self, v):
        if len(v) != self.n:
            raise ValueError("terminal vector of length %d expected" % self.n)

    def bounded_solve(self, b):
        """Solve all ranks; return particular and a compressed kernel module.

        ``particular`` is a flat block-major vector or None for inconsistency.
        ``kernel_rows[k][chain][spatial]`` stores coefficient k of each chain;
        each chain's polynomial multiplier has degree < ``kernel_limits``.
        Chain limits are retained from the original negative shifted rows.
        Writing an expanded nullspace basis is deliberately not part of this API.
        """
        self._vector(b)
        F, n, w = self.F, self.block_size, self.width
        start = perf_counter()
        constraints = self._plain_constraints(extra=1)
        for i in range(w):
            for a in range(n):
                constraints[w-1+i][2*n][a] = F.neg(b[i*n+a])
        shift = [-w]*n + [-(w-1)]*n + [-1]
        result = pm_basis(F, constraints, self.order, shift, self.mul)
        self._last_query_branches = _branch_summary(result)
        self._last_query_branches["query"] = "bounded_solve"
        B = result["basis"]
        negative = [i for i, d in enumerate(result["row_degree"]) if d < 0]
        limits = [-result["row_degree"][i] for i in negative]
        affine = next((i for i in negative if not F.is_zero(B[0][i][2*n])), None)
        dimension = sum(limits)
        particular = None
        chains, kept_limits = [], []
        inverse = None
        if affine is not None:
            if result["row_degree"][affine] != -1:
                raise ArithmeticError("nonconstant affine bounded approximant coordinate")
            inverse = F.inv(B[0][affine][2*n])  
            particular = [F.mul(inverse, B[k][affine][a]) if k < len(B) else F.zero()
                          for k in range(w) for a in range(n)]
        for i, limit in zip(negative, limits):
            if i == affine:
                continue
            factor = F.zero() if affine is None else F.mul(B[0][i][2*n], inverse)
            chain = []
            for k in range(w):
                row = []
                for a in range(n):
                    v = B[k][i][a] if k < len(B) else F.zero()
                    if affine is not None:
                        p = B[k][affine][a] if k < len(B) else F.zero()
                        v = F.sub(v, F.mul(factor, p))
                    row.append(v)
                chain.append(row)
            chains.append(chain); kept_limits.append(limit)
        kernel_rows = [[chains[j][k] for j in range(len(chains))] for k in range(w)]
        nullity = dimension-(1 if affine is not None else 0)
        if nullity != self._nullity:
            raise ArithmeticError("affine and homogeneous terminal nullities disagree")
        self.calls["bounded_solve"] += 1
        elapsed = perf_counter()-start
        self.seconds["last_bounded_solve"] = elapsed
        return {"particular": particular, "consistent": affine is not None,
                "nullity": nullity, "kernel_rows": kernel_rows,
                "kernel_limits": kept_limits,
                "diagnostics": {"seconds": elapsed, "components": 2*n+1,
                    "constraints": n, "order": self.order,
                    "negative_row_indices": negative, "negative_row_limits": limits,
                    "affine_row": affine, "full_terminal_materialized": False,
                    "expanded_kernel_materialized": False,
                    "basis_stored_scalar_cells": len(B)*(2*n+1)**2}}

    def solve(self, b):
        self._vector(b)
        if self.singular:
            return None
        result = self.bounded_solve(b)
        if result["particular"] is None:
            raise ArithmeticError("nonsingular terminal has inconsistent affine module")
        return result["particular"]

    def solve_transpose(self, b):
        self._vector(b)
        if self.singular:
            return None
        transposed = {k: [list(row) for row in zip(*self.lags[-k])]
                      for k in self.lags}
        terminal = BlockApproximant(self.F, transposed, self.width, self.cnt, self.mul)
        result = terminal.solve(b)
        self._last_query_branches = deepcopy(terminal._last_query_branches)
        if self._last_query_branches is not None:
            self._last_query_branches["query"] = "solve_transpose"
        return result

    def marked_determinant(self, r, q, eps=None):
        """Return det(H+eps*r*q^T), using a bordered effective window at all ranks."""
        self._vector(r); self._vector(q)
        F, n, w = self.F, self.block_size, self.width
        eps = F.one() if eps is None else eps
        start = perf_counter()
        constraints = self._plain_constraints(extra=2, extra_constraints=1)
        for i in range(w):
            for a in range(n):
                constraints[w-1+i][2*n][a] = F.mul(eps, r[i*n+a])
                constraints[2*w-2-i][a][n] = F.neg(q[i*n+a])
        constraints[2*w-2][2*n][n] = F.one()
        constraints[w-1][2*n+1][n] = F.neg(F.one())
        target = [w]*n + [w-1]*n + [1, w-1]
        result = pm_basis(F, constraints, self.order, [-x for x in target], self.mul,
                          offsets=[0]*n+[w-1])
        self._last_query_branches = _branch_summary(result,target)
        self._last_query_branches["query"] = "marked_determinant"
        value = self._target_determinant(result, target)
        self.calls["marked_determinant"] += 1
        elapsed = perf_counter()-start
        self.seconds["last_marked_determinant"] = elapsed
        self._last_mark_report = {"seconds": elapsed, "components": 2*n+2,
            "constraints": n+1, "order": self.order, "offsets": [0]*n+[w-1],
            "profile_matches_target": result["pivot_degree"] == target,
            "observed_profile": result["pivot_degree"], "expected_profile": target,
            "all_onto": result["ledger"]["all_onto"],
            "branches": self._last_query_branches,
            "full_terminal_materialized": False, "expanded_window_materialized": False,
            "inverse_of_base_terminal_used": False,
            "basis_stored_scalar_cells": len(result["basis"])*(2*n+2)**2}
        return value

    def marked_determinant_report(self):
        return None if self._last_mark_report is None else dict(self._last_mark_report)

    def summary(self):
        n, w = self.block_size, self.width
        stats = getattr(self.mul, "stats", None)
        basis_cells = len(self.factor["basis"])*(2*n)**2
        lag_cells = self.order*n*n

        
        trace_cells_bound = self.order*(2*n*n+n) + 2*(2*self.order-1)
        return {"kind": "block_approximant_" + (self.multiplier_kind if self.recognized_multiplier else "custom"), "terminal_rate": "O(n^3 M_K(w) log(w+1))",
                "published_rate_achieved": self.recognized_multiplier,
                "normal_chart_rate_achieved": self.recognized_multiplier,
                "all_input_fast_rate_achieved": self.recognized_multiplier,
                "scope": "all ranks over supported exact field adapters; direct block row PM-Basis",
                "chart": None, "root_orders_required": [], "fallback_reason": None,
                "full_terminal_materialized": False, "expanded_window_materialized": False,
                "stored_scalar_cells": basis_cells+lag_cells+trace_cells_bound,
                "stored_scalar_cells_scope": "upper bound for retained basis, lag blocks and ledger field scalars",
                "basis_stored_scalar_cells": basis_cells,
                "retained_ledger_scalar_cells_bound": trace_cells_bound,
                "polynomial_order": self.order, "components": 2*n, "constraints": n,
                "terminal_nullity": self._nullity, "terminal_rank": self.rank,
                "observed_profile": list(self.factor["pivot_degree"]),
                "determinant_ledger": "algorithm Lemma M with direct block-window profile and sign",
                "basis_degree_bound": self.order, "calls": dict(self.calls),
                "seconds": dict(self.seconds),
                "scalar_multiplier_kind": self.multiplier_kind,
                "scalar_multiplication_bound": getattr(self.mul,"operation_bound","caller supplied; rate unspecified"),
                "scalar_multiplier_stats_scope": "cumulative callable telemetry; may include shared frontend products",
                "scalar_multiplier": deepcopy(stats) if stats is not None else {"kind": "caller supplied; rate unspecified"},
                "branches": deepcopy({"prepare":self._prepare_branches, "last_query":self._last_query_branches}),
                "diagnostic_observer": "retained integer metadata traversal; no field operations; included in wall time",
                "literal_KN_algorithm_implemented": False,
                "structured_zero_skipping": False}
