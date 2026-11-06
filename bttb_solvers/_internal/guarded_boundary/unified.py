# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""GBE boundary comparator.

Role: Main/shared. Validate raw inputs, select width-one/general boundary preparation, retain transpose state and check query residuals.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter

from .fields import PrimeField
from . import operations, banded


class UnifiedError(RuntimeError):
    """A named wrapper outcome; report is a detached accounting snapshot."""

    def __init__(self, reason, message=None):
        self.reason = reason
        self.report = None
        super().__init__(message or reason)


class StructuredUnavailable(UnifiedError):
    """A named capability or checked-chart refusal, not singularity."""


class StructuredSingular(UnifiedError):
    """Source correction singularity violates the nonsingular-input promise."""


class VerificationFailure(UnifiedError):
    """A returned candidate failed the exact original-coordinate guard."""


def _delta(after, before):
    if after is None:
        return None
    before = before or {}
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in after.keys() | before.keys()}


def _accumulate(destination, values):
    if values is not None:
        for key, value in values.items():
            destination[key] = destination.get(key, 0) + value


def _make_structured_adapter(lags, block_count, n, k):
    """Fixed production factory, lazy so rootfree needs no F/L imports.

    The independent engineering controls can replace this module-local factory
    in their own test process; production offers no runtime hook argument.
    """
    raise NotImplementedError("only GBE guarded boundary elimination is included")


class UnifiedPrepared:
    """Eager reusable preparation for B[i*n+a,j*n+b]=a_(i-j)[a-b]."""

    def __init__(self, field, lags, block_count, *, strategy="auto"):
        started = perf_counter()
        self.field = field
        self.strategy = strategy
        self.m = self.n = self.k = None
        self.N = None
        self.lags = ()
        self.selected_backend = None
        self.fallback_reason = None
        self.status = "PREPARING"
        self._backend = self._structured = None
        self._structured_attempted = False
        self._construction_events = []
        self._construction_phases = []
        self._phase_totals = {}
        self._field_operations = {}
        self._partial_source_operations = {}
        self._source_model_diagnostics = {}
        self._partial_kernel_diagnostics = {}
        self._construction_seconds = 0.0
        self._query_seconds = 0.0
        self._query_count = self._successful_queries = 0
        self._prepare_count = 0
        self._last_query = None
        self._copied_block_entries = 0
        original_stage = field.stage
        named_failure = None
        try:
            with self._phase("validation", self._construction_phases):
                self._validate(lags, block_count)
            self._prepare()
            self.status = "READY"
        except UnifiedError as error:
            named_failure = error
            self.status = error.reason
            raise
        finally:
            field.stage = original_stage
            self._construction_seconds = perf_counter() - started
            self._add_overhead(self._construction_phases, self._construction_seconds)
            if named_failure is not None:
                named_failure.report = self.statistics()

    def _source_snapshot(self):
        return None if self._structured is None else self._structured.source_snapshot()

    def _kernel_snapshot(self):
        return None if self._structured is None else self._structured.kernel_snapshot()

    def _model_snapshot(self):
        return None if self._structured is None else self._structured.model_snapshot()

    @contextmanager
    def _phase(self, name, records):
        started = perf_counter()
        before = dict(self.field.counts)
        source_before, kernel_before = self._source_snapshot(), self._kernel_snapshot()
        model_before = self._model_snapshot()
        original_stage = self.field.stage
        self.field.stage = "unified_" + name
        outcome = "ERROR"
        try:
            yield
            outcome = "OK"
        finally:
            self.field.stage = original_stage
            counts = _delta(dict(self.field.counts), before)
            source = _delta(self._source_snapshot(), source_before)
            kernels = _delta(self._kernel_snapshot(), kernel_before)
            model = _delta(self._model_snapshot(), model_before)
            elapsed = perf_counter() - started
            record = {"phase": name, "elapsed_seconds": elapsed, "outcome": outcome,
                      "field_operations": counts,
                      "partial_source_operations": source,
                      "source_model_diagnostics": model,
                      "partial_kernel_diagnostics": kernels}
            records.append(record)
            self._phase_totals[name] = self._phase_totals.get(name, 0.0) + elapsed
            _accumulate(self._field_operations, counts)
            _accumulate(self._partial_source_operations, source)
            _accumulate(self._source_model_diagnostics, model)
            _accumulate(self._partial_kernel_diagnostics, kernels)

    def _add_overhead(self, records, total):
        overhead = max(0.0, total - sum(p["elapsed_seconds"] for p in records))
        records.append({"phase": "wrapper_overhead", "elapsed_seconds": overhead,
                        "outcome": "OK", "field_operations": {},
                        "partial_source_operations": None,
                        "source_model_diagnostics": None,
                        "partial_kernel_diagnostics": None})
        self._phase_totals["wrapper_overhead"] = (
            self._phase_totals.get("wrapper_overhead", 0.0) + overhead)

    def _validate(self, lags, block_count):
        if self.strategy not in ("auto", "rootfree"):
            raise ValueError("unsupported unified strategy")
        if type(block_count) is not int or block_count < 1:
            raise ValueError("positive integer block_count required")
        if isinstance(lags, Mapping):
            keys = set(lags)
            if not keys or any(type(key) is not int for key in keys):
                raise ValueError("lag mapping needs integer keys -k,...,k")
            k = max(keys)
            if k < 1 or keys != set(range(-k, k + 1)):
                raise ValueError("lag mapping needs exactly all keys -k,...,k, k>=1")
            ordered = [lags[j] for j in range(-k, k + 1)]
        else:
            ordered = list(lags)
            if len(ordered) < 3 or len(ordered) % 2 != 1:
                raise ValueError("odd lag sequence length 2*k+1 with k>=1 required")
            k = len(ordered) // 2
        vectors = tuple(tuple(vector) for vector in ordered)
        width = len(vectors[k])
        if width < 1 or width % 2 != 1 or any(len(v) != width for v in vectors):
            raise ValueError("equal positive odd inner lag lengths required")
        self.lags = vectors
        self.m, self.n, self.k = block_count, (width + 1) // 2, k
        self.N = self.m * self.n

    def _preflight(self):
        if type(self.field) is not PrimeField or self.field.p != 12289:
            return "FIELD_UNSUPPORTED"
        if self.m % 12289 == 0:
            return "CHARACTERISTIC_DIVIDES_M"
        if 12288 % self.m != 0:
            return "ROOT_UNAVAILABLE"
        if self.m < 2 * self.k + 1:
            return "BANDWIDTH_OVERLAP"
        if 12288 <= 2 * self.k * self.n * self.m:
            return "FIELD_TOO_SMALL"
        return None

    def _prepare(self):
        if self.strategy in ("auto", "rootfree"):
            self._construction_events.append({
                "event": "selection", "backend": "rootfree",
                "reason": "AUTO_CONSERVATIVE_ROOTFREE" if self.strategy == "auto"
                else "ROOTFREE_REQUESTED"})
            self._prepare_rootfree()
            return
        with self._phase("structured_preflight", self._construction_phases):
            reason = self._preflight()
        self._construction_events.append({"event": "structured_preflight",
                                          "reason": reason, "eligible": reason is None})
        if reason is None:
            with self._phase("structured_prepare", self._construction_phases):

                self._structured_attempted = True
                self._structured = _make_structured_adapter(self.lags, self.m, self.n, self.k)
                status = self._structured.prepare()
            self._construction_events.append({"event": "structured_status", "status": status})
            if status == "OK":
                self.selected_backend = "structured"
                self._backend = self._structured
                self._prepare_count += 1
                return
            if status == "SINGULAR":
                raise StructuredSingular("SINGULAR", "source correction is singular; "
                                         "input is outside the nonsingular promise")
            if status != "CHART_FAILURE":
                raise RuntimeError("unexpected structured status: " + str(status))
            reason = "CHART_FAILURE"
        if self.strategy == "structured":
            raise StructuredUnavailable(reason)
        self.fallback_reason = reason
        self._construction_events.append({"event": "fallback", "reason": reason,
                                          "backend": "rootfree"})
        self._prepare_rootfree()

    def _prepare_rootfree(self):
        with self._phase("block_expansion", self._construction_phases):
            n, k = self.n, self.k
            blocks = {lag: [[self.lags[lag+k][a-b+n-1] for b in range(n)]
                            for a in range(n)] for lag in range(-k, k+1)}
            self._copied_block_entries = (2*k+1)*n*n
        with self._phase("rootfree_prepare", self._construction_phases):
            if self.k == 1:
                self._backend = operations.prepare(self.field, blocks[1], blocks[0],
                                                   blocks[-1], self.m)
                self.selected_backend = "rootfree_operations"
            else:
                self._backend = banded.prepare_banded(self.field, blocks, self.m, self.k)
                self.selected_backend = "guarded_boundary"
            self._prepare_count += 1

    def _check_residual(self, candidate, rhs, transposed):
        if len(candidate) != self.N:
            raise VerificationFailure("OUTPUT_DIMENSION", "backend output dimension mismatch")
        f, n, k = self.field, self.n, self.k
        all_zero = True
        for i in range(self.m):
            for a in range(n):
                value = f.zero
                for j in range(max(0, i-k), min(self.m, i+k+1)):
                    lag = j-i if transposed else i-j
                    vector = self.lags[lag+k]
                    for b in range(n):
                        inner_lag = b-a if transposed else a-b
                        value = f.add(value, f.mul(vector[inner_lag+n-1], candidate[j*n+b]))
                if not f.zero_test(f.sub(value, rhs[i*n+a])):
                    all_zero = False
        if not all_zero:
            raise VerificationFailure("RESIDUAL_NONZERO", "exact physical residual is nonzero")

    def _query(self, rhs, transposed):
        if self.status != "READY":
            raise RuntimeError("unified preparation is not ready")
        started = perf_counter()
        original_stage = self.field.stage
        phases = []
        self._query_count += 1
        last = {"direction": "transpose" if transposed else "forward",
                "backend": self.selected_backend, "prepared_state_reused": True,
                "query_number": self._query_count, "phases": phases,
                "residual_outcome": "NOT_RUN", "status": "ERROR",
                "reversal_copied_entries": 0}
        named_failure = None
        try:
            with self._phase("rhs_validation", phases):
                copied_rhs = list(rhs)
                if len(copied_rhs) != self.N:
                    raise ValueError("physical RHS dimension mismatch")
            if transposed:
                with self._phase("input_reversal", phases):
                    work = list(reversed(copied_rhs))
                    last["reversal_copied_entries"] += self.N
            else:
                work = copied_rhs
            with self._phase("backend_solve", phases):
                candidate = list(self._backend.solve(work))
                if len(candidate) != self.N:
                    raise VerificationFailure("OUTPUT_DIMENSION",
                                              "backend output dimension mismatch")
            if transposed:
                with self._phase("output_reversal", phases):
                    candidate = list(reversed(candidate))
                    last["reversal_copied_entries"] += self.N
            last["residual_outcome"] = "ERROR"
            with self._phase("residual_guard", phases):
                self._check_residual(candidate, copied_rhs, transposed)
            last["residual_outcome"] = "EXACT_ZERO"
            last["status"] = "OK"
            self._successful_queries += 1
            return list(candidate)
        except UnifiedError as error:
            named_failure = error
            last["status"] = error.reason
            raise
        finally:
            self.field.stage = original_stage
            elapsed = perf_counter() - started
            self._query_seconds += elapsed
            self._add_overhead(phases, elapsed)
            last["elapsed_seconds"] = elapsed
            self._last_query = last
            if named_failure is not None:
                named_failure.report = self.statistics()

    def solve(self, rhs):
        return self._query(rhs, False)

    def solve_transpose(self, rhs):
        return self._query(rhs, True)

    def statistics(self):
        rootfree = None
        if self.selected_backend in ("rootfree_operations", "guarded_boundary"):
            rootfree = self._backend.statistics()

            
            rootfree.pop("operations", None)
        arithmetic_total = (None if self._structured_attempted
                            else sum(self._field_operations.values()))
        report = {
            "method": "guarded_boundary.unified", "status": self.status,
            "field": getattr(self.field, "name", type(self.field).__name__),
            "m": self.m, "n": self.n, "k": self.k, "N": self.N,
            "requested_strategy": self.strategy, "selected_backend": self.selected_backend,
            "fallback_reason": self.fallback_reason,
            "construction_events": self._construction_events,
            "construction_phases": self._construction_phases,
            "preparation_count": self._prepare_count,
            "query_count": self._query_count, "successful_queries": self._successful_queries,
            "prepared_state_reuse_count": self._query_count,
            "rhs_result_cache": False, "last_query": self._last_query,
            "wall_seconds": {"construction": self._construction_seconds,
                             "queries": self._query_seconds,
                             "total": self._construction_seconds + self._query_seconds,
                             "phase_totals": self._phase_totals},
            "block_expansion_copied_entries": self._copied_block_entries,
            "field_operations": self._field_operations,
            "complete_field_operation_total": arithmetic_total,
            "complete_arithmetic_coverage": ("unavailable: structured integer/NumPy work uncounted"
                if self._structured_attempted else "rootfree and wrapper exact field protocol calls"),
            "partial_source_operations": (self._partial_source_operations
                                           if self._structured_attempted else None),
            "source_model_diagnostics": (self._source_model_diagnostics
                                          if self._structured_attempted else None),
            "partial_kernel_diagnostics": (self._partial_kernel_diagnostics
                                            if self._structured_attempted else None),
            "partial_counter_notice": "source arithmetic, MODEL/call diagnostics, and kernel calls "
                                      "are separate; never pooled",
            "structured_attempted": self._structured_attempted,
            "structured": (None if self._structured is None else self._structured.diagnostics()),
            "rootfree": rootfree,
            "bound_policy": "auto uses rootfree; explicit structured work has no complete arithmetic claim"}
        return deepcopy(report)


def prepare_unified(field, lags, block_count, *, strategy="auto"):
    """Prepare native exact Toeplitz lags; every returned solve is checked."""
    return UnifiedPrepared(field, lags, block_count, strategy=strategy)
