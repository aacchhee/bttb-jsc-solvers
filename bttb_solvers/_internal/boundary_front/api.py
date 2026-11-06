# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from collections.abc import Mapping
from contextlib import contextmanager
from time import perf_counter, process_time

from bttb_solvers._internal.guarded_boundary.boundary import RankFailure

from .fields import Counted
from .homogeneous import build_types, root_inverse
from .incidence import merge_layout
from .kernel import backend_scope, merge_ledger
from .query import solve_batched, solve_unbatched
from .residual import residual_classical, residual_fast
from .schedule import occurrence_tree
from .selection import SelectionFailure

METHOD = "BFM"


class VerificationFailure(RuntimeError):
    """A returned candidate failed the exact residual check (never expected)."""


def _transpose_blocks(blocks, k):
    return {j: [list(row) for row in zip(*blocks[-j])] for j in range(-k, k + 1)}


def _ordered_blocks(blocks, k=None):
    if isinstance(blocks, Mapping):
        if k is None:
            k = max(blocks)
        if set(blocks) != set(range(-k, k + 1)):
            raise ValueError("mapping must have exactly all lag keys -k..k")
        ordered = [blocks[j] for j in range(-k, k + 1)]
    else:
        ordered = list(blocks)
        if k is None:
            k = len(ordered) // 2
        if len(ordered) != 2 * k + 1:
            raise ValueError("sequence must be ordered A_-k,...,A_k")
    s = len(ordered[k])
    if s < 1 or any(len(b) != s or any(len(r) != s for r in b) for b in ordered):
        raise ValueError("equal square blocks required")
    return {j: [list(r) for r in ordered[j + k]] for j in range(-k, k + 1)}, k, s


class _Core:
    """Forward preparation for repeated dense blocks A_-k..A_k (m block rows)."""

    def __init__(self, F, blocks, m, k, s, *, dense_backend, strassen_leaf,
                 schedule, stats, timers, ledger, tag=""):
        self.tag = tag
        self.F, self.blocks, self.m, self.k, self.s = F, blocks, m, k, s
        self.dense_backend, self.strassen_leaf = dense_backend, strassen_leaf
        self.schedule = schedule
        self.stats, self.timers, self.ledger = stats, timers, ledger
        self.query_stats = {}
        self.tree = occurrence_tree(m)
        with self._timed(tag + "setup_types", "setup", "types"):
            self.types = build_types(F, blocks, m, k, s, stats=stats)
        with self._timed(tag + "setup_root_inverse", "setup", "root"):
            self.root_inv = root_inverse(F, self.types[m], m, stats=stats)

    @contextmanager
    def _timed(self, name, stage, fstage):
        original = getattr(self.F, "stage", None)
        if original is not None:
            self.F.stage = stage + ":" + fstage
        w, c = perf_counter(), process_time()
        with backend_scope(self.dense_backend, self.strassen_leaf) as led:
            try:
                yield
            finally:
                merge_ledger(self.ledger.setdefault(stage, {}), led)
                t = self.timers.setdefault(name, {"wall": 0.0, "cpu": 0.0, "calls": 0})
                t["wall"] += perf_counter() - w
                t["cpu"] += process_time() - c
                t["calls"] += 1
                if original is not None:
                    self.F.stage = original

    def solve(self, rhs):
        run = solve_batched if self.schedule == "batched" else solve_unbatched
        with self._timed(self.tag + "query_solve", "query", "solve"):
            return run(self, rhs)


class _Base:
    method = METHOD

    def __init__(self, base, blocks, block_count, k, *, dense_backend="classical",
                 strassen_leaf=1, schedule="batched", residual_check="off",
                 count_ops=False):
        if type(block_count) is not int or block_count < 1:
            raise ValueError("positive integer block_count required")
        if schedule not in ("batched", "unbatched"):
            raise ValueError("schedule must be batched or unbatched")
        if residual_check not in ("off", "classical", "fast"):
            raise ValueError("residual_check must be off, classical or fast")
        if dense_backend not in ("classical", "strassen"):
            raise ValueError("dense_backend must be classical or strassen")
        t0w, t0c = perf_counter(), process_time()
        self.F = Counted(base) if count_ops else base
        self.m, self.schedule, self.residual_check = block_count, schedule, residual_check
        self.dense_backend, self.strassen_leaf = dense_backend, strassen_leaf
        self.blocks, self.k, self.s = blocks, k, len(blocks[0])
        self.setup_stats, self.timers, self.ledger = {}, {}, {}
        self.core = _Core(self.F, blocks, block_count, k, self.s,
                          dense_backend=dense_backend, strassen_leaf=strassen_leaf,
                          schedule=schedule, stats=self.setup_stats,
                          timers=self.timers, ledger=self.ledger)
        self.N = block_count * self.s
        self.residual_seconds = {"wall": 0.0, "cpu": 0.0}
        self.queries = 0
        self.setup_seconds = {"wall": perf_counter() - t0w, "cpu": process_time() - t0c}
        self._transposed_core = None


    def _check(self, ok, what):
        if not ok:
            raise VerificationFailure("exact physical residual is nonzero (" + what + ")")

    def _residual(self, mode_fn):
        if self.residual_check == "off":
            return
        w, c = perf_counter(), process_time()
        with backend_scope(self.dense_backend, self.strassen_leaf) as led:
            try:
                mode_fn()
            finally:
                merge_ledger(self.ledger.setdefault("residual", {}), led)
                self.residual_seconds["wall"] += perf_counter() - w
                self.residual_seconds["cpu"] += process_time() - c

    def solve(self, rhs):
        rhs = list(rhs)
        self.queries += 1
        x = self.core.solve(rhs)
        self._residual(lambda: self._check(self._forward_ok(self.core, x, rhs), "forward"))
        return x

    def _forward_ok(self, core, x, rhs):
        F, c = self.F, core
        if self.residual_check == "classical":
            return residual_classical(F, c.blocks, c.m, c.k, c.s, x, rhs, False)
        return residual_fast(F, c.blocks, c.m, c.k, c.s, x, rhs)

    def statistics(self):
        types = self.core.types
        nonleaf = [t for t in types.values() if t.layout is not None]
        st = self.setup_stats
        tree = self.core.tree
        per_depth = {}
        for depth, length, ids in tree.up_batches:
            per_depth.setdefault(depth, []).append((length, len(ids)))
        report = {
            "method": METHOD, "m": self.m, "n": self.s, "s": self.s, "k": self.k,
            "N": self.N, "schedule": self.schedule,
            "dense_backend": self.dense_backend,
            "strassen_leaf": self.strassen_leaf if self.dense_backend == "strassen" else None,
            "dense_exponent": 3 if self.dense_backend == "classical" else "log2(7)",
            "residual_check": self.residual_check,
            "types": len(types), "nonleaf_types": len(nonleaf),
            "type_lengths": sorted(types),
            "max_r": st.get("max_r", 0), "max_q": st.get("max_q", 0), "max_c": st.get("max_c", 0),
            "q0_merges": sum(1 for t in nonleaf if t.q == 0),
            "q_positive_merges": sum(1 for t in nonleaf if t.q > 0),
            "spill_routes": sum(t.layout.spill_routes for t in nonleaf),
            "selection": {k_: v for k_, v in st.items() if k_.startswith(("select_", "base_", "root_"))},
            "inv_calls": st.get("select_inv_calls", 0),
            "base_scan_offset": st.get("base_scan_offset", 0),
            "root_square_selection": st.get("root_square_selection", 0),
            "select_max_depth": st.get("select_max_depth", 0),
            "batches_per_depth": per_depth,
            "leaf_depths": sorted(tree.leaf_depths),
            "max_t": self.core.query_stats.get("max_t", 0),
            "up_batches": self.core.query_stats.get("up_batches", 0),
            "down_batches": self.core.query_stats.get("down_batches", 0),
            "queries": self.queries,
            "seconds": {"setup": dict(self.setup_seconds),
                        "phases": {k_: dict(v) for k_, v in self.timers.items()},
                        "residual": dict(self.residual_seconds)},
            "strassen_ledger": {stage: dict(led) for stage, led in self.ledger.items()},
        }
        if hasattr(self.F, "counts"):
            report["field_operations"] = dict(self.F.counts)
        return report


class Ex1159Prepared(_Base):
    """Raw Toeplitz lags: B[i*n+a, j*n+b] = lags[i-j][a-b+n-1], |i-j| <= k.

    solve_transpose uses T^t = J T J (physical reversal of all m*n coordinates)
    on the same preparation: J solve(J c).
    """
    def __init__(self, base, lags, block_count, **options):
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
        vectors = [tuple(v) for v in ordered]
        width = len(vectors[k])
        if width < 1 or width % 2 != 1 or any(len(v) != width for v in vectors):
            raise ValueError("equal positive odd inner lag lengths required")
        n = (width + 1) // 2
        self.lags = tuple(vectors)
        blocks = {lag: [[vectors[lag + k][a - b + n - 1] for b in range(n)] for a in range(n)]
                  for lag in range(-k, k + 1)}
        super().__init__(base, blocks, block_count, k, **options)

    def solve_transpose(self, rhs):
        rhs = list(rhs)
        if len(rhs) != self.N:
            raise ValueError("physical RHS dimension mismatch")
        self.queries += 1
        work = rhs[::-1]
        y = self.core.solve(work)
        x = y[::-1]
        if self.residual_check == "classical":
            self._residual(lambda: self._check(residual_classical(
                self.F, self.blocks, self.m, self.k, self.s, x, rhs, True), "transpose"))
        elif self.residual_check == "fast":      
            self._residual(lambda: self._check(
                residual_fast(self.F, self.blocks, self.m, self.k, self.s, y, work), "transpose"))
        return x


class Ex1159BlocksPrepared(_Base):
    """Arbitrary repeated dense blocks; the transpose is a literal second
    preparation from A'_j = A_-j^t, paid again and built lazily."""
    def __init__(self, base, blocks, block_count, k, **options):
        mapping, k, s = _ordered_blocks(blocks, k)
        self._options = dict(options)
        self._raw_base = base
        super().__init__(base, mapping, block_count, k, **options)

    def solve_transpose(self, rhs):
        rhs = list(rhs)
        if len(rhs) != self.N:
            raise ValueError("physical RHS dimension mismatch")
        if self._transposed_core is None:
            opts = dict(self._options)
            opts.pop("residual_check", None)
            opts.pop("count_ops", None)
            self._transposed_core = _Core(
                self.F, _transpose_blocks(self.blocks, self.k), self.m, self.k, self.s,
                dense_backend=self.dense_backend, strassen_leaf=self.strassen_leaf,
                schedule=self.schedule, stats={}, timers=self.timers, ledger=self.ledger,
                tag="transpose_")
        self.queries += 1
        x = self._transposed_core.solve(rhs)
        if self.residual_check == "classical":
            self._residual(lambda: self._check(residual_classical(
                self.F, self.blocks, self.m, self.k, self.s, x, rhs, True), "transpose"))
        elif self.residual_check == "fast":
            self._residual(lambda: self._check(residual_fast(
                self.F, self._transposed_core.blocks, self.m, self.k, self.s, x, rhs), "transpose"))
        return x


def prepare_bfm(base, lags, block_count, **options):
    """Raw Toeplitz lag mapping/sequence (k inferred as in guarded_boundary.unified)."""
    return Ex1159Prepared(base, lags, block_count, **options)


def prepare_bfm_blocks(base, blocks, block_count, k, **options):
    return Ex1159BlocksPrepared(base, blocks, block_count, k, **options)
