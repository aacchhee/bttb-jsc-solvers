# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
class RankFailure(ValueError):
    pass


def zeros(f, rows, cols):
    return [[f.zero for _ in range(cols)] for _ in range(rows)]


def identity(f, n):
    a = zeros(f, n, n)
    for i in range(n):
        a[i][i] = f.one
    return a


def transpose(a):
    return [list(row) for row in zip(*a)]


def matvec(f, a, x):
    out = []
    for row in a:
        total = f.zero
        for u, v in zip(row, x):
            total = f.add(total, f.mul(u, v))
        out.append(total)
    return out


def normalize(f, panel, columns):
    """Return transformed panel and local Q, pivoting only first columns.

    Fixed column order and first remaining nonzero row; Q*M=[I;0].
    panel has at most 4s rows. Its column count is also O(s).
    """
    from bttb_solvers._internal.halfstrip_core.flint_backend import normalize as flint_normalize
    accelerated = flint_normalize(f, panel, columns, RankFailure)
    if accelerated is not None:
        return accelerated
    a = [row[:] for row in panel]
    rows = len(a)
    q = identity(f, rows)
    for j in range(columns):
        pivot = next((i for i in range(j, rows)
                      if not f.zero_test(a[i][j])), None)
        if pivot is None:
            raise RankFailure("deficient local column " + str(j))
        a[j], a[pivot] = a[pivot], a[j]
        q[j], q[pivot] = q[pivot], q[j]
        scale = f.inv(a[j][j])
        a[j] = [f.mul(scale, v) for v in a[j]]
        q[j] = [f.mul(scale, v) for v in q[j]]
        for i in range(rows):
            if i == j or f.zero_test(a[i][j]):
                continue
            factor = a[i][j]
            a[i] = [f.sub(u, f.mul(factor, v))
                    for u, v in zip(a[i], a[j])]
            q[i] = [f.sub(u, f.mul(factor, v))
                    for u, v in zip(q[i], q[j])]
    return a, q


def endpoints(length):
    return [0] if length == 1 else [0, length - 1]


class Prepared:
    method = "guarded_boundary.boundary_normalized_q"

    def __init__(self, field, lower, diagonal, upper, block_count):
        self.field = field
        self.m = block_count
        self.s = len(diagonal)
        if not isinstance(self.m, int) or self.m < 1 or self.s < 1:
            raise ValueError("positive block count and block order required")
        for block in (lower, diagonal, upper):
            if len(block) != self.s or any(len(row) != self.s for row in block):
                raise ValueError("three equal square blocks required")
        self.L, self.D, self.U = ([row[:] for row in block]
                                  for block in (lower, diagonal, upper))
        self.cache = {}
        self.metrics = {"maximum_front_rows": self.s,
                        "maximum_seam_columns": 0, "rhs_nodes": 0}
        self.transpose_backend = None
        field.stage = "setup"
        self.root = self._build(self.m)
        field.stage = "root_setup"
        _, self.root_inverse = normalize(field, self.root["S"],
                                          len(self.root["S"]))

    def _build(self, length):
        if length in self.cache:
            return self.cache[length]
        f, s = self.field, self.s
        if length == 1:
            state = {"length": 1, "S": self.D, "F": self.L, "G": self.U}
            self.cache[1] = state
            return state
        a = length // 2
        b = length - a
        left, right = self._build(a), self._build(b)
        labels_left = endpoints(a)
        labels_right = [a + i for i in endpoints(b)]
        labels = labels_left + labels_right
        dl, dr = len(labels_left) * s, len(labels_right) * s
        rows = dl + dr
        h = zeros(f, rows, rows)
        jl, jr = zeros(f, rows, s), zeros(f, rows, s)
        for i in range(dl):
            h[i][:dl] = left["S"][i][:]
            jl[i] = left["F"][i][:]

            for j in range(s):
                h[i][dl + j] = f.add(h[i][dl + j], left["G"][i][j])
        for i in range(dr):
            h[dl + i][dl:] = right["S"][i][:]
            jr[dl + i] = right["G"][i][:]

            for j in range(s):
                h[dl + i][dl - s + j] = f.add(
                    h[dl + i][dl - s + j], right["F"][i][j])
        parent_labels = endpoints(length)
        seam_labels = [i for i in labels if i not in parent_labels]
        ordered_labels = seam_labels + parent_labels
        permutation = [labels.index(label) * s + j
                       for label in ordered_labels for j in range(s)]
        seam = len(seam_labels) * s
        retained = len(parent_labels) * s
        panel = [[h[i][j] for j in permutation] + jl[i] + jr[i]
                 for i in range(rows)]
        transformed, q = normalize(f, panel, seam)
        rest = [row[seam:] for row in transformed]
        state = {"length": length, "left": left, "right": right,
                 "a": a, "labels": labels, "seam_labels": seam_labels,
                 "parent_labels": parent_labels, "q": seam, "Q": q,
                 "C": [row[:retained] for row in rest[:seam]],
                 "V": [row[retained:retained+s] for row in rest[:seam]],
                 "W": [row[retained+s:] for row in rest[:seam]],
                 "S": [row[:retained] for row in rest[seam:]],
                 "F": [row[retained:retained+s] for row in rest[seam:]],
                 "G": [row[retained+s:] for row in rest[seam:]]}
        self.cache[length] = state
        self.metrics["maximum_front_rows"] = max(
            rows, self.metrics["maximum_front_rows"])
        self.metrics["maximum_seam_columns"] = max(
            seam, self.metrics["maximum_seam_columns"])
        return state

    def solve(self, rhs):
        if len(rhs) != self.m * self.s:
            raise ValueError("physical RHS dimension mismatch")
        f, s = self.field, self.s
        f.stage = "rhs_up"
        nodes = [0]

        def up(state, start):
            nodes[0] += 1
            if state["length"] == 1:
                return {"rho": list(rhs[start*s:(start+1)*s])}
            left = up(state["left"], start)
            right = up(state["right"], start + state["a"])
            transformed = matvec(f, state["Q"], left["rho"] + right["rho"])
            q = state["q"]
            return {"rho": transformed[q:], "eta": transformed[:q],
                    "left": left, "right": right}

        actual = up(self.root, 0)
        f.stage = "root_query"
        p = matvec(f, self.root_inverse, actual["rho"])
        output = [f.zero for _ in rhs]
        f.stage = "rhs_down"

        def down(state, node, start, parent, alpha, beta):
            if state["length"] == 1:
                output[start*s:(start+1)*s] = parent
                return
            z = node["eta"][:]
            for panel, vector in ((state["C"], parent),
                                  (state["V"], alpha),
                                  (state["W"], beta)):
                product = matvec(f, panel, vector)
                z = [f.sub(u, v) for u, v in zip(z, product)]
            values = {}
            for i, label in enumerate(state["parent_labels"]):
                values[label] = parent[i*s:(i+1)*s]
            for i, label in enumerate(state["seam_labels"]):
                values[label] = z[i*s:(i+1)*s]
            a = state["a"]
            lp = [v for label in endpoints(a) for v in values[label]]
            rp = [v for label in endpoints(state["length"]-a)
                  for v in values[a+label]]
            down(state["left"], node["left"], start, lp, alpha, values[a])
            down(state["right"], node["right"], start+a, rp, values[a-1], beta)

        zero = [f.zero] * s
        down(self.root, actual, 0, p, zero, zero)
        self.metrics["rhs_nodes"] += nodes[0]
        return output

    def solve_transpose(self, rhs):
        if self.transpose_backend is None:
            self.transpose_backend = Prepared(self.field, transpose(self.U),
                                              transpose(self.D), transpose(self.L),
                                              self.m)
        return self.transpose_backend.solve(rhs)

    def statistics(self):
        return {"method": self.method, "cached_lengths": sorted(self.cache),
                **self.metrics, "operations": dict(self.field.counts)}


def prepare(field, lower, diagonal, upper, block_count):
    return Prepared(field, lower, diagonal, upper, block_count)


def prepare_toeplitz(field, lower_lags, diagonal_lags, upper_lags, n):
    """Each lag list is ordered -(n-1),...,0,...,n-1."""
    def block(lags):
        if len(lags) != 2*n-1:
            raise ValueError("Toeplitz lag dimension mismatch")
        return [[lags[i-j+n-1] for j in range(n)] for i in range(n)]
    return prepare(field, block(lower_lags), block(diagonal_lags),
                   block(upper_lags), n)
