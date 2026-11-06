# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .boundary import Prepared, RankFailure, zeros, endpoints, matvec, transpose


def apply_operations(field, operations, vector):
    result = list(vector)
    for kind, target, source, factor in operations:
        if kind == "swap":
            result[target], result[source] = result[source], result[target]
        elif kind == "scale":
            result[target] = field.mul(factor, result[target])
        elif kind == "axpy":
            result[target] = field.sub(result[target], field.mul(factor, result[source]))
        else:
            raise ValueError("unknown saved row operation")
    return result


def normalize_operations(field, panel, columns):
    """Normalize seam columns, recording the exact invertible row program."""
    rows = [list(row) for row in panel]
    operations = []
    for j in range(columns):
        pivot = next((i for i in range(j, len(rows))
                      if not field.zero_test(rows[i][j])), None)
        if pivot is None:
            raise RankFailure("deficient local column " + str(j))
        if pivot != j:
            rows[j], rows[pivot] = rows[pivot], rows[j]
            operations.append(("swap", j, pivot, None))
        factor = field.inv(rows[j][j])
        rows[j] = [field.mul(factor, value) for value in rows[j]]
        operations.append(("scale", j, None, factor))
        for i in range(len(rows)):
            if i == j or field.zero_test(rows[i][j]):
                continue
            factor = rows[i][j]
            rows[i] = [field.sub(u, field.mul(factor, v))
                       for u, v in zip(rows[i], rows[j])]
            operations.append(("axpy", i, j, factor))
    if len(operations) > columns*(len(rows)+1):
        raise AssertionError("saved seam operation bound violated")
    return rows, operations


def factor_root(field, matrix):
    """Forward elimination only; no normalized dense row transform or inverse."""
    rows = [list(row) for row in matrix]
    operations = []
    for j in range(len(rows)):
        pivot = next((i for i in range(j, len(rows))
                      if not field.zero_test(rows[i][j])), None)
        if pivot is None:
            raise RankFailure("singular root column " + str(j))
        if pivot != j:
            rows[j], rows[pivot] = rows[pivot], rows[j]
            operations.append(("swap", j, pivot, None))
        inverse = field.inv(rows[j][j])
        for i in range(j+1, len(rows)):
            if field.zero_test(rows[i][j]):
                continue
            factor = field.mul(rows[i][j], inverse)
            rows[i] = [field.sub(u, field.mul(factor, v))
                       for u, v in zip(rows[i], rows[j])]
            operations.append(("axpy", i, j, factor))
    d = len(rows)
    if len(operations) > d+d*(d-1)//2:
        raise AssertionError("saved root operation bound violated")
    return rows, operations


def triangular_solve(field, upper, operations, rhs):
    transformed = apply_operations(field, operations, rhs)
    result = [field.zero]*len(upper)
    for i in range(len(upper)-1, -1, -1):
        value = transformed[i]
        for j in range(i+1, len(upper)):
            value = field.sub(value, field.mul(upper[i][j], result[j]))
        if field.zero_test(upper[i][i]):
            raise RankFailure("zero root diagonal during back substitution")
        result[i] = field.mul(value, field.inv(upper[i][i]))
    return result


class OperationsPrepared(Prepared):
    method = "guarded_boundary.operations"

    def __init__(self, field, lower, diagonal, upper, block_count):
        self.field, self.m, self.s = field, block_count, len(diagonal)
        if not isinstance(self.m, int) or self.m < 1 or self.s < 1:
            raise ValueError("positive block count and block order required")
        for block in (lower, diagonal, upper):
            if len(block) != self.s or any(len(row) != self.s for row in block):
                raise ValueError("three equal square blocks required")
        self.L, self.D, self.U = ([list(row) for row in block]
                                  for block in (lower, diagonal, upper))
        self.cache = {}
        self.metrics = {"maximum_front_rows": self.s,
                        "maximum_seam_columns": 0, "rhs_nodes": 0}
        self.transpose_backend = None
        field.stage = "operations_setup"
        self.root = self._build(self.m)
        field.stage = "operations_root_setup"
        self.root_upper, self.root_operations = factor_root(field, self.root["S"])

    def _build(self, length):
        if length in self.cache:
            return self.cache[length]
        f, s = self.field, self.s
        if length == 1:
            state = {"length": 1, "S": self.D, "F": self.L, "G": self.U}
            self.cache[1] = state
            return state
        a, b = length//2, length-length//2
        left, right = self._build(a), self._build(b)
        labels_left = endpoints(a)
        labels_right = [a+i for i in endpoints(b)]
        labels = labels_left+labels_right
        dl, dr = len(labels_left)*s, len(labels_right)*s
        rows = dl+dr
        h = zeros(f, rows, rows)
        jl, jr = zeros(f, rows, s), zeros(f, rows, s)
        for i in range(dl):
            h[i][:dl] = left["S"][i][:]
            jl[i] = left["F"][i][:]
            for j in range(s):
                h[i][dl+j] = f.add(h[i][dl+j], left["G"][i][j])
        for i in range(dr):
            h[dl+i][dl:] = right["S"][i][:]
            jr[dl+i] = right["G"][i][:]
            for j in range(s):
                h[dl+i][dl-s+j] = f.add(h[dl+i][dl-s+j], right["F"][i][j])
        parent_labels = endpoints(length)
        seam_labels = [i for i in labels if i not in parent_labels]
        permutation = [labels.index(label)*s+j
                       for label in seam_labels+parent_labels for j in range(s)]
        seam, retained = len(seam_labels)*s, len(parent_labels)*s
        panel = [[h[i][j] for j in permutation]+jl[i]+jr[i] for i in range(rows)]
        transformed, operations = normalize_operations(f, panel, seam)
        rest = [row[seam:] for row in transformed]
        state = {"length": length, "left": left, "right": right,
                 "a": a, "labels": labels, "seam_labels": seam_labels,
                 "parent_labels": parent_labels, "q": seam,
                 "operations": operations, "front_rows": rows,
                 "C": [row[:retained] for row in rest[:seam]],
                 "V": [row[retained:retained+s] for row in rest[:seam]],
                 "W": [row[retained+s:] for row in rest[:seam]],
                 "S": [row[:retained] for row in rest[seam:]],
                 "F": [row[retained:retained+s] for row in rest[seam:]],
                 "G": [row[retained+s:] for row in rest[seam:]]}
        self.cache[length] = state
        self.metrics["maximum_front_rows"] = max(rows, self.metrics["maximum_front_rows"])
        self.metrics["maximum_seam_columns"] = max(seam, self.metrics["maximum_seam_columns"])
        return state

    def solve(self, rhs):
        if len(rhs) != self.m*self.s:
            raise ValueError("physical RHS dimension mismatch")
        f, s = self.field, self.s
        f.stage = "operations_rhs_up"
        nodes = [0]

        def up(state, start):
            nodes[0] += 1
            if state["length"] == 1:
                return {"rho": list(rhs[start*s:(start+1)*s])}
            left = up(state["left"], start)
            right = up(state["right"], start+state["a"])
            transformed = apply_operations(f, state["operations"], left["rho"]+right["rho"])
            q = state["q"]
            return {"rho": transformed[q:], "eta": transformed[:q],
                    "left": left, "right": right}

        actual = up(self.root, 0)
        f.stage = "operations_root_query"
        p = triangular_solve(f, self.root_upper, self.root_operations, actual["rho"])
        output = [f.zero for _ in rhs]
        f.stage = "operations_rhs_down"

        def down(state, node, start, parent, alpha, beta):
            if state["length"] == 1:
                output[start*s:(start+1)*s] = parent
                return
            z = node["eta"][:]
            for panel, vector in ((state["C"], parent), (state["V"], alpha), (state["W"], beta)):
                product = matvec(f, panel, vector)
                z = [f.sub(u, v) for u, v in zip(z, product)]
            values = {label: parent[i*s:(i+1)*s]
                      for i, label in enumerate(state["parent_labels"])}
            values.update({label: z[i*s:(i+1)*s]
                           for i, label in enumerate(state["seam_labels"])})
            a = state["a"]
            lp = [v for label in endpoints(a) for v in values[label]]
            rp = [v for label in endpoints(state["length"]-a) for v in values[a+label]]
            down(state["left"], node["left"], start, lp, alpha, values[a])
            down(state["right"], node["right"], start+a, rp, values[a-1], beta)

        zero = [f.zero]*s
        down(self.root, actual, 0, p, zero, zero)
        self.metrics["rhs_nodes"] += nodes[0]
        return output

    def solve_transpose(self, rhs):
        if self.transpose_backend is None:
            self.transpose_backend = OperationsPrepared(self.field, transpose(self.U),
                                                        transpose(self.D), transpose(self.L),
                                                        self.m)
        return self.transpose_backend.solve(rhs)

    def statistics(self):
        data = super().statistics()
        data["seam_operation_counts"] = {
            str(length): len(state.get("operations", ()))
            for length, state in self.cache.items()}
        data["seam_operation_bounds"] = {
            str(length): state.get("q", 0)*(state.get("front_rows", self.s)+1)
            for length, state in self.cache.items()}
        data["root_operation_count"] = len(self.root_operations)
        d = len(self.root_upper)
        data["root_operation_bound"] = d+d*(d-1)//2
        return data


def prepare(field, lower, diagonal, upper, block_count):
    return OperationsPrepared(field, lower, diagonal, upper, block_count)
