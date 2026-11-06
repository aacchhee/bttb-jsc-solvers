# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from collections.abc import Mapping
from .boundary import zeros, normalize, matvec, transpose


def retained(length, k):
    return sorted(set(range(min(k, length))) |
                  set(range(max(0, length-k), length)))


class BandedPrepared:
    method = "guarded_boundary.retained_k_normalized_q"

    def __init__(self, field, blocks, block_count, k):
        if not isinstance(k, int) or k < 1:
            raise ValueError("positive integer bandwidth required")
        if not isinstance(block_count, int) or block_count < 1:
            raise ValueError("positive integer block count required")
        if isinstance(blocks, Mapping):
            if set(blocks) != set(range(-k, k+1)):
                raise ValueError("mapping must have exactly all lag keys -k..k")
            ordered = [blocks[j] for j in range(-k, k+1)]
        else:
            ordered = list(blocks)
            if len(ordered) != 2*k+1:
                raise ValueError("sequence must be ordered A_-k,...,A_k")
        self.field, self.k, self.m = field, k, block_count
        self.s = len(ordered[k])
        if self.s < 1:
            raise ValueError("positive block order required")
        for block in ordered:
            if len(block) != self.s or any(len(row) != self.s for row in block):
                raise ValueError("equal square blocks required")
        self.blocks = {j: [list(row) for row in ordered[j+k]]
                       for j in range(-k, k+1)}
        self.cache = {}
        self.metrics = {"maximum_front_rows": self.s,
                        "maximum_seam_columns": 0, "rhs_nodes": 0}
        self.transpose_backend = None
        field.stage = "fixedk_setup"
        self.root = self._build(self.m)
        field.stage = "fixedk_root_setup"
        _, self.root_inverse = normalize(field, self.root["S"], len(self.root["S"]))

    def _route(self, length, position, labels):
        """(source, block slot): 0=y, 1=left exterior, 2=right exterior."""
        if 0 <= position < length:
            if position not in labels:
                raise ValueError("internal exterior position missing from retained union")
            return 0, labels.index(position)
        if -self.k <= position < 0:
            return 1, position + self.k
        if length <= position < length + self.k:
            return 2, position - length
        raise ValueError("exterior position outside parent formal window")

    def _build(self, length):
        if length in self.cache:
            return self.cache[length]
        f, s, k = self.field, self.s, self.k
        if length == 1:
            left = [[v for lag in range(k, 0, -1) for v in self.blocks[lag][i]]
                    for i in range(s)]
            right = [[v for lag in range(-1, -k-1, -1) for v in self.blocks[lag][i]]
                     for i in range(s)]
            state = {"length": 1, "retained": [0], "S": self.blocks[0],
                     "F": left, "G": right}
            self.cache[1] = state
            return state
        a, b = length // 2, length - length // 2
        left, right = self._build(a), self._build(b)
        left_labels = retained(a, k)
        right_labels = [a+j for j in retained(b, k)]
        labels = left_labels + right_labels
        rows = len(labels)*s
        parent_labels = retained(length, k)
        seam_labels = [j for j in labels if j not in parent_labels]
        seam, d, exterior = len(seam_labels)*s, len(parent_labels)*s, k*s
        assembled = zeros(f, rows, rows + 2*exterior)
        routes = []
        row_start = 0
        for child, start, child_labels in ((left, 0, left_labels),
                                           (right, a, right_labels)):
            child_rows = len(child_labels)*s
            minus = [self._route(length, p, labels) for p in range(start-k, start)]
            plus = [self._route(length, p, labels)
                    for p in range(start+child["length"], start+child["length"]+k)]
            routes.append((minus, plus))
            for i in range(child_rows):
                for block_slot, label in enumerate(child_labels):
                    destination = labels.index(label)*s
                    for j in range(s):
                        assembled[row_start+i][destination+j] = child["S"][i][block_slot*s+j]
                for panel, routed in ((child["F"], minus), (child["G"], plus)):
                    for h, (source, slot) in enumerate(routed):
                        destination = slot*s if source == 0 else rows + (source-1)*exterior + slot*s
                        for j in range(s):
                            assembled[row_start+i][destination+j] = f.add(
                                assembled[row_start+i][destination+j], panel[i][h*s+j])
            row_start += child_rows
        order = seam_labels + parent_labels
        permutation = [labels.index(label)*s+j for label in order for j in range(s)]
        panel = [[row[j] for j in permutation] + row[rows:] for row in assembled]
        transformed, q = normalize(f, panel, seam)
        tail = [row[seam:] for row in transformed]
        state = {"length": length, "a": a, "left": left, "right": right,
                 "retained": parent_labels, "labels": labels,
                 "left_labels": left_labels, "right_labels": right_labels,
                 "seam_labels": seam_labels, "routes": routes,
                 "q": seam, "Q": q,
                 "C": [row[:d] for row in tail[:seam]],
                 "V": [row[d:d+exterior] for row in tail[:seam]],
                 "W": [row[d+exterior:] for row in tail[:seam]],
                 "S": [row[:d] for row in tail[seam:]],
                 "F": [row[d:d+exterior] for row in tail[seam:]],
                 "G": [row[d+exterior:] for row in tail[seam:]]}
        self.cache[length] = state
        self.metrics["maximum_front_rows"] = max(rows, self.metrics["maximum_front_rows"])
        self.metrics["maximum_seam_columns"] = max(seam, self.metrics["maximum_seam_columns"])
        return state

    def solve(self, rhs):
        if len(rhs) != self.m*self.s:
            raise ValueError("physical RHS dimension mismatch")
        f, s, k = self.field, self.s, self.k
        f.stage = "fixedk_rhs_up"
        node_count = [0]

        def up(state, start):
            node_count[0] += 1
            if state["length"] == 1:
                return {"rho": list(rhs[start*s:(start+1)*s])}
            left = up(state["left"], start)
            right = up(state["right"], start+state["a"])
            transformed = matvec(f, state["Q"], left["rho"]+right["rho"])
            return {"rho": transformed[state["q"]:], "eta": transformed[:state["q"]],
                    "left": left, "right": right}

        actual = up(self.root, 0)
        f.stage = "fixedk_root_query"
        parent = matvec(f, self.root_inverse, actual["rho"])
        output = [f.zero for _ in rhs]
        f.stage = "fixedk_rhs_down"

        def down(state, node, start, p, alpha, beta):
            if state["length"] == 1:
                output[start*s:(start+1)*s] = p
                return
            z = node["eta"][:]
            for panel, vector in ((state["C"], p), (state["V"], alpha), (state["W"], beta)):
                z = [f.sub(u, v) for u, v in zip(z, matvec(f, panel, vector))]
            values = {label: p[i*s:(i+1)*s] for i, label in enumerate(state["retained"])}
            values.update({label: z[i*s:(i+1)*s]
                           for i, label in enumerate(state["seam_labels"])})
            sources = ([values[label] for label in state["labels"]],
                       [alpha[i*s:(i+1)*s] for i in range(k)],
                       [beta[i*s:(i+1)*s] for i in range(k)])
            for index, child_key in enumerate(("left", "right")):
                child_p = [v for label in state[child_key+"_labels"] for v in values[label]]
                minus, plus = state["routes"][index]
                child_alpha = [v for source, slot in minus for v in sources[source][slot]]
                child_beta = [v for source, slot in plus for v in sources[source][slot]]
                child_start = start if index == 0 else start+state["a"]
                down(state[child_key], node[child_key], child_start,
                     child_p, child_alpha, child_beta)

        zero = [f.zero]*(k*s)
        down(self.root, actual, 0, parent, zero, zero)
        self.metrics["rhs_nodes"] += node_count[0]
        return output

    def solve_transpose(self, rhs):
        if self.transpose_backend is None:
            blocks = {j: transpose(self.blocks[-j]) for j in range(-self.k, self.k+1)}
            self.transpose_backend = BandedPrepared(self.field, blocks, self.m, self.k)
        return self.transpose_backend.solve(rhs)

    def statistics(self):
        return {"method": self.method, "k": self.k, "s": self.s, "m": self.m,
                "cached_lengths": sorted(self.cache), **self.metrics,
                "operations": dict(self.field.counts)}


def prepare_banded(field, blocks, block_count, k):
    """blocks is a full lag mapping or a sequence ordered A_-k,...,A_k."""
    return BandedPrepared(field, blocks, block_count, k)
