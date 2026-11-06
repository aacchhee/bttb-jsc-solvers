# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .kernel import matmul, matvec


def _columns(vectors):
    """list of equal-length column vectors -> row-list matrix (len x t)."""
    return [list(row) for row in zip(*vectors)]


def _distribute(layout, s, k, p, z, alpha, beta):
    """Child (p, alpha, beta) vectors from the parent's p, seam z and exterior vectors."""
    values = {}
    for i, label in enumerate(layout.parent_labels):
        values[label] = p[i * s:(i + 1) * s]
    for i, label in enumerate(layout.seam_labels):
        values[label] = z[i * s:(i + 1) * s]
    sources = ([values[label] for label in layout.labels],
               [alpha[i * s:(i + 1) * s] for i in range(k)],
               [beta[i * s:(i + 1) * s] for i in range(k)])
    out = []
    for index, child_labels in enumerate((layout.left_labels, layout.right_labels)):
        minus, plus = layout.routes[index]
        child_p = [v for label in child_labels for v in values[label]]
        child_alpha = [v for source, slot in minus for v in sources[source][slot]]
        child_beta = [v for source, slot in plus for v in sources[source][slot]]
        out.append((child_p, child_alpha, child_beta))
    return out


def _run(prep, rhs, batched):
    F, s, k, m = prep.F, prep.s, prep.k, prep.m
    tree, types = prep.tree, prep.types
    nodes = tree.nodes
    if len(rhs) != m * s:
        raise ValueError("physical RHS dimension mismatch")
    stats = prep.query_stats
    rho, eta = {}, {}
    for node in nodes:
        if node.length == 1:
            rho[node.id] = list(rhs[node.start * s:(node.start + 1) * s])

    for depth, length, ids in tree.up_batches:
        state = types[length]
        vectors = [rho[nodes[v].left] + rho[nodes[v].right] for v in ids]
        if state.q == 0:
            eta_part = [[] for _ in ids]
            rho_part = vectors             
        elif batched:
            Y = matmul(F, state.Q, _columns(vectors))
            cols = list(zip(*Y))
            eta_part = [list(col[:state.q]) for col in cols]
            rho_part = [list(col[state.q:]) for col in cols]
            stats["max_t"] = max(stats.get("max_t", 0), len(ids))
        else:
            ys = [matvec(F, state.Q, vec) for vec in vectors]
            eta_part = [y[:state.q] for y in ys]
            rho_part = [y[state.q:] for y in ys]
        for v, e, r in zip(ids, eta_part, rho_part):
            eta[v], rho[v] = e, list(r)
        stats["up_batches"] = stats.get("up_batches", 0) + 1

    root = nodes[0]
    zero = [F.zero() for _ in range(k * s)]
    inc = {0: (matvec(F, prep.root_inv, rho[0]), zero, zero)}
    output = [F.zero() for _ in range(m * s)]
    if root.length == 1:
        output[0:s] = inc[0][0]

    for depth, length, ids in tree.down_batches:
        state = types[length]
        layout = state.layout
        if state.q == 0:
            zs = [[] for _ in ids]
        else:
            vectors = [inc[v][0] + inc[v][1] + inc[v][2] for v in ids]
            if batched:
                Z = matmul(F, state.top, _columns(vectors))
                cols = list(zip(*Z))
                zs = [[F.sub(e, x) for e, x in zip(eta[v], col)] for v, col in zip(ids, cols)]
                stats["max_t"] = max(stats.get("max_t", 0), len(ids))
            else:
                zs = [[F.sub(e, x) for e, x in zip(eta[v], matvec(F, state.top, vec))]
                      for v, vec in zip(ids, vectors)]
        stats["down_batches"] = stats.get("down_batches", 0) + 1
        for v, z in zip(ids, zs):
            p, alpha, beta = inc.pop(v)
            children = _distribute(layout, s, k, p, z, alpha, beta)
            for child, triple in zip((nodes[v].left, nodes[v].right), children):
                if nodes[child].length == 1:
                    start = nodes[child].start
                    output[start * s:(start + 1) * s] = triple[0]
                else:
                    inc[child] = triple
            eta.pop(v, None)
            rho.pop(v, None)
    return output


def solve_batched(prep, rhs):
    return _run(prep, rhs, True)


def solve_unbatched(prep, rhs):
    return _run(prep, rhs, False)
