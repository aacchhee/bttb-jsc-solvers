# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from dataclasses import dataclass, field as dc_field

from .incidence import Layout, merge_layout
from .kernel import matmul
from .selection import select_rows, build_q, invert_square, _bump, _bump_max


@dataclass
class TypeState:
    length: int
    layout: object          
    s: int
    k: int
    q: int                  
    d: int                  
    c: int                  
    Q: object               
    top: list               
    bottom: list            


def leaf_state(F, blocks, k, s):
    """S = A0, F = [A_k ... A_1], G = [A_-1 ... A_-k] (exactly as algorithm)."""
    bottom = []
    for i in range(s):
        row = list(blocks[0][i])
        row += [v for lag in range(k, 0, -1) for v in blocks[lag][i]]
        row += [v for lag in range(-1, -k - 1, -1) for v in blocks[lag][i]]
        bottom.append(row)
    return TypeState(1, None, s, k, 0, s, s + 2 * k * s, None, [], bottom)


def assemble(F, layout, left_bottom, right_bottom, s, k):
    """Literal incidence assembly of the stacked child equations.

    Returns (A_seam, M): A_seam is r x q (seam columns first), M is r x c with
    parent-retained columns, then alpha (left exterior) and beta (right exterior)
    slots.  Overlapping contributions add.
    """
    labels = layout.labels
    ext = k * s
    rows = len(labels) * s
    order = layout.seam_labels + layout.parent_labels
    newpos = {label: i * s for i, label in enumerate(order)}
    seam = len(layout.seam_labels) * s
    width = rows + 2 * ext
    asm = [[F.zero() for _ in range(width)] for _ in range(rows)]
    row_start = 0
    for child_labels, bottom, (minus, plus) in (
            (layout.left_labels, left_bottom, layout.routes[0]),
            (layout.right_labels, right_bottom, layout.routes[1])):
        dc = len(child_labels) * s
        if len(bottom) != dc:
            raise ValueError("child bottom has the wrong number of rows")
        for i in range(dc):
            row = bottom[i]
            if len(row) != dc + 2 * ext:
                raise ValueError("child bottom has the wrong number of columns")
            target = asm[row_start + i]
            for slot, label in enumerate(child_labels):
                dest = newpos[label]
                target[dest:dest + s] = row[slot * s:(slot + 1) * s]
            for offset, routed in ((dc, minus), (dc + ext, plus)):
                for h, (source, slot) in enumerate(routed):
                    dest = (newpos[labels[slot]] if source == 0
                            else rows + (source - 1) * ext + slot * s)
                    for j in range(s):
                        target[dest + j] = F.add(target[dest + j], row[offset + h * s + j])
        row_start += dc
    return [row[:seam] for row in asm], [row[seam:] for row in asm]


def normalize_type(F, A_seam, M, d, *, context=None, stats=None):
    """Return (Q, top, bottom).  q = 0: copy branch, no kernel call."""
    r = len(M)
    q = len(A_seam[0]) if (A_seam and A_seam[0]) else 0
    if q == 0:
        _bump(stats, "q0_merges")
        return None, [], [list(row) for row in M]
    P, Binv = select_rows(F, A_seam, stats=stats, context=context)
    Q, _ = build_q(F, A_seam, P, Binv)
    QM = matmul(F, Q, M, stats=None)
    return Q, QM[:q], QM[q:]


def build_types(F, blocks, m, k, s, *, stats=None):
    """All length types of the balanced tree on m leaves (memoised by length)."""
    cache = {}

    def build(length):
        if length in cache:
            return cache[length]
        if length == 1:
            cache[1] = leaf_state(F, blocks, k, s)
            return cache[1]
        layout = merge_layout(length, k)
        left, right = build(layout.a), build(layout.b)
        A_seam, M = assemble(F, layout, left.bottom, right.bottom, s, k)
        d = len(layout.parent_labels) * s
        q = len(layout.seam_labels) * s
        Q, top, bottom = normalize_type(
            F, A_seam, M, d, context={"length": length, "role": "merge", "m": m},
            stats=stats)
        state = TypeState(length, layout, s, k, q, d, d + 2 * k * s, Q, top, bottom)
        cache[length] = state
        _bump(stats, "spill_routes", layout.spill_routes)
        _bump_max(stats, "max_r", q + d)
        _bump_max(stats, "max_q", q)
        _bump_max(stats, "max_c", state.c)
        return state

    build(m)
    return cache


def root_inverse(F, root, m, *, stats=None):
    """Inverse of the root S = bottom[:, :d] by the selection primitive."""
    d = root.d
    S = [row[:d] for row in root.bottom]
    _bump(stats, "root_square_selection")
    return invert_square(F, S, stats=stats, context={"length": m, "role": "root", "m": m})
