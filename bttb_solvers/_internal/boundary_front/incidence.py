# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""BFM local-front implementation.

Role: Main/shared. Describe how retained/seam/exterior rows enter a front merge.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from dataclasses import dataclass

from bttb_solvers._internal.guarded_boundary.banded import retained


def route(length, k, position, labels):
    """(source, slot): 0 = retained union Y, 1 = left exterior, 2 = right exterior."""
    if 0 <= position < length:
        if position not in labels:
            raise ValueError("internal exterior position missing from retained union")
        return 0, labels.index(position)
    if -k <= position < 0:
        return 1, position + k
    if length <= position < length + k:
        return 2, position - length
    raise ValueError("exterior position outside parent formal window")


@dataclass(frozen=True)
class Layout:
    length: int
    k: int
    a: int
    b: int
    left_labels: list
    right_labels: list
    labels: list
    parent_labels: list
    seam_labels: list
    routes: list           
    spill_routes: int      


_LAYOUTS = {}


def merge_layout(length, k):
    key = (length, k)
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    if length < 2:
        raise ValueError("a merge needs length >= 2")
    a, b = length // 2, length - length // 2
    left_labels = retained(a, k)
    right_labels = [a + j for j in retained(b, k)]
    labels = left_labels + right_labels
    parent_labels = retained(length, k)
    seam_labels = [j for j in labels if j not in parent_labels]
    routes, spill = [], 0
    for child_length, start in ((a, 0), (b, a)):
        minus_pos = list(range(start - k, start))
        plus_pos = list(range(start + child_length, start + child_length + k))
        minus = [route(length, k, p, labels) for p in minus_pos]
        plus = [route(length, k, p, labels) for p in plus_pos]

        
        if start > 0:
            spill += sum(1 for p in minus_pos if p < 0)
        else:
            spill += sum(1 for p in plus_pos if p >= length)
        routes.append((minus, plus))
    layout = Layout(length, k, a, b, left_labels, right_labels, labels,
                    parent_labels, seam_labels, routes, spill)
    _LAYOUTS[key] = layout
    return layout
