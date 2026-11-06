# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from dataclasses import dataclass, field as dc_field


@dataclass
class Node:
    id: int
    start: int
    length: int
    depth: int
    left: object = None     
    right: object = None


@dataclass
class Tree:
    nodes: list
    up_batches: list = dc_field(default_factory=list)    
    down_batches: list = dc_field(default_factory=list)  
    leaf_depths: set = dc_field(default_factory=set)


def occurrence_tree(m):
    """Balanced split a = length//2 (identical to algorithm); node 0 is the root."""
    nodes = []

    def build(start, length, depth):
        node = Node(len(nodes), start, length, depth)
        nodes.append(node)
        if length > 1:
            a = length // 2
            node.left = build(start, a, depth + 1).id
            node.right = build(start + a, length - a, depth + 1).id
        return node

    build(0, m, 0)
    groups = {}
    leaf_depths = set()
    for node in nodes:
        if node.length == 1:
            leaf_depths.add(node.depth)
        else:
            groups.setdefault((node.depth, node.length), []).append(node.id)
    ordered = sorted(groups, key=lambda key: (-key[0], key[1]))
    up = [(depth, length, groups[(depth, length)]) for depth, length in ordered]
    per_depth = {}
    for depth, length, _ in up:
        per_depth[depth] = per_depth.get(depth, 0) + 1
    if any(count > 2 for count in per_depth.values()):
        raise AssertionError("more than two non-leaf lengths at one depth")
    return Tree(nodes, up, list(reversed(up)), leaf_depths)
