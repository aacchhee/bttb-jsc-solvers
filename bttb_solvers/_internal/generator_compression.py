# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from dataclasses import dataclass
from types import SimpleNamespace
from bttb_solvers._internal.halfstrip_core import uniform_terminal as uniform
from bttb_solvers._internal.halfstrip_core.fast_linear import inverse
from bttb_solvers._internal.kn_krylov import row_rank_profile


@dataclass
class CompressedGenerators:
    G: list
    H: list
    rank: int
    column_ids: tuple
    row_ids: tuple
    coordinates: list
    order: int
    input_width: int
    stats: dict


def _transpose(A):
    return [list(row) for row in zip(*A)]


def compress_generators(F, G, H, *, width=None):
    """Guarded exact compression, soft O(D*max(1,beta)^(tau-1)).

    All arithmetic/decisions retain the caller's eight-operation native field
    protocol and original Restart/guard exceptions. Input arrays are borrowed
    read-only. Explicit source arrays occupy O(D*beta+beta²) field slots;
    zero-width outputs additionally require O(D) empty row containers.
    """
    D = len(G)
    if len(H) != D:
        raise ValueError("generator orders disagree")
    if width is None:
        width = len(G[0]) if D else 0
    if type(width) is not int or width < 0:
        raise ValueError("nonnegative integer generator width required")
    if any(len(row) != width for factor in (G, H) for row in factor):
        raise ValueError("rectangular equal-width generators required")
    beta = width
    stats = {"column_profile": {}, "row_profile": {}, "inverse": {},
             "coordinates_product": {}, "right_factor_product": {},
             "materialized_shapes": [], "source_copy_cells": 0}

    def record(name, rows, columns):
        stats["materialized_shapes"].append((name, rows, columns))
        stats["source_copy_cells"] += rows*columns

    if not D or not beta:
        return CompressedGenerators([[] for _ in range(D)], [[] for _ in range(D)],
            0, (), (), [], D, beta, stats)

    

    context = SimpleNamespace(F=F, stats=stats["column_profile"], transpose=_transpose)
    transposed = _transpose(G)
    record("G_transpose", beta, D)
    columns = tuple(row_rank_profile(context, transposed))
    del transposed
    r = len(columns)
    if not r:
        return CompressedGenerators([[] for _ in range(D)], [[] for _ in range(D)],
            0, (), (), [], D, beta, stats)
    basis = [[row[j] for j in columns] for row in G]
    record("G_basis", D, r)
    context.stats = stats["row_profile"]
    rows = tuple(row_rank_profile(context, basis))
    if len(rows) != r:
        raise ArithmeticError("column basis and original-row profile disagree")
    minor = [[basis[i][j] for j in range(r)] for i in rows]
    selected = [list(G[i]) for i in rows]
    record("basis_minor", r, r)
    record("G_selected_rows", r, beta)
    minor_inverse = inverse(F, minor, stats=stats["inverse"])
    record("minor_inverse", r, r)

    
    coordinates = uniform.matrix_product(F, minor_inverse, selected,
                                         stats=stats["coordinates_product"])
    record("coordinates", r, beta)
    del minor, minor_inverse, selected
    transposed_coordinates = _transpose(coordinates)
    record("coordinates_transpose", beta, r)
    right = uniform.matrix_product(F, H, transposed_coordinates,
                                   stats=stats["right_factor_product"])
    record("H_compressed", D, r)
    return CompressedGenerators(basis, right, r, columns, rows, coordinates,
                                D, beta, stats)
