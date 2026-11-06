# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.joint_arrays import SeriesOps, trim_factor
from bttb_solvers._internal.rectangular_cauchy import RectangularCauchy


def rect_at(A, P):
    if P < 1 or P > A.precision:
        raise ValueError("history precision unavailable")
    return RectangularCauchy(A.catalog, A.row_ids, A.column_ids,
        trim_factor(A.G, P), trim_factor(A.H, P),
        {i: list(poly[:P]) for i, poly in A.collisions.items()}, P, width=A.width)


class PrincipalHistory:
    def __init__(self, old_ids, pivot_ids, remainder_ids, inverse, ARI, AIR, precision):
        self.old_ids, self.pivot_ids, self.remainder_ids = tuple(old_ids), tuple(pivot_ids), tuple(remainder_ids)
        self.precision = precision
        self.inverse, self.ARI, self.AIR = (rect_at(A, precision) for A in (inverse, ARI, AIR))
        self.ops = SeriesOps(inverse.F)

    def left(self, vector, P):
        pos = {i: a for a, i in enumerate(self.old_ids)}
        pivot = rect_at(self.inverse, P).action([vector[pos[i]] for i in self.pivot_ids])
        correction = rect_at(self.ARI, P).action(pivot)
        remainder = [self.ops.sub(vector[pos[i]], correction[a]) for a, i in enumerate(self.remainder_ids)]
        return pivot, remainder

    def right(self, pivot, remainder, P):
        correction = rect_at(self.AIR, P).action(remainder)
        correction = rect_at(self.inverse, P).action(correction)
        transformed = [self.ops.sub(a, b) for a, b in zip(pivot, correction)] + list(remainder)
        ordered = self.pivot_ids+self.remainder_ids
        pos = {i: a for a, i in enumerate(ordered)}
        return [transformed[pos[i]] for i in self.old_ids]


class PairedHistories:
    def __init__(self, F, original_ids, precision):
        self.F, self.original_ids, self.precision = F, tuple(original_ids), precision
        self.records = []
        self.residual_ids = tuple(original_ids)

    def append(self, record):
        self.records.append(record)
        self.residual_ids = record.remainder_ids

    def left(self, vector, P):
        if type(P) is not int or len(vector) != len(self.original_ids) or not 1 <= P <= self.precision or any(len(p) != P for p in vector):
            raise ValueError("history input shape/precision mismatch")
        emitted, remainder = [], list(vector)
        for record in self.records:
            pivot, remainder = record.left(remainder, P)
            emitted.extend(pivot)
        return emitted+remainder

    def right(self, vector, P):
        if type(P) is not int or len(vector) != len(self.original_ids) or not 1 <= P <= self.precision or any(len(p) != P for p in vector):
            raise ValueError("history input shape/precision mismatch")
        pivot_vectors, at = [], 0
        for record in self.records:
            count = len(record.pivot_ids)
            pivot_vectors.append(vector[at:at+count])
            at += count
        remainder = list(vector[at:])
        for record, pivot in reversed(list(zip(self.records, pivot_vectors))):
            remainder = record.right(pivot, remainder, P)
        return remainder
