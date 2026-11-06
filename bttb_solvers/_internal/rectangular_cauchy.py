# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.cauchy_transforms import CauchyCatalog
from bttb_solvers._internal.compact_cauchy import CompactCauchy


def _ids(catalog, ids):
    result = tuple(ids)
    if any(type(i) is not int or i < 0 or i >= len(catalog.labels) for i in result):
        raise ValueError("catalog IDs must be valid integer positions")
    if len(set(result)) != len(result):
        raise ValueError("duplicate catalog ID")
    return result


def _union(*groups):

    return tuple(dict.fromkeys(i for group in groups for i in group))


class RectangularCauchy:
    def __init__(self, catalog, row_ids, column_ids, G, H, collisions,
                 precision, *, width=None):
        if type(precision) is not int or precision < 1:
            raise ValueError("positive integer precision required")
        self.catalog, self.F, self.precision = catalog, catalog.F, precision
        self.row_ids, self.column_ids = _ids(catalog, row_ids), _ids(catalog, column_ids)
        if len(G) != len(self.row_ids) or len(H) != len(self.column_ids):
            raise ValueError("factor row count mismatch")
        if width is None:
            width = len(G[0]) if G else len(H[0]) if H else 0
        if type(width) is not int or width < 0:
            raise ValueError("nonnegative integer width required")
        if any(len(row) != width for factor in (G, H) for row in factor):
            raise ValueError("generator width mismatch")
        if any(len(poly) != precision for factor in (G, H)
               for row in factor for poly in row):
            raise ValueError("generator coefficient count mismatch")
        overlap = set(self.row_ids) & set(self.column_ids)
        if set(collisions) != overlap or any(type(i) is not int for i in collisions):
            raise ValueError("exactly one collision per shared catalog ID required")
        if any(len(poly) != precision for poly in collisions.values()):
            raise ValueError("collision coefficient count mismatch")
        self.width = width
        self.G = [[list(p) for p in row] for row in G]
        self.H = [[list(p) for p in row] for row in H]
        self.collisions = {i: list(poly) for i, poly in collisions.items()}

        
        self._multiplier = None
        if overlap and width:
            from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier
            self._multiplier = make_multiplier(self.F)
            rows = {i: a for a, i in enumerate(self.row_ids)}
            columns = {i: b for b, i in enumerate(self.column_ids)}
            for i in overlap:
                defect = [self.F.zero() for _ in range(precision)]
                for u in range(width):
                    product = self._multiplier(self.G[rows[i]][u], self.H[columns[i]][u])
                    defect = [self.F.add(a, b) for a, b in zip(defect, product)]
                if any(not self.F.is_zero(value) for value in defect):
                    raise ValueError("nonzero collision displacement product")

    def _embed(self, union_ids, active_catalog):
        F, P, r = self.F, self.precision, self.width
        index = {i: a for a, i in enumerate(union_ids)}
        G = [[[F.zero() for _ in range(P)] for _ in range(r)] for _ in union_ids]
        H = [[[F.zero() for _ in range(P)] for _ in range(r)] for _ in union_ids]
        diagonal = [[F.zero() for _ in range(P)] for _ in union_ids]
        for a, i in enumerate(self.row_ids):
            G[index[i]] = [list(p) for p in self.G[a]]
        for b, i in enumerate(self.column_ids):
            H[index[i]] = [list(p) for p in self.H[b]]
        for i, poly in self.collisions.items():
            diagonal[index[i]] = list(poly)
        return CompactCauchy(active_catalog, G, H, diagonal, P, width=r)

    def _active(self, ids):

        return CauchyCatalog(self.F, [self.catalog.labels[i] for i in ids])

    def action(self, vector, *, transpose=False):
        source_ids = self.row_ids if transpose else self.column_ids
        target_ids = self.column_ids if transpose else self.row_ids
        P, F = self.precision, self.F
        if len(vector) != len(source_ids) or any(len(poly) != P for poly in vector):
            raise ValueError("vector shape mismatch")
        union_ids = _union(self.row_ids, self.column_ids)
        index = {i: a for a, i in enumerate(union_ids)}
        active = self._active(union_ids)
        embedded = self._embed(union_ids, active)
        weights = [[F.zero() for _ in range(P)] for _ in union_ids]
        for a, i in enumerate(source_ids):
            weights[index[i]] = list(vector[a])
        output = embedded.action(weights, transpose=transpose)
        return [output[index[i]] for i in target_ids]

    def product(self, other):
        """(6)–(8), including output collisions absent from the middle set."""
        if self.catalog is not other.catalog or self.precision != other.precision:
            raise ValueError("same provided catalog object and precision required")
        if set(self.column_ids) != set(other.row_ids):
            raise ValueError("middle catalog ID sets must agree")
        union_ids = _union(self.row_ids, self.column_ids, other.column_ids)
        index = {i: a for a, i in enumerate(union_ids)}
        active = self._active(union_ids)
        product = self._embed(union_ids, active).product(other._embed(union_ids, active))
        G = [product.G[index[i]] for i in self.row_ids]
        H = [product.H[index[i]] for i in other.column_ids]
        overlap = set(self.row_ids) & set(other.column_ids)
        collisions = {i: product.diagonal[index[i]] for i in overlap}
        return RectangularCauchy(self.catalog, self.row_ids, other.column_ids,
            G, H, collisions, self.precision, width=self.width+other.width)
