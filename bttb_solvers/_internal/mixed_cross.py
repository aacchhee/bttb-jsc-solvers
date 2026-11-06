# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.rectangular_cauchy import RectangularCauchy, _ids
from bttb_solvers._internal.spatial_transforms import SpatialTransform, correlation, row_product


def _shape(factor, rows, width, precision):
    if len(factor) != rows or any(len(row) != width for row in factor):
        raise ValueError("factor dimensions mismatch")
    if any(len(poly) != precision for row in factor for poly in row):
        raise ValueError("factor precision mismatch")


def _transpose_rect(A):
    F = A.F
    return RectangularCauchy(A.catalog, A.column_ids, A.row_ids,
        [[[F.neg(x) for x in poly] for poly in row] for row in A.H],
        A.G, A.collisions, A.precision, width=A.width)


class MixedUpperCross:
    """N^t Cphys-Cphys Lambda=E H^t, plus passive rectangular rows.

    External ordering is physical rows followed by passive_ids. Active columns
    are active_ids. Gpassive and H obey the passive exact displacement.
    """
    def __init__(self, catalog, active_ids, E, H, passive_ids, Gpassive,
                 collisions, precision, *, width=None):
        self.catalog, self.F, self.precision = catalog, catalog.F, precision
        self.active_ids, self.passive_ids = _ids(catalog, active_ids), _ids(catalog, passive_ids)
        self.n = len(E)
        if width is None:
            width = len(E[0]) if E else len(H[0]) if H else len(Gpassive[0]) if Gpassive else 0
        self.passive = RectangularCauchy(catalog, self.passive_ids, self.active_ids,
            Gpassive, H, collisions, precision, width=width)
        self.width = width
        _shape(E, self.n, width, precision)
        self.E = [[list(poly) for poly in row] for row in E]
        self.H = self.passive.H

        
        points = [self.F.inv(catalog.labels[i]) for i in self.active_ids]
        self.transform = SpatialTransform(self.F, points, max(1, self.n+1))
        self.mul = self.transform.mul

    def _zero(self): return [self.F.zero() for _ in range(self.precision)]
    def _add(self, a, b): return [self.F.add(x, y) for x, y in zip(a, b)]
    def _neg(self, a): return [self.F.neg(x) for x in a]
    def _mul(self, a, b): return self.mul(a, b)[:self.precision]

    def action(self, vector, *, transpose=False):
        F, P, n, s = self.F, self.precision, self.n, len(self.active_ids)
        source_size = n+len(self.passive_ids) if transpose else s
        if len(vector) != source_size or any(len(poly) != P for poly in vector):
            raise ValueError("mixed vector shape mismatch")
        if not transpose:
            physical = [self._zero() for _ in range(n)]
            for u in range(self.width):
                weights = [self._mul(self.H[j][u], vector[j]) for j in range(s)]
                moments = self.transform.moments(weights, P, n, first_power=True)
                terms = correlation(F, [row[u] for row in self.E], moments, P, self.mul)
                for i in range(n):
                    physical[i] = self._add(physical[i], self._neg(terms[i]))
            return physical + self.passive.action(vector)
        out = self.passive.action(vector[n:], transpose=True)
        for u in range(self.width):
            f = correlation(F, [row[u] for row in self.E], vector[:n], P, self.mul)
            values = self.transform.evaluate([self._zero()] + f, P)
            for j in range(s):
                out[j] = self._add(out[j], self._neg(self._mul(self.H[j][u], values[j])))
        return out

    def normalize(self, B):
        """Equation (22): literal upper cross C B with actual collisions."""
        if self.catalog is not B.catalog or self.precision != B.precision:
            raise ValueError("same provided catalog and precision required")
        if set(self.active_ids) != set(B.row_ids):
            raise ValueError("normalization middle IDs must agree")
        positions = {i: a for a, i in enumerate(B.row_ids)}
        physical = [[list(poly) for poly in row] for row in self.E]
        for u in range(B.width):
            vector = [B.G[positions[i]][u] for i in self.active_ids]
            column = self.action(vector)
            for i in range(self.n):
                physical[i].append(column[i])
        passive = self.passive.product(B)
        return MixedUpperCross(self.catalog, B.column_ids, physical, passive.H,
            self.passive_ids, passive.G, passive.collisions, self.precision,
            width=self.width+B.width)


class MixedLowerCross:
    """D by its transposed upper cross, preserving the required (-F,G) sign."""
    def __init__(self, transposed_upper):
        self.upper = transposed_upper
        self.catalog = transposed_upper.catalog
        self.active_ids = transposed_upper.active_ids
        self.passive_ids = transposed_upper.passive_ids
        self.n, self.precision = transposed_upper.n, transposed_upper.precision

    @classmethod
    def from_factors(cls, catalog, active_ids, G, Fphysical, passive_ids,
                     Fpassive, collisions, precision, *, width=None):
        F = catalog.F
        negative = lambda factor: [[[F.neg(x) for x in poly] for poly in row] for row in factor]
        return cls(MixedUpperCross(catalog, active_ids, negative(Fphysical), G,
            passive_ids, negative(Fpassive), collisions, precision, width=width))

    def action(self, vector, *, transpose=False):
        return self.upper.action(vector, transpose=not transpose)

    def normalize_left(self, B):
        """B D=(D^t B^t)^t, including passive collisions and transpose signs."""
        return MixedLowerCross(self.upper.normalize(_transpose_rect(B)))


def external_product_boundary(C, B, D):
    """Literal first external row and passive diagonal of C B D (1141 §5)."""
    if C.n < 1 or C.n != D.n or C.catalog is not D.catalog:
        raise ValueError("matching nonempty physical external spaces required")
    if C.precision != D.precision or set(C.passive_ids) != set(D.passive_ids):
        raise ValueError("matching passive external IDs and precision required")
    normalized = C.normalize(B)
    passive_product = normalized.passive.product(_transpose_rect(D.upper.passive))
    e0 = [C._zero() for _ in range(C.n+len(C.passive_ids))]
    e0[0][0] = C.F.one()
    first = C.action(e0, transpose=True)
    position = {i: a for a, i in enumerate(C.active_ids)}
    middle = B.action([first[position[i]] for i in B.row_ids], transpose=True)
    position = {i: a for a, i in enumerate(B.column_ids)}
    row = D.action([middle[position[i]] for i in D.active_ids], transpose=True)
    return {"first_row": row, "passive_diagonal": passive_product.collisions}


class PhysicalDisplacement:
    """algorithm (25), actual N^t X-X N factors and literal first row."""
    def __init__(self, F, E, H, first_row, *, width=None):
        from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier
        self.F, self.n = F, len(first_row)
        if not self.n or not first_row[0]:
            raise ValueError("nonempty physical first row required")
        self.precision = len(first_row[0])
        self.width = len(E[0]) if width is None and E else width or 0
        if type(self.width) is not int or self.width < 0:
            raise ValueError("nonnegative integer width required")
        if any(len(poly) != self.precision for poly in first_row):
            raise ValueError("first row precision mismatch")
        _shape(E, self.n, self.width, self.precision)
        _shape(H, self.n, self.width, self.precision)
        self.E = [[list(poly) for poly in row] for row in E]
        self.H = [[list(poly) for poly in row] for row in H]
        self.first_row = [list(poly) for poly in first_row]
        self.mul = make_multiplier(F)

    def action(self, vector, *, transpose=False):
        F, n, P = self.F, self.n, self.precision
        if len(vector) != n or any(len(poly) != P for poly in vector):
            raise ValueError("physical vector shape mismatch")
        out = correlation(F, self.first_row, vector, P, self.mul)
        for u in range(self.width):
            e, h = [row[u] for row in self.E], [row[u] for row in self.H]
            if not transpose:
                v = correlation(F, h, vector, P, self.mul)
                v = row_product(F, e, v, P, self.mul)
                term = [[F.zero() for _ in range(P)]] + v[:n-1]
            else:
                shifted = list(vector[1:]) + [[F.zero() for _ in range(P)]]
                v = correlation(F, shifted, e, P, self.mul)
                term = correlation(F, h, v, P, self.mul)
            out = [[F.add(x, y) for x, y in zip(a, b)] for a, b in zip(out, term)]
        return out

    def symmetric_moment_diagonal(self, moment_catalog, moment_ids):
        """(24): S_ii=1-v(point_i)^t X v(point_i), assuming X symmetric.

        moment_catalog contains actual moment points, NOT operator labels.
        Only first Hasse/formal derivative coefficients occur; no division2.
        """
        if moment_catalog.F is not self.F:
            raise ValueError("same native field instance required")
        ids = _ids(moment_catalog, moment_ids)
        F, n, P = self.F, self.n, self.precision
        zero = lambda: [F.zero() for _ in range(P)]

        def derivative(rows):
            index, result = F.zero(), []
            for i, row in enumerate(rows):
                if i:
                    result.append([F.mul(index, x) for x in row])
                index = F.add(index, F.one())
            return result

        polynomial = [list(row) for row in self.first_row] + [zero() for _ in range(n)]
        for i, row in enumerate(derivative(self.first_row), start=1):
            polynomial[i] = [F.add(x, y) for x, y in zip(polynomial[i], row)]
        for u in range(self.width):
            term = row_product(F, [row[u] for row in self.E],
                derivative([row[u] for row in self.H]), P, self.mul)
            for i, row in enumerate(term, start=2):
                polynomial[i] = [F.add(x, y) for x, y in zip(polynomial[i], row)]
        transform = SpatialTransform(F, [moment_catalog.labels[i] for i in ids], 2*n)
        moments = transform.evaluate(polynomial, P)
        result = []
        for moment in moments:
            value = [F.neg(x) for x in moment]
            value[0] = F.add(value[0], F.one())
            result.append(value)
        return result
