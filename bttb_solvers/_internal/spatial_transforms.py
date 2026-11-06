# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip.allfield_poly import multiply_rectangles
from bttb_solvers._internal.cauchy_transforms import CauchyCatalog


class SpatialTransform:
    def __init__(self, F, points, length):
        if type(length) is not int or length < 1:
            raise ValueError("positive spatial length required")
        self.F, self.length = F, length
        self.tree = CauchyCatalog(F, points)
        self.points, self.mul = self.tree.labels, self.tree.mul
        self.root_inverse = []
        self.denominator_inverse = []
        if not self.points:
            return

        def prepare(node):
            node["moment_denominator"] = list(reversed(node["ordinary"]))
            for child in node["children"]:
                prepare(child)
        prepare(self.tree.root)
        Q = self.tree.root["moment_denominator"]
        self.denominator_inverse = self.tree._reciprocal(Q, length, True)
        degree = len(self.tree.root["confluent"])-1
        if length > degree:
            self.root_inverse = self.tree._reciprocal(
                list(reversed(self.tree.root["confluent"])), length-degree, True)

    def _shape(self, rows, precision):
        if type(precision) is not int or precision < 1:
            raise ValueError("positive temporal precision required")
        if any(len(row) != precision for row in rows):
            raise ValueError("temporal row length mismatch")

    def evaluate(self, rows, precision):
        """Evaluate spatial polynomial rows at every static point, all z slices.

        The root reduction pays for long polynomials even with few points.
        """
        self._shape(rows, precision)
        if len(rows) > self.length:
            raise ValueError("spatial input exceeds preprocessed length")
        F = self.F
        out = [[F.zero() for _ in range(precision)] for _ in self.points]
        if not self.points:
            return out
        divisor = self.tree.root["confluent"]
        degree = len(divisor)-1
        for z in range(precision):
            polynomial = [row[z] for row in rows]
            if len(polynomial) > degree:
                count = len(polynomial)-degree
                revq = self.mul(list(reversed(polynomial))[:count],
                                self.root_inverse[:count])[:count]
                product = self.mul(list(reversed(revq)), divisor)
                polynomial = [F.sub(polynomial[i], product[i]) for i in range(degree)]
            jets = self.tree._jets(polynomial)
            for i in range(len(self.points)):
                out[i][z] = jets[i][0]
        return out

    def moments(self, weights, precision, length, *, first_power=False):
        """Return sum_j weights_j point_j^(k+[first_power]), 0<=k<length."""
        self._shape(weights, precision)
        if len(weights) != len(self.points):
            raise ValueError("point weight count mismatch")
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("moment length exceeds preprocessed length")
        F = self.F
        out = [[F.zero() for _ in range(precision)] for _ in range(length)]
        if not self.points or length == 0:
            return out

        def numerator(node, values):
            if not node["children"]:
                return [values[node["lo"]]]
            left, right = node["children"]
            a = self.mul(numerator(left, values), right["moment_denominator"])
            b = self.mul(numerator(right, values), left["moment_denominator"])
            return self.tree._add(a, b)

        for z in range(precision):
            values = [F.mul(row[z], point) if first_power else row[z]
                      for row, point in zip(weights, self.points)]
            N = numerator(self.tree.root, values)
            coefficients = self.mul(N, self.denominator_inverse[:length])[:length]
            coefficients += [F.zero()]*(length-len(coefficients))
            for k in range(length):
                out[k][z] = coefficients[k]
        return out


def row_product(F, a, b, precision, multiplier):
    """Bivariate spatial/temporal convolution, truncated only in z."""
    if not a or not b:
        return []
    result = multiply_rectangles(F, a, b, multiplier)
    return [row[:precision] for row in result]


def correlation(F, a, b, precision, multiplier):
    """For equal spatial length n, output_i=sum_k a_(i+k) b_k mod z^P."""
    if len(a) != len(b):
        raise ValueError("equal correlation spatial lengths required")
    n = len(a)
    if not n:
        return []
    product = row_product(F, a, list(reversed(b)), precision, multiplier)
    return product[n-1:2*n-1]
