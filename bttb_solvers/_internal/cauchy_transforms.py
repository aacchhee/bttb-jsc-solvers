# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial/shared. Multipoint/Cauchy transform catalogs, reciprocal products and evaluation maps for generator acquisition.

Entry points: CauchyCatalog.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier


class CauchyCatalog:
    """Distinct unit labels in original catalog order, reusable across slices.

    F uses the halfstrip_core callable zero/one and guarded inv/is_zero protocol.
    Restart and zero-denominator exceptions propagate to the caller unchanged.
    """
    def __init__(self, F, labels):
        self.F = F
        self.labels = list(labels)
        self.s = len(self.labels)
        self.mul = make_multiplier(F)
        self.root = None
        if not self.s:
            return
        for label in self.labels:
            F.inv(label)  
        self.root = self._build(0, self.s)
        self.local_inverse = []
        p_jets = self._jets(self.root["ordinary"])
        for jet in p_jets:

            

            self.local_inverse.append(self._reciprocal(jet[1:5], 4))

    def _add(self, a, b):
        F = self.F
        return [F.add(a[i] if i < len(a) else F.zero(),
                      b[i] if i < len(b) else F.zero())
                for i in range(max(len(a), len(b)))]

    def _reciprocal(self, a, length, monic_reverse=False):
        F = self.F
        b = [F.one() if monic_reverse else F.inv(a[0])]
        while len(b) < length:
            t = min(2*len(b), length)
            ab = self.mul(a[:t], b)[:t]
            ab += [F.zero()]*(t-len(ab))
            error = [F.neg(x) for x in ab]
            error[0] = F.add(error[0], F.add(F.one(), F.one()))
            b = self.mul(b, error)[:t]
        return b

    def _build(self, lo, hi):
        F = self.F
        node = {"lo": lo, "hi": hi}
        if hi-lo == 1:
            linear = [F.neg(self.labels[lo]), F.one()]
            confluent = [F.one()]
            for _ in range(5):
                confluent = self.mul(confluent, linear)
            node.update(ordinary=linear, confluent=confluent, children=())
        else:
            mid = (lo+hi)//2
            left, right = self._build(lo, mid), self._build(mid, hi)
            node.update(ordinary=self.mul(left["ordinary"], right["ordinary"]),
                        confluent=self.mul(left["confluent"], right["confluent"]),
                        children=(left, right))

            
            for child, sibling in ((left, right), (right, left)):
                length = len(sibling["confluent"])-1
                child["reverse_inverse"] = self._reciprocal(
                    list(reversed(child["confluent"])), length, True)
        return node

    def _remainder(self, a, node):
        F = self.F
        b = node["confluent"]
        degree = len(b)-1
        if len(a) <= degree:
            return list(a)
        count = len(a)-degree
        quotient_reverse = self.mul(list(reversed(a))[:count],
                                    node["reverse_inverse"][:count])[:count]
        quotient = list(reversed(quotient_reverse))
        product = self.mul(quotient, b)
        return [F.sub(a[i], product[i]) for i in range(degree)]

    def _jets(self, polynomial):
        F = self.F
        out = [None]*self.s

        def descend(node, remainder):
            if not node["children"]:

                
                jet = []
                label = self.labels[node["lo"]]
                for value in reversed(remainder):
                    shifted = [F.zero()]*(len(jet)+1)
                    for j, term in enumerate(jet):
                        shifted[j] = F.add(shifted[j], F.mul(label, term))
                        shifted[j+1] = F.add(shifted[j+1], term)
                    shifted[0] = F.add(shifted[0], value)
                    jet = shifted
                out[node["lo"]] = jet + [F.zero()]*(5-len(jet))
            else:
                for child in node["children"]:
                    descend(child, self._remainder(remainder, child))

        descend(self.root, polynomial)
        return out

    def _numerator(self, weights):
        def recurse(node):
            if not node["children"]:
                return [weights[node["lo"]]]
            left, right = node["children"]
            return self._add(self.mul(recurse(left), right["ordinary"]),
                             self.mul(recurse(right), left["ordinary"]))
        return recurse(self.root)

    def apply(self, values, precision):
        """Return (S1,S2,S3), each s rows of exactly precision coefficients.

        values must contain s rows of exactly precision z coefficients. Static
        catalog IDs are row indices; subset weights may be zero-filled by the
        caller in O(sP). This API has no distinct source/target subset indices.
        """
        if type(precision) is not int or precision < 1:
            raise ValueError("precision must be a positive integer")
        if len(values) != self.s or any(len(row) != precision for row in values):
            raise ValueError("expected s by precision coefficient rows")
        F = self.F
        result = [[[F.zero() for _ in range(precision)] for _ in range(self.s)]
                  for _ in range(3)]
        if not self.s:
            return tuple(result)
        for z in range(precision):
            jets = self._jets(self._numerator([row[z] for row in values]))
            for i in range(self.s):
                quotient = self.mul(jets[i][:4], self.local_inverse[i])[:4]
                result[0][i][z] = quotient[1]
                result[1][i][z] = F.neg(quotient[2])
                result[2][i][z] = quotient[3]
        return tuple(result)
