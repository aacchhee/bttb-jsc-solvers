# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
class HBPreparedDense:
    def __init__(self, F, g, h, stats):
        self.F, self.g, self.h, self.stats = F, g, h, stats
        d = len(g)
        if not d or len(h) != d:
            raise ValueError('positive matching generator dimensions required')
        width = len(g[0])
        if any(len(row) != width for row in g+h):
            raise ValueError('matching generator widths required')
        A = [[F.zero() for _ in range(d)] for _ in range(d)]
        for i in range(d):
            for j in range(d):
                value = F.zero()
                for a in range(width):
                    value = F.add(value, F.mul(g[i][a], h[j][a]))
                A[i][j] = F.add(value, A[i-1][j-1]) if i and j else value
        self.permutation = list(range(d))
        self.lu, self.singular = A, False
        self.diagonal_inverse = []
        for k in range(d):
            pivot = next((i for i in range(k,d) if not F.is_zero(A[i][k])), None)
            if pivot is None:
                self.singular = True
                break
            if pivot != k:
                A[k], A[pivot] = A[pivot], A[k]
                self.permutation[k], self.permutation[pivot] = self.permutation[pivot], self.permutation[k]
            inverse = F.inv(A[k][k])
            self.diagonal_inverse.append(inverse)
            for i in range(k+1,d):
                A[i][k] = F.mul(A[i][k], inverse)
                for j in range(k+1,d):
                    A[i][j] = F.sub(A[i][j], F.mul(A[i][k], A[k][j]))
        stats['hb_dense_preparations'] = stats.get('hb_dense_preparations',0)+1
        stats.update(hb_dense_order=d, hb_dense_generator_width=width,
                     hb_dense_stored_cells=d*d, hb_dense_inverse_materialized=False)

    def solve_stein(self, g, h, rhs=None, *, homogeneous=False):
        if g is self.g and h is self.h:
            transpose = False
        elif g is self.h and h is self.g:
            transpose = True
        else:
            raise ValueError('prepared HB terminal requires same-attempt generators')
        d, F, A = len(self.g), self.F, self.lu
        metadata = {'method':'HB_CLASSICAL_PREPARED_PLU', 'paid_dense_base':True}
        if homogeneous:

            
            return dict(metadata, status='NULLSPACE', nullity=int(self.singular),
                        nullity_is_gate_only=True, dimension=d)
        if self.singular:
            raise ArithmeticError('query on singular prepared HB inner matrix')
        if rhs is None or len(rhs) != d:
            raise ValueError('RHS dimension mismatch')
        if not transpose:
            x = [rhs[i] for i in self.permutation]
            for i in range(d):
                for j in range(i): x[i] = F.sub(x[i], F.mul(A[i][j],x[j]))
            for i in range(d-1,-1,-1):
                for j in range(i+1,d): x[i] = F.sub(x[i], F.mul(A[i][j],x[j]))
                x[i] = F.mul(x[i], self.diagonal_inverse[i])
        else:
            x = list(rhs)
            for i in range(d):
                for j in range(i): x[i] = F.sub(x[i], F.mul(A[j][i],x[j]))
                x[i] = F.mul(x[i], self.diagonal_inverse[i])
            for i in range(d-1,-1,-1):
                for j in range(i+1,d): x[i] = F.sub(x[i], F.mul(A[j][i],x[j]))
            unpermuted = [F.zero() for _ in range(d)]
            for i, old in enumerate(self.permutation): unpermuted[old] = x[i]
            x = unpermuted
        key = 'hb_dense_transpose_applications' if transpose else 'hb_dense_forward_applications'
        self.stats[key] = self.stats.get(key,0)+1
        return dict(metadata,status='CONSISTENT',solution=x)

    def invalidate(self):
        self.__dict__.clear()
        self.invalidated = True
