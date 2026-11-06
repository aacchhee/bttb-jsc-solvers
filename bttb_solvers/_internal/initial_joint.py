# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.cauchy_transforms import CauchyCatalog
from bttb_solvers._internal.compact_cauchy import CompactCauchy
from bttb_solvers._internal.mixed_cross import MixedUpperCross, MixedLowerCross, PhysicalDisplacement
from bttb_solvers._internal.spatial_transforms import SpatialTransform


class InitialJoint:
    """Literal initial [[BA,0,BA V],[0,0,I],[V^t BA,I,-S]].

    The raw joint factors use Delta(-S)=g j^t; active_diagonal stores S.
    External rows are n physical rows, then passive_ids in original order.
    Fscaled=z^common_depth F; initially depth is zero and Fscaled=F.
    """
    def __init__(self, setup):
        F, n, q, P = setup.F, setup.n, setup.q, setup.P
        if P != 8*setup.L:
            raise ValueError('initial full-source joint requires P=8L')
        self.setup, self.F = setup, F
        self.n, self.w, self.L, self.h, self.P = n, setup.w, setup.L, setup.h, P
        self.N = self.precision = P
        self.q, self.width = q, 4
        self.active_ids = self.passive_ids = tuple(range(q))
        self.common_depth = self.emitted_delay_sum = 0
        self.dU, self.dV = F.one(), F.one()
        self.operator_catalog = CauchyCatalog(F, setup.operator_labels)
        self.moment_transform = SpatialTransform(F, setup.moment_points, 2*n)
        self.moment_catalog = self.moment_transform.tree
        zero = self._zero

        a = [zero()] + [setup.H.lag_series(n-i, P) for i in range(1, n)]
        en = [zero() for _ in range(n)]
        en[-1][0] = F.one()
        b = setup.R.apply(en, P)  
        c = setup.R.apply(list(reversed(a)), P)  
        vb = self.moment_transform.evaluate(b, P)
        vc = self.moment_transform.evaluate(c, P)
        negative = lambda row: [F.neg(x) for x in row]
        Ephys = [[negative(b[i]), negative(c[i]), zero(), negative(b[i])]
                 for i in range(n)]
        Fphys = [[negative(c[i]), list(b[i]), list(b[i]), zero()]
                 for i in range(n)]
        self.E = Ephys + [[zero() for _ in range(4)] for _ in range(q)]
        self.Fscaled = Fphys + [[zero() for _ in range(4)] for _ in range(q)]
        self.g, self.j = [], []
        for i in range(q):
            rho = zero()
            rho[0] = setup.operator_labels[i]
            self.g.append([negative(vb[i]), negative(vc[i]), list(rho), negative(vb[i])])
            self.j.append([negative(vc[i]), list(vb[i]), list(vb[i]), list(rho)])
        self.physical = PhysicalDisplacement(F, Ephys, Fphys, b, width=4)
        self.active_diagonal = self.physical.symmetric_moment_diagonal(
            self.moment_catalog, self.active_ids)
        if any(not F.is_zero(row[0]) for row in self.active_diagonal):
            raise ArithmeticError('repaired initial Schur constant is nonzero')
        self.S = CompactCauchy(self.operator_catalog,
            [[negative(poly) for poly in row] for row in self.g],
            self.j, self.active_diagonal, P, width=4)
        unit = zero()
        unit[0] = F.one()
        self.C_collisions = {i: list(unit) for i in self.passive_ids}
        self.D_collisions = {i: list(unit) for i in self.passive_ids}
        self.X_first_row = [list(row) for row in b] + [zero() for _ in range(q)]
        self.X_passive_diagonal = {i: zero() for i in self.passive_ids}
        self.C = MixedUpperCross(self.operator_catalog, self.active_ids,
            Ephys, self.j, self.passive_ids, self.E[n:], self.C_collisions, P, width=4)
        self.D = MixedLowerCross.from_factors(self.operator_catalog, self.active_ids,
            self.g, Fphys, self.passive_ids, self.Fscaled[n:], self.D_collisions,
            P, width=4)
        self.stats = {'joint_width': 4, 'precision': P, 'physical_rows': n,
                      'active_rows': q, 'passive_rows': q,
                      'inverse_boundary_actions': 2,
                      'fixed_moment_evaluations': 2,
                      'factor_coefficient_cells': 8*(n+2*q)*P,
                      'active_diagonal_cells': q*P}

    def _zero(self):
        return [self.F.zero() for _ in range(self.P)]

    def _shape(self, vector, rows):
        if len(vector) != rows or any(len(row) != self.P for row in vector):
            raise ValueError('vector must have the exact initial-source shape and precision')

    def moment_evaluate(self, vector):
        """V^t times one n-vector of series, coefficientwise tree evaluation."""
        self._shape(vector, self.n)
        return self.moment_transform.evaluate(vector, self.P)

    def moment_synthesize(self, vector):
        """V times one q-vector, weighted rational-sum moments; no nqP cross."""
        self._shape(vector, self.q)
        return self.moment_transform.moments(vector, self.P, self.n)

    def BA_action(self, vector):
        """Symmetric BA=RJ action using the same analytic compact carrier."""
        self._shape(vector, self.n)
        return self.setup.R.apply(list(reversed(vector)), self.P)

    def action(self, vector):
        """One action of the literal full initial joint matrix, compactly."""
        n, q, F = self.n, self.q, self.F
        self._shape(vector, n+2*q)
        external, active = vector[:n+q], vector[n+q:]
        left = self.BA_action(external[:n]) + [self._zero() for _ in range(q)]
        upper = self.C.action(active)
        lower, schur = self.D.action(external), self.S.action(active)
        return [[F.add(x, y) for x, y in zip(a, b)] for a, b in zip(left, upper)] + [
            [F.sub(x, y) for x, y in zip(a, b)] for a, b in zip(lower, schur)]
