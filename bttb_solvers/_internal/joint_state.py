# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from copy import copy
from bttb_solvers._internal.joint_arrays import SeriesOps, trim_factor
from bttb_solvers._internal.rectangular_cauchy import RectangularCauchy
from bttb_solvers._internal.mixed_cross import MixedUpperCross, MixedLowerCross


class JointState:
    @classmethod
    def from_initial(cls, initial):
        result = cls()
        result.setup, result.catalog = initial.setup, initial.operator_catalog
        result.F, result.n, result.w = initial.setup.F, initial.setup.n, initial.setup.w
        result.L, result.h, result.precision = initial.setup.L, initial.setup.h, initial.setup.P
        if result.precision != 8*result.L:
            raise ValueError("full source requires initial precision8L")
        result.active_ids = tuple(initial.active_ids)
        result.passive_ids = tuple(initial.active_ids)
        result.width = 4
        result.g, result.j = trim_factor(initial.g, result.precision), trim_factor(initial.j, result.precision)
        result.E = trim_factor(initial.E, result.precision)
        result.Fscaled = trim_factor(initial.Fscaled, result.precision)
        result.active_diagonal = [list(poly) for poly in initial.active_diagonal]
        result.C_collisions = {i: list(poly) for i, poly in initial.C_collisions.items()}
        result.D_collisions = {i: list(poly) for i, poly in initial.D_collisions.items()}
        result.X_first_row = [list(poly) for poly in initial.X_first_row]
        result.X_passive_diagonal = {i: list(poly) for i, poly in initial.X_passive_diagonal.items()}
        result.common_depth = result.emitted_delay_sum = 0
        result.ops = SeriesOps(result.F)
        return result

    def copy(self):

        
        return copy(self)

    def active_block(self, rows, columns, precision=None):
        P = self.precision if precision is None else precision
        positions = {i: a for a, i in enumerate(self.active_ids)}
        G = [[[self.F.neg(x) for x in poly[:P]] for poly in self.g[positions[i]]] for i in rows]
        H = [[list(poly[:P]) for poly in self.j[positions[i]]] for i in columns]
        collisions = {i: list(self.active_diagonal[positions[i]][:P])
                      for i in set(rows) & set(columns)}
        return RectangularCauchy(self.catalog, rows, columns, G, H, collisions, P, width=self.width)

    def upper(self, ids, precision=None):
        P = self.precision if precision is None else precision
        positions = {i: a for a, i in enumerate(self.active_ids)}
        H = [[list(poly[:P]) for poly in self.j[positions[i]]] for i in ids]
        return MixedUpperCross(self.catalog, ids, trim_factor(self.E[:self.n], P), H,
            self.passive_ids, trim_factor(self.E[self.n:], P),
            {i: list(self.C_collisions[i][:P]) for i in ids if i in self.C_collisions},
            P, width=self.width)

    def lower(self, ids, precision=None):
        P = self.precision if precision is None else precision
        positions = {i: a for a, i in enumerate(self.active_ids)}
        G = [[list(poly[:P]) for poly in self.g[positions[i]]] for i in ids]
        return MixedLowerCross.from_factors(self.catalog, ids, G,
            trim_factor(self.Fscaled[:self.n], P), self.passive_ids,
            trim_factor(self.Fscaled[self.n:], P),
            {i: list(self.D_collisions[i][:P]) for i in ids if i in self.D_collisions},
            P, width=self.width)
