# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.uniform_terminal import matrix_product, _inverse
from bttb_solvers._internal.constant_profile import rank_profile_kernel
from bttb_solvers._internal.q0_source import RawToeplitzLags, CompactInverse


class InitialCarrier:
    """Repair raw F to unit H, then acquire its compact inverse through P.

    Defaults to P=8(n+w), the full-source precision. An explicit positive P
    requests only this local carrier acquisition at that precision, without
    claiming that a later full-source stage can use less than 8(n+w).
    Moment candidates are ordered, distinct, nonzero units of native F.
    """
    def __init__(self, F, lags, n, w, moment_candidates, *, precision=None):
        raw = RawToeplitzLags(F, lags, n, w)
        if len(moment_candidates) != n:
            raise ValueError('exactly n candidate moment points required')
        P = 8*(n+w) if precision is None else precision
        if type(P) is not int or P < 1:
            raise ValueError('positive integer carrier precision required')
        candidates = list(moment_candidates)
        for i, point in enumerate(candidates):
            if F.is_zero(point):
                raise ValueError('candidate moment points must be nonzero units')
            for previous in candidates[:i]:
                if F.is_zero(F.sub(point, previous)):
                    raise ValueError('candidate moment points must be distinct with unit differences')
        self.F, self.n, self.w, self.L, self.h, self.P = F, n, w, n+w, n*w, P
        self.stats = {'raw_lag_cells': len(raw.lags)*(2*n-1),
                      'constant_profile': {}, 'moment_product': {},
                      'moment_selection': {}, 'vandermonde_cells': n*n,
                      'repair_additions': 2*n-1}
        profile = rank_profile_kernel(F, raw.constant_matrix(), kernel=True,
                                      stats=self.stats['constant_profile'])
        self.constant_rank = profile['rank']
        self.kernel = profile['kernel']
        self.q = n-self.constant_rank

        
        powers = [[F.one() for _ in range(n)]]
        for i in range(1, n):
            powers.append([F.mul(powers[-1][j], candidates[j]) for j in range(n)])
        kernel_transpose = [[self.kernel[i][a] for i in range(n)] for a in range(self.q)]
        moment_matrix = matrix_product(F, kernel_transpose, powers,
                                       stats=self.stats['moment_product'])
        selection = rank_profile_kernel(F, moment_matrix, ncols=n,
                                         stats=self.stats['moment_selection'])
        self.selected_ids = selection['column_profile']
        if selection['rank'] != self.q:
            raise ArithmeticError('moment candidate evaluation did not span the full kernel dual')
        self.moment_points = [candidates[i] for i in self.selected_ids]
        self.operator_labels = [_inverse(F, point) for point in self.moment_points]
        moments = [F.zero() for _ in range(2*n-1)]
        for point in self.moment_points:
            power = F.one()
            for d in range(2*n-1):
                moments[d] = F.add(moments[d], power)
                if d+1 < 2*n-1:
                    power = F.mul(power, point)
        repaired = [list(row) for row in raw.lags]
        repaired[0] = [F.add(value, moments[2*n-2-i])
                       for i, value in enumerate(repaired[0])]
        self.stats['kernel_cells'] = n*self.q
        self.stats['moment_matrix_shape'] = (self.q, n)
        self.stats['repaired_lag_cells'] = len(repaired)*(2*n-1)

        del powers, kernel_transpose, moment_matrix, selection, profile
        self.H = RawToeplitzLags(F, repaired, n, w)
        self.R = CompactInverse(self.H, P)
        self.detH0 = self.R.constant_determinant
