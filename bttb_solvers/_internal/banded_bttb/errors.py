# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
class BandedBTTBError(Exception):
    """Root of every error this package raises."""


class HypothesisFailure(BandedBTTBError):
    """An (H) of the solver is violated: refuse, with the reason."""


class CharacteristicDividesOrder(HypothesisFailure):
    """(H1) char K divides m.  NO extension of K contains a primitive m-th root
    of unity (x^m - 1 = (x^{m/p} - 1)^p has fewer than m distinct roots), so the
    block-DFT diagonalisation is unavailable altogether.  REFUSE."""

    def __init__(self, p, m):
        self.p, self.m = p, m
        super().__init__("char K = %d divides m = %d: no extension helps" % (p, m))


class NoRootOfUnity(HypothesisFailure):
    """(H1) m does not divide q-1 while char K does not divide m.  Carries
    ord_m(q), the required extension degree, which can be Theta(m)."""

    def __init__(self, order, q=None, m=None):
        self.order, self.q, self.m = order, q, m
        super().__init__("m does not divide q-1; required degree ord_m(q) = %s" % (order,))


class BandwidthOverlap(HypothesisFailure):
    """(H2) m < 2k+1.  The two-triangle description and the 2kn-column count are
    both REFUTED there (`-02` (d)).  Do not 'handle' the case."""

    def __init__(self, m, k):
        self.m, self.k = m, k
        super().__init__("m = %d < 2k+1 = %d: embedding invalid" % (m, 2 * k + 1))


class FieldTooSmall(HypothesisFailure):
    """(H3) (q-1)/m > 2kn fails.  Carries the degree d of the named extension."""

    def __init__(self, required_degree, q=None, m=None, k=None, n=None,
                 needed_nonzero=None):
        self.required_degree = required_degree
        self.q, self.m, self.k, self.n = q, m, k, n
        self.needed_nonzero = needed_nonzero
        if needed_nonzero is None:
            message = "(q-1)/m <= 2kn; required extension degree d = %d" % required_degree
        else:
            message = ("field has fewer than %d required nonzero evaluation points; "
                       "required extension degree d = %d"
                       % (needed_nonzero, required_degree))
        super().__init__(message)


class NeedsGenerator(HypothesisFailure):
    """Deterministic coset representatives need a generator of K*, equivalently
    the factorisation of q-1, which could not be obtained under the printed
    trial-division bound (T05 Open item 4)."""


class InvalidClassParameter(HypothesisFailure):
    """The requested shape is outside algorithm's stated class."""

    def __init__(self, parameter, value, requirement):
        self.parameter, self.value, self.requirement = parameter, value, requirement
        super().__init__("%s = %r is outside the proved class; require %s"
                         % (parameter, value, requirement))


class SingularToeplitz(BandedBTTBError):
    """det T == 0.  Normal control flow for the mu search."""

    def __init__(self, lag=None, n=None):
        self.lag, self.n = lag, n
        super().__init__("singular Toeplitz matrix of order %s" % (n,))


class SymbolSingular(BandedBTTBError):
    """det Chat_j(mu) == 0 at frequency j: this mu is bad."""

    def __init__(self, j=None, mu=None):
        self.j, self.mu = j, mu
        super().__init__("symbol singular at j = %s, mu = %s" % (j, mu))


class InvalidMu(BandedBTTBError):
    """A caller-supplied conjugation scalar is not in K*."""

    def __init__(self, mu=None):
        self.mu = mu
        super().__init__("caller-supplied mu must be a nonzero field element")


class CounterMismatch(BandedBTTBError):
    """The solver and its field must share one operation counter."""

    def __init__(self):
        super().__init__("solver counter must be identical to sym.F.cnt")


class SingularInnerMatrix(BandedBTTBError):
    """det M == 0.  By `-02` (b) det M = det B / det C with C nonsingular, so
    this CERTIFIES det B = 0.  It is a RESULT, not a bug."""

    def __init__(self, det_B_is_zero=True, r=None):
        self.det_B_is_zero, self.r = det_B_is_zero, r
        super().__init__("inner matrix of order %s singular: CERTIFIES det B = 0" % (r,))


class NotAPrimePower(BandedBTTBError):
    def __init__(self, q):
        self.q = q
        super().__init__("%r is not a prime power" % (q,))


class ZeroDivisionInField(BandedBTTBError):
    pass


class RefutationFound(BandedBTTBError):
    """A PROVED statement failed on a concrete instance.  NOBODY CATCHES THIS."""

    def __init__(self, kind, witness):
        self.kind, self.witness = kind, witness
        super().__init__("REFUTATION (%s): %r" % (kind, witness))


class NoGoodMu(RefutationFound):
    """Contradicts `-01` (b) under (H3)."""

    def __init__(self, tried, witness=None):
        self.tried = tried
        super().__init__("NoGoodMu", {"tried": tried, "witness": witness})


class RepairFailed(RefutationFound):
    """Contradicts `-03` (d) under |K| >= 2n-2.  Over GF(2), n = 4 it is the
    lemma being honest (`-03` (e1)) and the run must SAY so, not work around it."""

    def __init__(self, witness):
        super().__init__("RepairFailed", witness)
