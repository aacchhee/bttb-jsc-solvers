# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
class HalfStripError(Exception):
    """Root of every error this package raises."""


class HypothesisFailure(HalfStripError):
    """A hypothesis of algorithm is violated: refuse, with the reason."""


class WidthOutOfRange(HypothesisFailure):
    """The Statement fixes 1 <= w < n.  w = 0 is the diagonal-only case (a different theorem) and
    w >= n is unrestricted BTTB, where the ledger is O(n^6) and the class is not
    this one."""

    def __init__(self, w, n):
        self.w, self.n = w, n
        super().__init__("1 <= w < n required; got w = %d, n = %d" % (w, n))


class SupportViolation(HypothesisFailure):
    """A lag A[delta_o, delta_i] with delta_i < -w is nonzero: the input is not
    in the half-strip class at this w."""

    def __init__(self, delta_i, delta_o, w):
        self.delta_i, self.delta_o, self.w = delta_i, delta_o, w
        super().__init__(
            "nonzero lag at (delta_o, delta_i) = (%d, %d) with delta_i < -w = -%d"
            % (delta_o, delta_i, w))


class SingularLeadingBlock(HypothesisFailure):
    """(H0) det B_{-w} = 0.  The causal recurrence has no leading coefficient to
    divide by; nothing in algorithm applies.  This is the ONLY condition this
    package imposes on B_{-w}."""

    def __init__(self, n=None, lag=None):
        self.n, self.lag = n, lag
        super().__init__("det B_{-w} = 0 (order %s): (H0) fails" % (n,))


class NoNTTRoot(HypothesisFailure):
    """(H3) the field carries no primitive root of unity of the order a
    quasi-linear 2-D convolution of this size would need.  Raised ONLY under
    `require_ntt_field=True` (or an explicit NTT backend). Without that flag
    auto mode may run the exact schoolbook oracle, whose report says
    `cubic_ledger_achieved=False`."""

    def __init__(self, q, order, field=None):
        self.q, self.order, self.field = q, order, field
        if q is None:
            super().__init__(
                "radix-two NTT backend is unavailable over %s (requested order "
                "%d); exact rational runs use the schoolbook correctness oracle "
                "and make no cubic or bit-complexity claim"
                % (field or "this non-finite field", order))
        else:
            super().__init__("no primitive root of unity of order %d in a field of "
                             "order %d (H3 cost model unattainable here)"
                             % (order, q))


class SingularTerminalMatrix(HalfStripError):
    """(H2) det B = 0 for the order-wn terminal matrix.

    A RESULT, not a bug: algorithm (a) proves nullity(T) = nullity(B), so this
    CERTIFIES that the physical operator T is singular, with nullity equal to
    the terminal nullity carried on this object."""

    def __init__(self, order=None, nullity=None):
        self.order, self.nullity = order, nullity
        super().__init__("terminal matrix of order %s singular: CERTIFIES "
                         "det T = 0 (nullity %s)" % (order, nullity))


class NotPrepared(HalfStripError):
    """prepare() has not been called."""


class RefutationFound(HalfStripError):
    """A PROVED statement failed on a concrete instance.  NOBODY CATCHES THIS."""

    def __init__(self, kind, witness):
        self.kind, self.witness = kind, witness
        super().__init__("REFUTATION (%s): %r" % (kind, witness))
