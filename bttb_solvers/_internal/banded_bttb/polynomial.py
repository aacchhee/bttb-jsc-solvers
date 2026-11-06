# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
def trim(F, a):
    end = len(a)
    while end and F.is_zero(a[end - 1]):
        end -= 1
    return list(a[:end])


def schoolbook(F, a, b):
    if not a or not b:
        return []
    out = [F.zero()] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] = F.add(out[i+j], F.mul(x, y))
    return trim(F, out)


def _ntt(F, values, inverse=False):
    size = len(values)
    if size < 1 or size & (size-1) or size > 4096:
        raise ValueError('GF12289 NTT requires power-of-two length <=4096')

    root = pow(11, (12289-1)//size, 12289)
    if inverse:
        root = F.inv(root)
    out = list(values)
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    width = 2
    while width <= size:
        stage_root = F.pow(root, size//width)
        for start in range(0, size, width):
            weight = F.one()
            for offset in range(width//2):
                even = out[start+offset]
                odd = F.mul(weight, out[start+offset+width//2])
                out[start+offset] = F.add(even, odd)
                out[start+offset+width//2] = F.sub(even, odd)
                weight = F.mul(weight, stage_root)
        width *= 2
    if inverse:
        scale = F.inv(size % 12289)
        out = [F.mul(x, scale) for x in out]
    return out


class PolynomialArithmetic:
    """Arithmetic context shared by every node of one or more trees.

    ``multiply`` permits a separately checked exact fast kernel. Its complexity
    is a caller assumption; ``stats`` reports which default kernels were used.
    Counters count calls, not field operations (the field owns those).
    """
    def __init__(self, F, multiply=None):
        self.F = F
        self.multiply = multiply
        self.stats = dict(ntt=0, schoolbook=0, callback=0)

    def add(self, a, b):
        F = self.F
        return trim(F, [F.add(a[i] if i < len(a) else F.zero(),
                              b[i] if i < len(b) else F.zero())
                        for i in range(max(len(a), len(b)))])

    def sub(self, a, b):
        F = self.F
        return trim(F, [F.sub(a[i] if i < len(a) else F.zero(),
                              b[i] if i < len(b) else F.zero())
                        for i in range(max(len(a), len(b)))])

    def mul(self, a, b):
        F = self.F
        if not a or not b:
            return []
        if self.multiply is not None:
            self.stats['callback'] += 1
            return trim(F, self.multiply(F, list(a), list(b)))
        needed = len(a)+len(b)-1
        size = 1 << (needed-1).bit_length()
        if (getattr(F, 'p', None), getattr(F, 'e', None)) == (12289, 1) and isinstance(F.one(), int) and needed >= 32 and min(len(a),len(b)) > 8 and size <= 4096:
            self.stats['ntt'] += 1
            left = _ntt(F, list(a)+[F.zero()]*(size-len(a)))
            right = _ntt(F, list(b)+[F.zero()]*(size-len(b)))
            return trim(F, _ntt(F, [F.mul(x,y) for x,y in zip(left,right)], True)[:needed])
        self.stats['schoolbook'] += 1
        return schoolbook(F, a, b)

    def inverse_series(self, a, length):
        """a^-1 mod x^length by Newton doubling, valid also in characteristic 2."""
        if length < 0:
            raise ValueError('negative series length')
        if not length:
            return []
        F = self.F
        if not a or F.is_zero(a[0]):
            raise ZeroDivisionError('series constant coefficient is zero')
        result = [F.inv(a[0])]
        while len(result) < length:
            target = min(2*len(result), length)
            product = self.mul(a[:target], result)[:target]
            correction = [F.neg(x) for x in product]
            if not correction:
                correction = [F.zero()]
            correction[0] = F.add(correction[0], F.add(F.one(), F.one()))
            result = self.mul(result, correction)[:target]
            result += [F.zero()]*(target-len(result))
        return result

    def divmod(self, a, b, inverse_cache=None):
        """Reversed Newton division; optional cache belongs to this divisor only."""
        F = self.F
        a, b = trim(F,a), trim(F,b)
        if not b:
            raise ZeroDivisionError('zero polynomial divisor')
        length = len(a)-len(b)+1
        if length <= 0:
            return [], a
        cached = inverse_cache.get('inverse', []) if inverse_cache is not None else []
        if len(cached) < length:
            cached = self.inverse_series(list(reversed(b)), length)
            if inverse_cache is not None:
                inverse_cache['inverse'] = cached
        reverse_q = self.mul(list(reversed(a))[:length], cached[:length])[:length]
        reverse_q += [F.zero()]*(length-len(reverse_q))
        quotient = trim(F, list(reversed(reverse_q)))
        remainder = self.sub(a, self.mul(quotient,b))
        return quotient, remainder

    def derivative(self, a):

        
        F = self.F
        integer = F.zero()
        out = []
        for value in a[1:]:
            integer = F.add(integer, F.one())
            out.append(F.mul(integer, value))
        return trim(F, out)


class _Node:
    def __init__(self, arithmetic, xs, start, stop):
        self.start, self.stop = start, stop
        self.inverse_cache = {}
        self.left = self.right = None
        if stop-start == 1:
            self.product = [arithmetic.F.neg(xs[start]), arithmetic.F.one()]
        else:
            mid = (start+stop)//2
            self.left = _Node(arithmetic, xs, start, mid)
            self.right = _Node(arithmetic, xs, mid, stop)
            self.product = arithmetic.mul(self.left.product, self.right.product)


class ProductTree:
    """One reusable tree for distinct, arbitrary nodes (zero is allowed).

    evaluate supports arbitrary polynomial degree. Interpolation reuses the
    tree and derivative inverse weights, initialized on first interpolation.
    Duplicate nodes are rejected when those weights are first requested.
    Conditional on multiplication cost M(N), balanced-degree tree operations
    take O(M(N) log N) under the usual summability assumptions. Arbitrary input
    degree D additionally requires the initial division at cost O(M(D)).
    The default schoolbook fallback does not satisfy a soft-linear bound.
    """
    def __init__(self, F, xs, arithmetic=None):
        self.arithmetic = arithmetic if arithmetic is not None else PolynomialArithmetic(F)
        if self.arithmetic.F is not F:
            raise ValueError('tree and arithmetic must use the same field object')
        self.xs = tuple(xs)
        self.root = _Node(self.arithmetic, self.xs, 0, len(xs)) if len(xs) else None
        self._weights = None

    def evaluate(self, polynomial):
        if self.root is None:
            return []
        A, F = self.arithmetic, self.arithmetic.F
        out = [F.zero()]*len(self.xs)
        def visit(node, value):
            remainder = A.divmod(value, node.product, node.inverse_cache)[1]
            if node.left is None:
                out[node.start] = remainder[0] if remainder else F.zero()
            else:
                visit(node.left, remainder)
                visit(node.right, remainder)
        visit(self.root, polynomial)
        return out

    def interpolate(self, values):
        if len(values) != len(self.xs):
            raise ValueError('one value per node required')
        if self.root is None:
            return []
        A, F = self.arithmetic, self.arithmetic.F
        if self._weights is None:
            denominators = self.evaluate(A.derivative(self.root.product))
            if any(F.is_zero(d) for d in denominators):
                raise ValueError('interpolation nodes must be distinct')
            self._weights = [F.inv(d) for d in denominators]
        def combine(node):
            if node.left is None:
                return trim(F,[F.mul(values[node.start], self._weights[node.start])])
            return A.add(A.mul(combine(node.left), node.right.product),
                         A.mul(combine(node.right), node.left.product))
        return combine(self.root)
