# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
LEAF_CUTOFF = 64
QUOTIENT_LEAF_EXPONENT = 2


def make_multiplier(F, cutoff=LEAF_CUTOFF):
    if cutoff != LEAF_CUTOFF:
        raise ValueError("the CK scalar leaf cutoff is fixed at 64")
    zero, one = F.zero(), F.one()
    two = F.add(one, one)
    s = 3 if F.is_zero(two) else 2
    scalar = F.add(two, one) if s == 3 else two
    if F.is_zero(scalar):
        raise ArithmeticError("CK transform radix must be invertible")
    inv_s = F.inv(scalar)
    stats = {"kind":"ck", "radix":s, "calls":0, "leaf_calls":0,
             "quotient_calls":0, "quotient_leaf_calls":0,
             "recursive_nodes":0, "pointwise_products":0, "fft_calls":0,
             "largest_input":0, "largest_quotient_dimension":0,
             "max_depth":0, "input_lengths":{}, "quotient_exponents":{},
             "recursion_shapes":{}, "scalar_leaf_cutoff":LEAF_CUTOFF,
             "quotient_leaf_exponent":QUOTIENT_LEAF_EXPONENT}

    def bump(hist, key):
        key = str(key)
        hist[key] = hist.get(key, 0)+1

    def leaf(a, b):
        out = [zero]*(len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i+j] = F.add(out[i+j], F.mul(x, y))
        return out

    def reduce(values, Q):
        t = s**(Q-1)
        d = (s-1)*t
        out = list(values)+[zero]*max(0, d-len(values))
        for k in range(len(out)-1, d-1, -1):
            value = out[k]
            for j in range(s-1):
                pos = k-d+j*t
                out[pos] = F.sub(out[pos], value)
        return out[:d]

    def shift(values, exponent, Q):
        period = s**Q
        exponent %= period
        out = [zero]*period
        for i, value in enumerate(values):
            out[(i+exponent)%period] = value
        return reduce(out, Q)

    def add(u, v):
        return [F.add(x, y) for x, y in zip(u, v)]

    def sub(u, v):
        return [F.sub(x, y) for x, y in zip(u, v)]

    def scale(u, c):
        return [F.mul(x, c) for x in u]

    def fft(values, root_exponent, Q):
        stats["fft_calls"] += 1
        length = len(values)
        if length == 1:
            return [list(values[0])]
        m = length//s
        parts = [fft(values[j::s], s*root_exponent, Q) for j in range(s)]
        d = (s-1)*s**(Q-1)
        out = []
        for k in range(length):
            acc = [zero]*d
            for j in range(s):
                acc = add(acc, shift(parts[j][k%m], root_exponent*j*k, Q))
            out.append(acc)
        return out

    def quotient(a, b, Q, depth=0):
        stats["quotient_calls"] += 1
        bump(stats["quotient_exponents"], Q)
        stats["max_depth"] = max(stats["max_depth"], depth)
        d = (s-1)*s**(Q-1)
        stats["largest_quotient_dimension"] = max(stats["largest_quotient_dimension"], d)
        if len(a) != d or len(b) != d:
            raise AssertionError("quotient product requires canonical coefficient vectors")
        if Q <= QUOTIENT_LEAF_EXPONENT:
            stats["quotient_leaf_calls"] += 1
            return reduce(leaf(a, b), Q)
        q = Q//2+1
        r = Q-q
        n, m = s**r, s**q
        dq = (s-1)*s**(q-1)
        if q >= Q or q < r+1 or n*dq != d:
            raise AssertionError("CK split violates quotient descent")
        stats["recursive_nodes"] += 1
        stats["pointwise_products"] += 2*n
        bump(stats["recursion_shapes"], "%d:%d:%d:%d" % (Q,q,r,2*n))
        A = [[a[i*n+j] for i in range(dq)] for j in range(n)]
        B = [[b[i*n+j] for i in range(dq)] for j in range(n)]
        inverse_n = one
        for _ in range(r):
            inverse_n = F.mul(inverse_n, inv_s)
        root = m//n
        eta = m//(s*n)

        def cyclic(U, V):
            Uh, Vh = fft(U, root, q), fft(V, root, q)
            products = [quotient(u,v,q,depth+1) for u,v in zip(Uh,Vh)]
            return [scale(u, inverse_n) for u in fft(products, -root, q)]

        D = cyclic(A, B)
        twisted_A = [shift(u, eta*j, q) for j,u in enumerate(A)]
        twisted_B = [shift(u, eta*j, q) for j,u in enumerate(B)]
        E = [shift(u, -eta*j, q) for j,u in enumerate(cyclic(twisted_A,twisted_B))]
        out = [zero]*d
        for j in range(n):
            difference = sub(D[j], E[j])

            
            if s == 3:
                difference = sub(difference, shift(difference, 2*m//s, q))
            high = scale(difference, inv_s)
            low = sub(D[j], high)
            value = add(low, shift(high, 1, q))
            for i in range(dq):
                out[i*n+j] = value[i]
        return out

    def multiply(a, b):
        if not a or not b:
            return []
        a, b = list(a), list(b)
        stats["calls"] += 1
        bump(stats["input_lengths"], "%d:%d" % (len(a),len(b)))
        stats["largest_input"] = max(stats["largest_input"],len(a),len(b))
        length = len(a)+len(b)-1
        if max(len(a),len(b)) <= LEAF_CUTOFF:
            stats["leaf_calls"] += 1
            return leaf(a,b)
        Q = 1
        while (s-1)*s**(Q-1) < length:
            Q += 1
        d = (s-1)*s**(Q-1)
        return quotient(a+[zero]*(d-len(a)), b+[zero]*(d-len(b)), Q)[:length]

    multiply.stats = stats
    multiply.primitive_kind = "ck"
    multiply.operation_bound = "O(d log(2d) loglog(4d))"
    return multiply


def multiply(F, a, b):
    """Return the complete ordinary product, preserving zero tails and lengths."""
    return make_multiplier(F)(a,b)

