# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
LEAF_CUTOFF = 64
NATIVE_ACCELERATOR_MAX = 4096


def make_multiplier(F, cutoff=LEAF_CUTOFF):
    """Return a reusable exact full-convolution callable for the adapter F.

    The cutoff is fixed: allowing a caller to grow it with input size would
    invalidate the uniform recursive multiplication route.
    """
    if cutoff != LEAF_CUTOFF:
        raise ValueError("the all-field multiplication leaf cutoff is fixed at 64")
    zero, one = F.zero(), F.one()
    two = F.add(one, one)
    radix = 3 if F.is_zero(two) else 2
    radix_element = F.add(two, one) if radix == 3 else two
    inv_radix = F.inv(radix_element)
    stats = {"calls":0,"leaf_calls":0,"synthetic_fft_calls":0,"native_products":0,
             "largest_input":0,"largest_synthetic_dimension":0,"radix":radix}

    def leaf(a,b):

        

        native_leaf = getattr(F, 'polynomial_leaf', None)
        if native_leaf is not None:
            return native_leaf(a, b)
        result = [zero]*(len(a)+len(b)-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b):
                result[i+j] = F.add(result[i+j],F.mul(x,y))
        return result

    def multiply(a,b):
        if not a or not b:
            return []
        stats["calls"] += 1
        n = max(len(a),len(b))
        stats["largest_input"] = max(stats["largest_input"],n)
        answer_length = len(a)+len(b)-1
        if n <= LEAF_CUTOFF:
            stats["leaf_calls"] += 1
            return leaf(a,b)
        native = getattr(F, 'polynomial_native', None)
        if native is not None and n <= NATIVE_ACCELERATOR_MAX:

            
            product = native(a,b)
            if product is not None:
                stats["native_products"] += 1
                return product
        stats["synthetic_fft_calls"] += 1
        block = 1
        while block*block < n:
            block *= radix
        dimension = 2*block
        period = (4 if radix == 2 else 3)*block
        h = (n+block-1)//block
        length,levels = 1,0
        while length < 2*h-1:
            length *= radix
            levels += 1
        if period % length or dimension >= n:
            raise AssertionError("synthetic transform parameters violate descent")
        stats["largest_synthetic_dimension"] = max(stats["largest_synthetic_dimension"],dimension)

        def reduce(coefficients):
            coefficients = list(coefficients)
            coefficients += [zero]*max(0,dimension-len(coefficients))
            for t in range(len(coefficients)-1,dimension-1,-1):
                value = coefficients[t]
                coefficients[t-dimension] = F.sub(coefficients[t-dimension],value)
                if radix == 3:
                    coefficients[t-block] = F.sub(coefficients[t-block],value)
            return coefficients[:dimension]

        def add(u,v):
            return [F.add(x,y) for x,y in zip(u,v)]

        def shift(u,exponent):
            exponent %= period
            coefficients = [zero]*period
            for i,value in enumerate(u):
                coefficients[(i+exponent)%period] = value
            return reduce(coefficients)

        def fft(values,root_exponent):
            size = len(values)
            if size == 1:
                return [list(values[0])]
            m = size//radix
            parts = [fft(values[r::radix],radix*root_exponent) for r in range(radix)]
            transformed = []
            for k in range(size):
                acc = [zero]*dimension
                for r in range(radix):
                    acc = add(acc,shift(parts[r][k%m],root_exponent*r*k))
                transformed.append(acc)
            return transformed

        def pack(values):
            packed = []
            for k in range(h):
                u = list(values[k*block:(k+1)*block])
                packed.append(u+[zero]*(dimension-len(u)))
            packed += [[zero]*dimension for _ in range(length-h)]
            return packed

        exponent = period//length
        ahat,bhat = fft(pack(a),exponent),fft(pack(b),exponent)
        chat = [reduce(multiply(u,v)) for u,v in zip(ahat,bhat)]
        blocks = fft(chat,-exponent)
        inverse_length = one
        for _ in range(levels):
            inverse_length = F.mul(inverse_length,inv_radix)
        result = [zero]*answer_length
        for k in range(2*h-1):
            for j in range(dimension):
                t = k*block+j
                if t < answer_length:
                    result[t] = F.add(result[t],F.mul(blocks[k][j],inverse_length))
        return result

    stats["kind"] = "allfield"
    multiply.stats = stats
    multiply.primitive_kind = "allfield"
    multiply.operation_bound = "O(n log^5(n+2))"
    return multiply


def multiply(F,a,b):
    """Return all len(a)+len(b)-1 coefficients, or [] for an empty factor."""
    return make_multiplier(F)(a,b)


def multiply_rectangles(F,a,b,multiplier=None):
    """No-carry Kronecker convolution of rectangular coefficient matrices."""
    if not a or not b or not a[0] or not b[0]:
        return []
    ay,ax,by,bx = len(a),len(a[0]),len(b),len(b[0])
    if any(len(row)!=ax for row in a) or any(len(row)!=bx for row in b):
        raise ValueError("rectangular coefficient matrices required")
    stride,height = ax+bx-1,ay+by-1
    zero = F.zero()
    def flatten(rows):
        coefficients = [zero]*(stride*(len(rows)-1)+len(rows[0]))
        for y,row in enumerate(rows):
            coefficients[y*stride:y*stride+len(row)] = row
        return coefficients
    result = (multiplier or make_multiplier(F))(flatten(a),flatten(b))
    result += [zero]*(stride*height-len(result))
    return [result[y*stride:(y+1)*stride] for y in range(height)]
