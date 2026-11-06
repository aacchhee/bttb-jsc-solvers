# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Shared. Reference PM-Basis, shifted-degree and polynomial-matrix operations used by mosaic terminals.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
def _shape(A):
    if not A or not A[0] or not A[0][0]:
        raise ValueError("a nonempty polynomial coefficient matrix is required")
    m, n = len(A[0]), len(A[0][0])
    if any(len(C) != m or any(len(row) != n for row in C) for C in A):
        raise ValueError("coefficient matrices must have one rectangular shape")
    return m, n


def _zeros(F, m, n):
    return [[F.zero() for _ in range(n)] for _ in range(m)]


def _trim(F, A):
    while len(A) > 1 and all(F.is_zero(x) for row in A[-1] for x in row):
        A.pop()
    return A


def polynomial_matrix_product(F, A, B, mul, limit=None, *, skip_zeros=False):
    """Classical entrywise polynomial product; optional coefficient truncation.

    ``mul`` returns the full product of two low-to-high coefficient lists.
    Its internal algorithm, arithmetic cost and exact field are caller owned.
    """
    m, inner = _shape(A)
    binner, n = _shape(B)
    if inner != binner:
        raise ValueError("incompatible polynomial matrix dimensions")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("positive coefficient limit required")
    length = len(A)+len(B)-1
    if limit is not None:
        length = min(length, limit)
    out = [_zeros(F, m, n) for _ in range(length)]
    if skip_zeros:

        
        ap = [[[C[i][t] for C in A] for t in range(inner)] for i in range(m)]
        bp = [[[C[t][j] for C in B] for j in range(n)] for t in range(inner)]
        az = [[all(F.is_zero(x) for x in entry) for entry in row] for row in ap]
        bz = [[all(F.is_zero(x) for x in entry) for entry in row] for row in bp]
    for i in range(m):
        for j in range(n):
            for t in range(inner):
                if skip_zeros and (az[i][t] or bz[t][j]):
                    continue
                product = (mul(ap[i][t], bp[t][j]) if skip_zeros else
                           mul([C[i][t] for C in A], [C[t][j] for C in B]))
                for k, x in enumerate(product[:length]):
                    out[k][i][j] = F.add(out[k][i][j], x)
    return _trim(F, out)


def shifted_row_degrees(F, A, shift):
    """Return (shifted degrees, rightmost shifted pivots, pivot degrees)."""
    m, n = _shape(A)
    if len(shift) != n:
        raise ValueError("one integer shift per column required")
    degrees, pivots, pivot_degrees = [], [], []
    for i in range(m):
        best = None
        for j in range(n):
            for k in reversed(range(len(A))):
                if not F.is_zero(A[k][i][j]):
                    candidate = (k+shift[j], j, k)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
                    break
        if best is None:
            raise ValueError("zero rows have no finite shifted row degree")
        degrees.append(best[0]); pivots.append(best[1]); pivot_degrees.append(best[2])
    return degrees, pivots, pivot_degrees


def _order_one(F, A, shift, *, skip_zeros=False):
    """Stable greedy row rank profile and kernel rows in original coordinates.

    Echelon auxiliary rows carry expressions in ORIGINAL independent rows;
    simply subtracting the printed LSP entries would be incorrect (algorithm K).
    Every inverse follows a zero-tested, nonzero pivot.
    """
    m, n = len(A), len(A[0])
    permutation = sorted(range(m), key=lambda i: (shift[i], i))
    constant, linear = _zeros(F, m, m), _zeros(F, m, m)
    echelon, profile, columns, pivot_values = [], [], [], []
    for i in permutation:
        v = list(A[i])
        expression = [F.zero() for _ in range(m)]
        expression[i] = F.one()
        for col, row, representation in echelon:
            factor = v[col]
            if skip_zeros and F.is_zero(factor):
                continue

            
            for j in range(n):
                v[j] = F.sub(v[j], F.mul(factor, row[j]))
            for j in range(m):
                expression[j] = F.sub(expression[j], F.mul(factor, representation[j]))
        col = next((j for j in range(n) if not F.is_zero(v[j])), None)
        if col is None:
            constant[i] = expression
        else:
            pivot = v[col]
            inv = F.inv(pivot)
            echelon.append((col, [F.mul(inv, x) for x in v],
                            [F.mul(inv, x) for x in expression]))
            profile.append(i); columns.append(col); pivot_values.append(pivot)
            linear[i][i] = F.one()
    basis = _trim(F, [constant, linear])
    delta = [1 if i in profile else 0 for i in range(m)]
    return {"basis": basis, "row_degree": [s+d for s, d in zip(shift, delta)],
            "pivot_degree": delta, "pivots": list(range(m)),
            "ledger": {"order": 1, "permutation": permutation,
                       "row_rank_profile": profile, "pivot_columns": columns,
                       "elimination_pivot_values": pivot_values,
                       "basis_determinant_exponent": len(profile),
                       "basis_determinant_coefficient": F.one()}}


def _constant_determinant(F, A, *, skip_zeros=False):
    """Small leaf-window determinant with exact zero-tested pivots."""
    M = [list(row) for row in A]
    n = len(M)
    value = F.one()
    for j in range(n):
        pivot = next((i for i in range(j, n) if not F.is_zero(M[i][j])), None)
        if pivot is None:
            return F.zero()
        if pivot != j:
            M[j], M[pivot] = M[pivot], M[j]
            value = F.neg(value)
        value = F.mul(value, M[j][j])
        inv = F.inv(M[j][j])
        for i in range(j+1, n):
            if skip_zeros and F.is_zero(M[i][j]):
                continue
            factor = F.mul(M[i][j], inv)
            for k in range(j+1, n):
                M[i][k] = F.sub(M[i][k], F.mul(factor, M[j][k]))
    return value


def _decorate_window_ledger(F, node, *, skip_zeros=False):
    """algorithm Lemma M, transposed to the row approximant convention."""
    if node["order"] == 1:
        active = [j for j, e in enumerate(node["offsets"]) if e == 0]
        pivots = [i for i, p in enumerate(node["profile"]) if p == 1]
        onto = len(pivots) == len(active)
        node["all_onto"] = onto
        node["active_constraints"] = active
        node["window_pivot_indices"] = pivots
        node["window_determinant"] = (_constant_determinant(F,
            [[node["input_constant"][i][j] for j in active] for i in pivots],
            skip_zeros=skip_zeros)
            if onto else F.zero())
        node["domain_sign_exponent"] = node["codomain_sign_exponent"] = 0
    else:
        d, d1 = node["order"], node["order_left"]
        p1, p2 = node["left_profile"], node["right_profile"]
        low = [max(d1-e, 0) for e in node["offsets"]]
        high = [max(d-max(e, d1), 0) for e in node["offsets"]]
        domain = sum(p2[i]*p1[j] for i in range(len(p1)) for j in range(i+1, len(p1)))
        codomain = sum(high[i]*low[j] for i in range(len(low)) for j in range(i+1, len(low)))
        value = F.mul(node["left"]["window_determinant"], node["right"]["window_determinant"])
        node["window_determinant"] = F.neg(value) if (domain+codomain)%2 else value
        node["all_onto"] = node["left"]["all_onto"] and node["right"]["all_onto"]
        node["domain_sign_exponent"], node["codomain_sign_exponent"] = domain, codomain
    node["window_ledger_theorem"] = "algorithm Lemma M; input effective window, not basis determinant"


def pm_basis(F, coefficients, order, shift, mul, *, offsets=None, ledger_hook=None,
             skip_zeros=False):
    """Return a shifted approximant basis for p*coefficients = 0 mod x^order.

    Result keys: ``basis`` (coefficient first), ``shift`` (original input),
    ``row_degree`` (shifted output degrees), ``pivot_degree`` (degree increments),
    ``pivots`` (original row/column indices), and ``ledger`` (recursive trace).
    Each ledger describes the BASIS determinant x^sum(pivot_degree), coefficient
    one; it must not be confused with an input/window/Toeplitz determinant.

    Optional ``offsets`` contains one known divisibility lower bound per input
    column. The exact algorithm Lemma M effective-window determinant is returned
    in ledger.window_determinant, with ledger.all_onto false on failed leaf
    cardinality. This scalar is not the determinant of a supplied target matrix;
    a target reduction must supply its expected profile and external sign.
    ``ledger_hook`` receives
    a node dictionary after its children finish and can attach such a ledger.
    ``skip_zeros=True`` optionally skips guard-certified zero factors; its
    default preserves the reference arithmetic path.
    Nodes expose ``input_constant`` on leaves, ``left_profile/right_profile`` on
    splits, and offsets propagated by max(e-order_left,0). No hook is needed for
    PM-Basis itself. The hook's arithmetic cost is separate from the core bound.
    """
    m, n = _shape(coefficients)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("positive integer approximant order required")
    shift = list(shift)
    if len(shift) != m or any(not isinstance(s, int) or isinstance(s, bool) for s in shift):
        raise ValueError("one integer shift per input row required")
    if not callable(mul):
        raise ValueError("scalar polynomial multiplier required")
    if offsets is None:
        offsets = [0]*n
    else:
        offsets = list(offsets)
    if len(offsets) != n or any(not isinstance(e, int) or isinstance(e, bool) or e < 0 for e in offsets):
        raise ValueError("one nonnegative integer offset per input column required")

    for j, e in enumerate(offsets):
        if any(not F.is_zero(C[i][j]) for C in coefficients[:min(e, order)] for i in range(m)):
            raise ValueError("column does not satisfy its declared divisibility offset")

    def recurse(A, d, s, e):
        if d == 1:
            out = _order_one(F, A[0], s, skip_zeros=skip_zeros)
            node = out["ledger"]
            node["input_constant"] = [list(row) for row in A[0]]
        else:
            d1, d2 = d//2, d-d//2
            first = recurse(A[:d1], d1, s, e)
            product = polynomial_matrix_product(F, first["basis"], A, mul, limit=d,
                                                skip_zeros=skip_zeros)
            residual = product[d1:d]
            residual += [_zeros(F, m, n) for _ in range(d2-len(residual))]
            second = recurse(residual, d2, first["row_degree"],
                             [max(x-d1, 0) for x in e])
            B = polynomial_matrix_product(F, second["basis"], first["basis"], mul,
                                          skip_zeros=skip_zeros)
            delta = [a+b for a, b in zip(first["pivot_degree"], second["pivot_degree"])]
            out = {"basis": B, "row_degree": [x+y for x, y in zip(s, delta)],
                   "pivot_degree": delta, "pivots": list(range(m))}
            node = {"order": d, "order_left": d1, "order_right": d2,
                    "left_profile": list(first["pivot_degree"]),
                    "right_profile": list(second["pivot_degree"]),
                    "left": first["ledger"], "right": second["ledger"],
                    "basis_determinant_exponent": sum(delta),
                    "basis_determinant_coefficient": F.one()}
            out["ledger"] = node
        node["shift"] = list(s)
        node["profile"] = list(out["pivot_degree"])
        node["offsets"] = list(e)
        node["rows"], node["columns"] = m, n
        _decorate_window_ledger(F, node, skip_zeros=skip_zeros)
        if ledger_hook is not None:
            ledger_hook(node)
        return out

    A = [[list(row) for row in C] for C in coefficients[:order]]
    result = recurse(A, order, shift, offsets)
    result["shift"] = shift
    result["degree_bound"] = order
    result["algorithm"] = "balanced classical row PM-Basis (JNV Algorithm 2)"
    return result
