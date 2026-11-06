# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip.approximant import _shape, _zeros, _trim
from . import dense_backend
from bttb_solvers._internal.halfstrip.allfield_poly import make_multiplier as _make_multiplier


class _FieldOperations:
    """Expose only guarded field operations, hiding optional native hooks."""
    def __init__(self, field):
        for name in ('zero', 'one', 'add', 'sub', 'neg', 'mul', 'inv', 'is_zero'):
            setattr(self, name, getattr(field, name))


def make_multiplier(F):
    """Default reference recursion with the original metadata-hiding proxy.

    Explicit FLINT construction may replace prime polynomial products. That
    external kernel inherits no uniform field-operation bound. Unknown/ring
    adapters retain the exact original proxy path and guarded arithmetic.
    """
    from .flint_backend import enabled_prime, product as flint_product
    owner = getattr(F.mul, '__self__', None)
    if enabled_prime(owner) is None:
        return _make_multiplier(_FieldOperations(F))
    original = _make_multiplier(_FieldOperations(F))
    def multiply(a,b):
        result = flint_product(owner,a,b)
        if result is None:
            return original(a,b)
        if a and b:
            original.stats['calls'] += 1
            original.stats['largest_input'] = max(original.stats['largest_input'],len(a),len(b))
            original.stats['compiled_calls'] = original.stats.get('compiled_calls',0)+1
        return result
    multiply.stats = original.stats
    multiply.primitive_kind = 'explicit-flint-nmod_poly-with-context-fallback'
    multiply.operation_bound = 'external FLINT products; no inherited uniform field-operation bound'
    return multiply


def _count(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _inverse(F, value):
    if F.is_zero(value):
        raise ZeroDivisionError('uniform terminal denominator is zero')
    return F.inv(value)


def _power(F, value, exponent):
    out = F.one()
    while exponent:
        if exponent & 1:
            out = F.mul(out, value)
        exponent //= 2
        if exponent:
            value = F.mul(value, value)
    return out


def _classical(A,B,rows,inner,cols,zero,add,multiply,stats,*,leaf=False,domain='constant',empty_skips=None):

    

    if domain == 'constant' and rows and inner and cols:
        from .flint_backend import matmul as flint_matmul
        owner = getattr(multiply, '__self__', None)
        accelerated = flint_matmul(owner,A,B)
        if accelerated is not None:
            dense_backend.record('flint_nmod_mat_product_calls')
            dense_backend.record('flint_compiled_scalar_counts_unavailable')
            if stats is not None:
                stats['flint_compiled_scalar_counts_unavailable'] = True
                _count(stats,'flint_nmod_mat_product_calls')
            return accelerated
    out=[[zero() for _ in range(cols)] for _ in range(rows)]
    prefix='strassen_leaf' if leaf else 'classical'
    completed=0
    try:
        for i in range(rows):
            for k in range(inner):
                for j in range(cols):

                    

                    if empty_skips is not None and (not A[i][k] or not B[k][j]):
                        empty_skips[0]+=1
                        continue
                    term=multiply(A[i][k],B[k][j])
                    completed+=1
                    out[i][j]=add(out[i][j],term)
    finally:

        dense_backend.record(prefix+'_entry_products',completed)
        dense_backend.record(prefix+'_'+domain+'_entry_products',completed)
        _count(stats,prefix+'_entry_products',completed)
        _count(stats,'copy_cells',rows*cols)
    return out


def _square(A, B, add, sub, multiply, stats, zero=None, domain='constant'):
    backend,leaf=dense_backend.selection()
    if backend!='strassen':
        raise RuntimeError('direct Strassen square bypass in classical dense scope')
    n = len(A)
    _count(stats, 'strassen_nodes')
    dense_backend.record('strassen_nodes')
    if n<=leaf:
        dense_backend.record('strassen_leaf_calls')
        dense_backend.record('strassen_leaf_size_'+str(n))
        _count(stats,'scalar_products',n*n*n)
        if n==1:

            
            value=multiply(A[0][0],B[0][0])
            dense_backend.record('strassen_leaf_entry_products')
            dense_backend.record('strassen_leaf_'+domain+'_entry_products')
            return [[value]]
        if zero is None:
            raise ValueError('a fixed Strassen leaf requires its zero factory')
        return _classical(A,B,n,n,n,zero,add,multiply,stats,leaf=True,domain=domain)
    dense_backend.record('strassen_recursive_nodes')
    if n==2 and leaf==1:

        

        

        a,b=A[0]
        c,d=A[1]
        e,f=B[0]
        g,hb=B[1]
        def scalar(x,y):
            _count(stats,'strassen_nodes')
            dense_backend.record('strassen_nodes')
            dense_backend.record('strassen_leaf_calls')
            dense_backend.record('strassen_leaf_size_1')
            _count(stats,'scalar_products')
            value=multiply(x,y)
            dense_backend.record('strassen_leaf_entry_products')
            dense_backend.record('strassen_leaf_'+domain+'_entry_products')
            return value
        p1=scalar(add(a,d),add(e,hb))
        p2=scalar(add(c,d),e)
        p3=scalar(a,sub(f,hb))
        p4=scalar(d,sub(g,e))
        p5=scalar(add(a,b),hb)
        p6=scalar(sub(c,a),add(e,f))
        p7=scalar(sub(b,d),add(g,hb))
        x=add(sub(add(p1,p4),p5),p7)
        y=add(p3,p5)
        z=add(p2,p4)
        w=add(add(sub(p1,p2),p3),p6)
        _count(stats,'copy_cells',4)
        return [[x,y],[z,w]]
    h = n//2
    def split(X):
        _count(stats, 'copy_cells', n*n)
        return ([row[:h] for row in X[:h]], [row[h:] for row in X[:h]],
                [row[:h] for row in X[h:]], [row[h:] for row in X[h:]])
    def op(X, Y, f):
        _count(stats, 'copy_cells', h*h)
        return [[f(x,y) for x,y in zip(u,v)] for u,v in zip(X,Y)]
    a,b,c,d = split(A)
    e,f,g,hb = split(B)
    def rec(X,Y):
        return _square(X,Y,add,sub,multiply,stats,zero,domain)
    p1 = rec(op(a,d,add),op(e,hb,add))
    p2 = rec(op(c,d,add),e)
    p3 = rec(a,op(f,hb,sub))
    p4 = rec(d,op(g,e,sub))
    p5 = rec(op(a,b,add),hb)
    p6 = rec(op(c,a,sub),op(e,f,add))
    p7 = rec(op(b,d,sub),op(g,hb,add))
    x = op(op(op(p1,p4,add),p5,sub),p7,add)
    y = op(p3,p5,add)
    z = op(p2,p4,add)
    w = op(op(op(p1,p2,sub),p3,add),p6,add)
    _count(stats, 'copy_cells', n*n)
    return [u+v for u,v in zip(x,y)]+[u+v for u,v in zip(z,w)]


def _rectangular(A, B, rows, inner, cols, zero, add, sub, multiply, stats, *, domain='constant',empty_skips=None):
    """Classical rectangular loops or reference fixed-leaf Strassen blocking."""
    _count(stats, 'product_calls')
    backend,_leaf=dense_backend.selection()
    dense_backend.record('dense_product_calls')
    dense_backend.record(domain+'_matrix_product_calls')
    if backend=='classical':
        dense_backend.record('classical_product_calls')
        return _classical(A,B,rows,inner,cols,zero,add,multiply,stats,domain=domain,empty_skips=empty_skips)
    _count(stats, 'copy_cells', rows*cols)
    out = [[zero() for _ in range(cols)] for _ in range(rows)]
    if not rows or not inner or not cols:
        return out
    block = min(rows,inner,cols)
    padded = 1
    while padded < block:
        padded *= 2
    for i in range(0,rows,block):
        nr = min(block,rows-i)
        for j in range(0,cols,block):
            nc = min(block,cols-j)
            for k in range(0,inner,block):
                nk = min(block,inner-k)

                

                if nr==nk==padded and i==k==0 and rows==inner==padded:
                    X=A
                else:
                    _count(stats,'copy_cells',padded*padded)
                    X = [[A[i+u][k+v] if u<nr and v<nk else zero()
                          for v in range(padded)] for u in range(padded)]
                if nk==nc==padded and k==j==0 and inner==cols==padded:
                    Y=B
                else:
                    _count(stats,'copy_cells',padded*padded)
                    Y = [[B[k+u][j+v] if u<nk and v<nc else zero()
                          for v in range(padded)] for u in range(padded)]
                Z = _square(X,Y,add,sub,multiply,stats,zero,domain)
                for u in range(nr):
                    for v in range(nc):
                        out[i+u][j+v] = add(out[i+u][j+v], Z[u][v])
    return out


def matrix_product(F, A, B, *, stats=None):
    rows, inner = len(A), len(B)
    cols = len(B[0]) if B else 0
    if any(len(row)!=inner for row in A) or any(len(row)!=cols for row in B):
        raise ValueError('incompatible rectangular matrices')
    return _rectangular(A,B,rows,inner,cols,F.zero,F.add,F.sub,F.mul,stats)


def polynomial_matrix_product(F, A, B, mul, limit=None, *, skip_zeros=False,
                              stats=None):
    """Selected dense product; optional guarded entry policy, default off.

    skip_zeros keeps its historical unused meaning. Classical/off trims owned
    capped input entries once and skips empty factors. Nonempty products use
    the supplied scalar callback; Strassen/off retains its original schedule.
    """
    rows, inner = _shape(A)
    binner, cols = _shape(B)
    if binner != inner:
        raise ValueError('incompatible polynomial matrix dimensions')
    if limit is not None and (type(limit) is not int or limit<1):
        raise ValueError('positive coefficient limit required')
    length = len(A)+len(B)-1
    if limit is not None:
        length = min(length,limit)
    policy=dense_backend.entry_policy()
    classical_cleanup=(dense_backend.selection()[0]=='classical' and policy=='off')
    cleanup_skipped=cleanup_started=cleanup_completed=0

    def note(key,amount=1):
        key='poly_entry_'+key
        dense_backend.record(key,amount)
        _count(stats,key,amount)

    def maximum(key,value):
        key='poly_entry_'+key
        dense_backend.record_max(key,value)
        if stats is not None:
            stats[key]=max(stats.get(key,0),value)

    def effective_length(a,origin):

        
        remaining=len(a);attempted=completed=proved_zero=0
        note('normalization_calls')
        try:
            while remaining:
                attempted+=1
                zero=F.is_zero(a[remaining-1])
                completed+=1
                if not zero:break
                remaining-=1;proved_zero+=1
        finally:
            note('guard_attempts',attempted)
            note('guard_completions',completed)
            note('tails_proved_zero',proved_zero)
            note(origin+'_guard_attempts',attempted)
        note('effective_slots',remaining)
        note(origin+'_effective_slots',remaining)
        if not remaining:
            note('zero_entries_observed')
            note(origin+'_zero_entries_observed')
        return remaining

    def normalized(a,origin):
        effective=effective_length(a,origin)
        if policy=='normalize':
            removed=len(a)-effective
            del a[effective:]
            note('tail_slots_removed',removed)
        return a

    def op(a,b,operation):
        size = min(length,max(len(a),len(b)))
        _count(stats,'copy_cells',size)
        if policy=='off':

            return [operation(a[i] if i<len(a) else F.zero(),
                              b[i] if i<len(b) else F.zero()) for i in range(size)]
        note('op_calls')
        result=[]
        try:
            for i in range(size):
                result.append(operation(a[i] if i<len(a) else F.zero(),
                                        b[i] if i<len(b) else F.zero()))
        finally:
            note('op_allocated_slots',len(result))
        result=normalized(result,'op')
        note('op_retained_slots',len(result))
        return result

    def multiply(a,b):
        nonlocal cleanup_skipped,cleanup_started,cleanup_completed
        if policy=='off':
            if classical_cleanup:
                if not a or not b:
                    cleanup_skipped+=1
                    return []
                cleanup_started+=1

            
            result=mul(a,b)[:length]
            if classical_cleanup:cleanup_completed+=1
            _count(stats,'copy_cells',len(result))
            return result
        note('multiply_invocations')
        note('operand_slots',len(a)+len(b))
        maximum('operand_max_length',max(len(a),len(b)))

        
        ea,eb=((len(a),len(b)) if policy=='normalize' else
               (effective_length(a,'multiply'),effective_length(b,'multiply')))
        note('effective_operand_slots',ea+eb)
        if not ea or not eb:
            note('zero_factor_pairs_observed')
            if policy=='normalize':
                note('zero_products_skipped')
                return []
        note('scalar_calls_started')
        note('scalar_forwarded_slots',len(a)+len(b))
        largest=max(len(a),len(b))
        note('scalar_forwarded_fixed64' if largest<=64 else 'scalar_forwarded_above64')
        bound=0 if not largest else 1
        while bound<largest:bound*=2
        note('scalar_length_bin_'+str(bound))
        raw=mul(a,b)
        note('scalar_calls_completed')
        note('scalar_full_output_slots',len(raw))
        result=raw[:length]
        note('scalar_limited_output_slots',len(result))
        _count(stats,'copy_cells',len(result))
        result=normalized(result,'product')
        note('scalar_retained_output_slots',len(result))
        return result

    if classical_cleanup:

        

        attempted=completed=removed=zero_entries=0
        def lift_classical(coefficients,r,c):
            nonlocal attempted,completed,removed,zero_entries
            cap=min(len(coefficients),length)
            result=[]
            for i in range(r):
                row=[]
                for j in range(c):
                    entry=[coefficients[d][i][j] for d in range(cap)]
                    while entry:
                        attempted+=1
                        is_zero=F.is_zero(entry[-1]);completed+=1
                        if not is_zero:break
                        entry.pop();removed+=1
                    zero_entries+=not entry
                    row.append(entry)
                result.append(row)
            return result
        try:
            ap=lift_classical(A,rows,inner)
            bp=lift_classical(B,inner,cols)
        finally:
            for key,value in [('guard_attempts',attempted),('guard_completions',completed),
                              ('input_tail_slots_removed',removed),('zero_input_entries',zero_entries)]:
                dense_backend.record('classical_poly_cleanup_'+key,value)
                _count(stats,'classical_poly_cleanup_'+key,value)
        _count(stats,'copy_cells',rows*inner*min(len(A),length)+inner*cols*min(len(B),length))
        zero=lambda:[]
    elif policy=='off':
        ap = [[[C[i][j] for C in A] for j in range(inner)] for i in range(rows)]
        bp = [[[C[i][j] for C in B] for j in range(cols)] for i in range(inner)]
        _count(stats,'copy_cells',rows*inner*len(A)+inner*cols*len(B))
        zero=lambda:[F.zero()]
    else:
        note('matrix_product_calls')
        note('input_entries',rows*inner+inner*cols)
        note('input_supplied_slots',rows*inner*len(A)+inner*cols*len(B))
        acap,bcap=min(len(A),length),min(len(B),length)
        note('input_ignored_slots',rows*inner*(len(A)-acap)+inner*cols*(len(B)-bcap))
        def lift(coefficients,r,c,cap):
            result=[]
            for i in range(r):
                row=[]
                for j in range(c):
                    entry=[coefficients[d][i][j] for d in range(cap)]
                    note('input_acquired_slots',len(entry))
                    _count(stats,'copy_cells',len(entry))
                    row.append(normalized(entry,'input'))
                result.append(row)
            return result
        ap,bp=lift(A,rows,inner,acap),lift(B,inner,cols,bcap)
        zero=(lambda:[]) if policy=='normalize' else (lambda:[F.zero()])
    empty_skips=[0] if classical_cleanup else None
    try:
        product = _rectangular(ap,bp,rows,inner,cols,zero,
                              lambda a,b:op(a,b,F.add),lambda a,b:op(a,b,F.sub),
                              multiply,stats,domain='polynomial',empty_skips=empty_skips)
    finally:
        if classical_cleanup:
            cleanup_skipped+=empty_skips[0]
            dense_backend.record('classical_poly_cleanup_empty_accumulations_skipped',empty_skips[0])
            _count(stats,'classical_poly_cleanup_empty_accumulations_skipped',empty_skips[0])
            for key,value in [('empty_factor_products_skipped',cleanup_skipped),
                              ('scalar_calls_started',cleanup_started),('scalar_calls_completed',cleanup_completed)]:
                dense_backend.record('classical_poly_cleanup_'+key,value)
                _count(stats,'classical_poly_cleanup_'+key,value)
    out = [_zeros(F,rows,cols) for _ in range(length)]
    for i in range(rows):
        for j in range(cols):
            for k,x in enumerate(product[i][j]):
                out[k][i][j] = x
    _count(stats,'copy_cells',length*rows*cols)
    return _trim(F,out)


def gauss_jordan(F, A, k=0, d0=None, *, stats=None):
    """Corrected reference GJ, U=dI with only rank exceptional columns stored.

    columns is n by rank and contains the ACTUAL columns U[:,k:k+rank].
    permutation is a row gather map, or None for the identity. eta=det(U),
    epsilon=sgn(P) are the proved determinant augmentation.
    """
    if d0 is None:
        d0 = F.one()
    n = len(A)
    width = len(A[0]) if n else 0
    if not 0<=k<=n or any(len(row)!=width for row in A):
        raise ValueError('invalid GJ dimensions')
    _count(stats,'gj_nodes')
    nonzero = any(not F.is_zero(A[i][j])
                  for i in range(k,n) for j in range(width))
    if not nonzero:
        return {'columns':[[] for _ in range(n)],'permutation':None,
                'rank':0,'start':k,'d':d0,'eta':_power(F,d0,n),
                'epsilon':F.one()}
    if width == 1:
        pivot = next(i for i in range(k,n) if not F.is_zero(A[i][0]))
        p = list(range(n))
        p[k],p[pivot] = p[pivot],p[k]
        d = A[pivot][0]
        W = [[F.neg(A[p[i]][0])] for i in range(n)]
        W[k][0] = d0
        _count(stats,'copy_cells',2*n)
        return {'columns':W,'permutation':p,'rank':1,'start':k,'d':d,
                'eta':F.mul(_power(F,d,n-1),d0),
                'epsilon':F.neg(F.one()) if pivot!=k else F.one()}
    left = width//2
    first = gauss_jordan(F,[row[:left] for row in A],k,d0,stats=stats)
    r1,d1,W1 = first['rank'],first['d'],first['columns']
    p1 = first['permutation']
    X = [A[p1[i] if p1 is not None else i][left:] for i in range(n)]
    Z = (_rectangular(W1,X[k:k+r1],n,r1,width-left,
                      F.zero,F.add,F.sub,F.mul,stats))
    inv0 = _inverse(F,d0)
    A2 = [[F.mul(inv0, Z[i][j] if k<=i<k+r1 else
                          F.add(F.mul(d1,X[i][j]),Z[i][j]))
           for j in range(width-left)] for i in range(n)]
    second = gauss_jordan(F,A2,k+r1,d1,stats=stats)
    r2,d,W2 = second['rank'],second['d'],second['columns']
    p2 = second['permutation']
    Y = [list(W1[p2[i] if p2 is not None else i]) for i in range(n)]
    for j in range(r1):
        Y[k+j][j] = F.sub(Y[k+j][j],d1)
    E = [list(row) for row in W2]
    for j in range(r2):
        E[k+r1+j][j] = F.sub(E[k+r1+j][j],d)
    EY = _rectangular(E,Y[k+r1:k+r1+r2],n,r2,r1,
                      F.zero,F.add,F.sub,F.mul,stats)
    inv1 = _inverse(F,d1)
    W = [[F.mul(inv1,F.add(F.mul(d,Y[i][j]),EY[i][j]))
          for j in range(r1)] + list(W2[i]) for i in range(n)]
    for j in range(r1):
        W[k+j][j] = F.add(W[k+j][j],d)
    if p1 is None and p2 is None:
        permutation = None
    else:
        permutation = [p1[p2[i] if p2 is not None else i]
                       if p1 is not None else p2[i] for i in range(n)]
    eta = F.mul(F.mul(second['eta'],first['eta']),_power(F,inv1,n))
    _count(stats,'copy_cells',n*(width-left+3*r1+3*r2+1))
    return {'columns':W,'permutation':permutation,'rank':r1+r2,'start':k,
            'd':d,'eta':eta,'epsilon':F.mul(second['epsilon'],first['epsilon'])}


def constant_determinant(F, A, *, stats=None):
    _count(stats,'determinant_calls')
    n = len(A)
    if any(len(row)!=n for row in A):
        raise ValueError('square determinant required')
    if n == 0:
        return F.one()
    result = gauss_jordan(F,A,stats=stats)
    if result['rank'] != n:
        return F.zero()
    denominator = F.mul(result['eta'],result['epsilon'])
    return F.mul(_power(F,result['d'],n),_inverse(F,denominator))


def order_one(F, A, shift, *, stats=None, skip_zeros=False):
    """Identical original-row leaf via reference Proposition P and Lemma K."""
    _count(stats,'leaf_calls')
    m,n = len(A),len(A[0])
    permutation = sorted(range(m),key=lambda i:(shift[i],i))
    transpose = [[A[i][j] for i in permutation] for j in range(n)]
    result = gauss_jordan(F,transpose,stats=stats)
    rank = result['rank']
    profile = []
    reduced = []
    if rank:
        p = result['permutation']
        U = [row[:rank] for row in result['columns'][:rank]]
        PA = [transpose[p[i] if p is not None else i] for i in range(rank)]
        reduced = matrix_product(F,U,PA,stats=stats)
        inv = _inverse(F,result['d'])
        reduced = [[F.mul(inv,x) for x in row] for row in reduced]
        start = 0
        for row in reduced:
            pivot = next(j for j in range(start,m) if not F.is_zero(row[j]))
            _count(stats,'profile_scans',pivot-start+1)
            profile.append(pivot)
            start = pivot+1
    independent = set(profile)
    constant,linear = _zeros(F,m,m),_zeros(F,m,m)
    for i in range(m):
        original = permutation[i]
        if i in independent:
            linear[original][original] = F.one()
        else:
            constant[original][original] = F.one()
            for t,j in enumerate(profile):
                if j<i:
                    constant[original][permutation[j]] = F.neg(reduced[t][i])
    delta = [0]*m
    original_profile = [permutation[i] for i in profile]
    for i in original_profile:
        delta[i] = 1
    _count(stats,'copy_cells',2*m*m+n*m+rank*m)
    return {'basis':_trim(F,[constant,linear]),
            'row_degree':[s+d for s,d in zip(shift,delta)],
            'pivot_degree':delta,'pivots':list(range(m)),
            'ledger':{'order':1,'permutation':permutation,
                      'row_rank_profile':original_profile,
                      'basis_determinant_exponent':rank,
                      'basis_determinant_coefficient':F.one()}}


def _decorate(F,node,stats):
    if node['order']==1:
        active = [j for j,e in enumerate(node['offsets']) if e==0]
        pivots = [i for i,p in enumerate(node['profile']) if p==1]
        onto = len(active)==len(pivots)
        node['all_onto'] = onto
        node['active_constraints'],node['window_pivot_indices'] = active,pivots
        node['window_determinant'] = (constant_determinant(F,
            [[node['input_constant'][i][j] for j in active] for i in pivots],
            stats=stats) if onto else F.zero())
        node['domain_sign_exponent'] = node['codomain_sign_exponent'] = 0
    else:
        d,a = node['order'],node['order_left']
        p1,p2 = node['left_profile'],node['right_profile']
        low = [max(a-e,0) for e in node['offsets']]
        high = [max(d-max(e,a),0) for e in node['offsets']]

        def cross(x,y):
            tail,total = sum(y),0
            for xi,yi in zip(x,y):
                tail -= yi
                total += xi*tail
            return total
        domain,codomain = cross(p2,p1),cross(high,low)
        value = F.mul(node['left']['window_determinant'],
                      node['right']['window_determinant'])
        node['window_determinant'] = F.neg(value) if (domain+codomain)%2 else value
        node['all_onto'] = node['left']['all_onto'] and node['right']['all_onto']
        node['domain_sign_exponent'],node['codomain_sign_exponent'] = domain,codomain
    node['window_ledger_theorem'] = 'algorithm Lemma M; input effective window, not basis determinant'


def pm_basis(F,coefficients,order,shift,mul,*,offsets=None,ledger_hook=None,
             skip_zeros=False,stats=None):
    """Decorated balanced PM-Basis with identical leaves and fast products."""
    m,n = _shape(coefficients)
    if type(order) is not int or order<1:
        raise ValueError('positive integer approximant order required')
    shift = list(shift)
    if len(shift)!=m or any(type(s) is not int for s in shift):
        raise ValueError('one integer shift per input row required')
    if not callable(mul):
        raise ValueError('scalar polynomial multiplier required')
    offsets = [0]*n if offsets is None else list(offsets)
    if len(offsets)!=n or any(type(e) is not int or e<0 for e in offsets):
        raise ValueError('one nonnegative integer offset per input column required')
    for j,e in enumerate(offsets):
        if any(not F.is_zero(C[i][j]) for C in coefficients[:min(e,order)] for i in range(m)):
            raise ValueError('column does not satisfy its declared divisibility offset')
    def product(A,B,limit=None):
        return polynomial_matrix_product(F,A,B,mul,limit=limit,stats=stats)
    def recurse(A,d,s,e):
        if d==1:
            out = order_one(F,A[0],s,stats=stats)
            node = out['ledger']
            node['input_constant'] = [list(row) for row in A[0]]
        else:
            a = d//2
            first = recurse(A[:a],a,s,e)
            residual = product(first['basis'],A,limit=d)[a:d]
            residual += [_zeros(F,m,n) for _ in range(d-a-len(residual))]
            second = recurse(residual,d-a,first['row_degree'],[max(x-a,0) for x in e])
            basis = product(second['basis'],first['basis'])
            delta = [x+y for x,y in zip(first['pivot_degree'],second['pivot_degree'])]
            out = {'basis':basis,'row_degree':[x+y for x,y in zip(s,delta)],
                   'pivot_degree':delta,'pivots':list(range(m))}
            node = {'order':d,'order_left':a,'order_right':d-a,
                    'left_profile':list(first['pivot_degree']),
                    'right_profile':list(second['pivot_degree']),
                    'left':first['ledger'],'right':second['ledger'],
                    'basis_determinant_exponent':sum(delta),
                    'basis_determinant_coefficient':F.one()}
            out['ledger'] = node
        node.update({'shift':list(s),'profile':list(out['pivot_degree']),
                     'offsets':list(e),'rows':m,'columns':n})
        _decorate(F,node,stats)
        if ledger_hook is not None:
            ledger_hook(node)
        return out
    A = [[list(row) for row in C] for C in coefficients[:order]]
    result = recurse(A,order,shift,offsets)
    result.update({'shift':shift,'degree_bound':order,
                   'algorithm':'uniform Strassen original-row PM-Basis (algorithm)'})
    return result
