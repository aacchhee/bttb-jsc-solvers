# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Structured CC comparison.

Role: Trial. Acquire the supplied-root structured CC front end, symbol rank/determinant, valid parameter and coordinate context. This is distinct from cc_prepared.py.

Entry points: prepare_hb_frontend.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.banded_bttb.field import PrimeField, ExtensionField
from bttb_solvers._internal.banded_bttb.polynomial import ProductTree
from bttb_solvers._internal.native_field_view import NativeBaseFieldView
from bttb_solvers._internal.label_algebra import LabelAlgebraContext
from bttb_solvers._internal.banded_frontend_rank import narrow_raw_minor
from bttb_solvers._internal.banded_polynomial_determinant import determinant_from_maximum_minor, NonsingularDeterminant
from bttb_solvers._internal.toeplitz_repair import repair_toeplitz_symbol
from bttb_solvers._internal.fast_scalar import FastScalar
from bttb_solvers._internal.halfstrip_core.dense_backend import selection, metadata as dense_metadata


METHOD='STRUCTURED_CC_SUPPLIED_ROOT_MU_STRASSEN'


def _method():
    backend,leaf=selection()
    if (backend,leaf)==('strassen',1):return METHOD
    if backend=='classical':return 'STRUCTURED_CC_SUPPLIED_ROOT_MU_CLASSICAL_DENSE'
    return 'STRUCTURED_CC_SUPPLIED_ROOT_MU_'+backend.upper()+'_LEAF_'+str(leaf)


def _power(F,value,exponent):
    result=F.one()
    while exponent:
        if exponent&1:result=F.mul(result,value)
        exponent//=2
        if exponent:value=F.mul(value,value)
    return result


def _coordinate_key(coordinate_field,value):

    
    if isinstance(coordinate_field,PrimeField):
        if type(value) is not int or not 0<=value<coordinate_field.q:
            raise ValueError('noncanonical supplied prime-field coordinate')
        return value
    if (type(value) is not tuple or len(value)!=coordinate_field.e or
            any(type(digit) is not int or not 0<=digit<coordinate_field.p for digit in value)):
        raise ValueError('noncanonical supplied extension-field coordinate')
    return value


def _prepare_context(coordinate_field,n,k,omega,raw):
    if not isinstance(coordinate_field,(PrimeField,ExtensionField)):
        raise ValueError('supplied explicit PrimeField/ExtensionField representation required')
    if type(n) is not int or type(k) is not int or n<3 or not 1<=2*k<n:
        raise ValueError('HB square regime requires integer n>=3 and 1<=2k<n')
    q=coordinate_field.q
    if type(q) is not int or (q-1)%n or (q-1)//n<=2*k*n:
        raise ValueError('HB requires n dividing q-1 and (q-1)/n>2kn')
    F=NativeBaseFieldView(coordinate_field)
    _coordinate_key(coordinate_field,omega)
    value=F.one()
    for order in range(1,n+1):
        value=F.mul(value,omega)
        is_one=F.is_zero(F.sub(value,F.one()))
        if is_one!=(order==n):
            raise ValueError('supplied omega must have exact order n')
    keys=tuple((outer,inner) for outer in range(-k,k+1) for inner in range(1-n,n))
    if any(key not in raw for key in keys):
        raise ValueError('missing original HB raw coefficient')
    coefficients=tuple(raw[key] for key in keys)
    for a in coefficients:_coordinate_key(coordinate_field,a)
    D=2*k*n;M=D+1
    labels=tuple(coordinate_field.from_int(i) for i in range(1,M+1))
    for a in labels:
        _coordinate_key(coordinate_field,a)
        if F.is_zero(a):raise ArithmeticError('coordinate catalog returned zero')
    if not F.is_zero(F.sub(labels[0],F.one())):
        raise ArithmeticError('coordinate catalog must start with one')
    metadata={'n':n,'k':k,'D':D,'M':M,'required_labels':M,
              'added_algebra':False,'coefficient_count':len(coefficients),
              'method':_method(),'coordinate_representation':'supplied int/tuple; from_int is coordinate enumeration',
              'root_verification_products':n,'catalog_creation_count':M,
              'cost_scope':'supplied finite field/root; no extension construction or q-1 factorization'}
    context=LabelAlgebraContext(coordinate_field,F,coefficients,labels,metadata,False)
    metadata.update(dense_metadata())
    context.label_ids=tuple(range(M))
    return context,keys


def prepare_hb_frontend(coordinate_field,n,k,omega,raw):
    """Raw native lags -> actual repaired/scaled HB data and a genuine c1 chart.

    This is frontend-only. It returns no physical determinant or solution.
    Only the accepted finite-window inequality can certify physical singularity.
    """
    context,keys=_prepare_context(coordinate_field,n,k,omega,raw)
    F=context.field;raw_original=dict(zip(keys,context.coefficients))
    stats={'rank':{},'minor_determinant':{},'repair':{},'repaired_determinant':{}}
    minor=narrow_raw_minor(F,n,k,raw_original,stats=stats['rank'])
    delta=determinant_from_maximum_minor(F,minor,stats=stats['minor_determinant'])
    repair=repair_toeplitz_symbol(context,minor['symbol'],minor,delta,stats=stats['repair'])
    common={'status':repair['status'],'method':_method(),'field':F,'context':context,
            'coordinate_field':coordinate_field,'n':n,'k':k,'omega':omega,
            'original_native_cache':context.coefficients,'raw_keys':keys,
            'raw_original':raw_original,'symbol_raw':minor['symbol'],
            'maximum_minor':minor,'minor_determinant':delta,'normal_rank':minor['normal_rank'],
            'repair':repair,'stats':stats,
            'scope':'HB supplied-field/root frontend only; no physical determinant or solve'}
    if repair['status']=='PHYSICAL_SINGULAR_CERTIFICATE':
        common.update(physical_singular=True,certificate=repair['certificate'])
        return common
    if repair['status']!='REPAIRED_SYMBOL':
        raise ArithmeticError('unexpected actual repair result')
    Hraw=repair['H']
    K=NonsingularDeterminant(F,stats['repaired_determinant'])
    fraw=K.determinant_nonsingular(Hraw)
    D,M=context.metadata['D'],context.metadata['M']
    if not fraw or len(fraw)-1>D:
        raise ArithmeticError('certified repaired determinant lost degree bound')

    
    seen=set();representatives=[];coset_keys=[];examined=0
    for code in range(1,n*D+2):
        mu=coordinate_field.from_int(code)
        _coordinate_key(coordinate_field,mu)
        if F.is_zero(mu):raise ArithmeticError('nonzero coordinate prefix returned zero')
        key=_coordinate_key(coordinate_field,_power(F,mu,n))
        examined+=1
        if key not in seen:
            seen.add(key);representatives.append(mu);coset_keys.append(key)
            if len(representatives)==M:break
    if len(representatives)!=M:
        raise ArithmeticError('proved HB coset prefix exhausted')
    inverse_omega=F.inv(omega)
    roots=[F.one()]
    for _ in range(n-1):roots.append(F.mul(roots[-1],inverse_omega))
    points=[F.mul(mu,root) for mu in representatives for root in roots]
    scalar=FastScalar(F)
    values=ProductTree(F,points,arithmetic=scalar).evaluate(fraw)
    chosen=None
    for i in range(M):
        if all(not F.is_zero(value) for value in values[i*n:(i+1)*n]):
            chosen=i;break
    if chosen is None:raise ArithmeticError('nonzero polynomial exhausted proved disjoint HB cosets')
    mu=representatives[chosen];inverse_mu=F.inv(mu)

    scales=[_power(F,inverse_mu,k)]
    for _ in range(2*k):scales.append(F.mul(scales[-1],mu))
    H=[[K.trim([F.mul(a,scales[d]) for d,a in enumerate(p)]) for p in row] for row in Hraw]
    scaled_raw={(outer,inner):F.mul(value,scales[outer+k])
                for (outer,inner),value in raw_original.items()}
    fscale=_power(F,inverse_mu,k*n)
    fscaled=[];scale=fscale
    for a in fraw:
        fscaled.append(F.mul(a,scale));scale=F.mul(scale,mu)
    fscaled=K.trim(fscaled)
    modulus=[F.neg(F.one())]+[F.zero() for _ in range(n-1)]+[F.one()]
    remainder=scalar.divmod(fscaled,modulus)[1]
    gcd,u,v=scalar.xgcd(remainder,modulus)
    if len(gcd)!=1 or not F.is_zero(F.sub(gcd[0],F.one())):
        raise ArithmeticError('certified HB root grid did not give a unit c1 chart')
    selected_values=values[chosen*n:(chosen+1)*n]
    chart={'status':'UNIT_CHART','method':_method(),'acquisition':'HB_SUPPLIED_ROOT_GRID',
           'field':F,'context':context,'c':F.one(),'candidate_id':0,'modulus':modulus,
           'remainder':remainder,'gcd':gcd,'bezout_remainder':u,'bezout_modulus':v,
           'omega':omega,'roots':roots,'raw_grid':points[chosen*n:(chosen+1)*n],
           'raw_values':selected_values,'scaled_values':[F.mul(fscale,a) for a in selected_values],
           'metadata':{'source':'algorithm / algorithm; actual supplied-root grid',
                       'grouped_search_used':False,'witness_scope':'u*remainder+v*(z^n-1)=1'}}
    common.update(status='HB_FRONTEND_READY',physical_singular=None,mu=mu,
                  raw=scaled_raw,H=H,H_raw=Hraw,X=repair['X'],Y=repair['Y'],E0=repair['E0'],
                  repaired_determinant_raw=fraw,repaired_determinant=fscaled,
                  chart=chart,c=chart['c'],modulus=modulus,
                  search={'representatives':representatives,'coset_keys':coset_keys,
                          'examined_coordinates':examined,'prefix_bound':n*D+1,
                          'points':points,'values':values,'selected_coset':chosen,
                          'scalar_multipoint_calls':1,'point_count':n*M},
                  scalar_stats=scalar.stats)
    return common
