# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .exact import base_factory, irreducible_modulus, EtaleField

def raw_scalar(F,name,x):
    if name=='F4':return F.reduce([x&1,(x>>1)&1])
    return F.from_int(x)

def setup_field(fixture,restriction=None):
    base=base_factory(fixture['field']);n=fixture['n'];p=base.p
    meta={'base':fixture['field'],'restriction':restriction,'construction':None}
    probes=[];small_characteristic=None
    for integer in range(1,n+1):
        value=base.from_int(integer);zero=base.is_zero(value);probes.append({'integer':integer,'value':value,'zero':zero})
        if zero:small_characteristic=integer;break
    meta['characteristic_probes']=probes
    if small_characteristic is not None:
        p=small_characteristic
        e=1
        while p**e<n+1:e+=1
        f,cert=irreducible_modulus(p,e)
        modulus=[base.from_int(x) for x in f] if restriction is None else restriction
        F=EtaleField(base,modulus);meta['construction']={'p':p,'e':e,'prime_modulus':f,'certificate':cert}
        labels=[]
        for code in range(1,n+1):
            digits=[];c=code
            for _ in range(e):digits.append(base.from_int(c%p));c//=p
            labels.append(F.reduce(digits))
        blocks=[[[F.embed(raw_scalar(base,fixture['field'],x)) for x in row] for row in block] for block in fixture['blocks']]
    else:
        F=base;labels=[F.from_int(i+1) for i in range(n)]
        blocks=[[[raw_scalar(F,fixture['field'],x) for x in row] for row in block] for block in fixture['blocks']]
    meta['labels']=labels;meta['algebra']=F.info()
    return F,blocks,labels,meta

def descend(F,x):
    if not isinstance(F,EtaleField):return x
    if not F.is_zero(F.sub(x,F.embed(x[0]))):
        raise ArithmeticError('physical output did not descend to named base field')
    return x[0]


class BaseFieldView:
    """Treat the caller's entire field as the named base, including extensions.

    The view deliberately is not an EtaleField: automatic descent removes only
    the extra working algebra constructed by setup_effective_field. Arithmetic
    and element representations remain owned by the original adapter.
    """
    def __init__(self, base):
        self._base = base
        self.p = getattr(base, 'p', 0)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def info(self, include_inversions=True):
        describe = getattr(self._base, 'info', None)
        if describe is None:
            return {'kind': 'effective', 'adapter': type(self._base).__name__,
                    'p': self.p}
        try:
            return describe(include_inversions)
        except TypeError:
            return describe()


def setup_effective_field(base, blocks, n, restriction=None):
    """Rebuild a working algebra from already-embedded base coefficients."""
    base = BaseFieldView(base)
    probes = []
    characteristic = None
    for integer in range(1, n+1):
        value = base.from_int(integer)
        zero = base.is_zero(value)
        probes.append({'integer': integer, 'value': value, 'zero': zero})
        if zero:
            characteristic = integer
            break
    meta = {'base': base.info(False), 'restriction': restriction,
            'construction': None, 'characteristic_probes': probes}
    if characteristic is None:
        F = base
        labels = [base.from_int(i+1) for i in range(n)]
        embed = lambda value: value
    else:
        p = characteristic
        e = 1
        while p**e < n+1:
            e += 1
        polynomial, certificate = irreducible_modulus(p, e)
        modulus = ([base.from_int(value) for value in polynomial]
                   if restriction is None else restriction)
        F = EtaleField(base, modulus)
        labels = []
        for code in range(1, n+1):
            digits = []
            value = code
            for _ in range(e):
                digits.append(base.from_int(value % p))
                value //= p
            labels.append(F.reduce(digits))
        embed = F.embed
        meta['construction'] = {'p': p, 'e': e,
                                'prime_modulus': polynomial,
                                'certificate': certificate}
    embedded = [[[embed(value) for value in row] for row in block]
                for block in blocks]
    meta.update(labels=labels, algebra=F.info())
    return F, embedded, labels, meta, embed
