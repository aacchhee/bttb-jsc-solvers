# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.exact import Restart, pd
from bttb_solvers._internal.label_algebra import setup_banded_label_algebra
from bttb_solvers._internal.grouped_chart import _GroupedChartSearch
from bttb_solvers._internal.banded_frontend_rank import narrow_raw_minor
from bttb_solvers._internal.banded_polynomial_determinant import NonsingularDeterminant, determinant_from_maximum_minor
from bttb_solvers._internal.toeplitz_repair import repair_toeplitz_symbol


def _retained_factor(context,event):
    """Validate a monic actual proper factor of the CURRENT owned modulus."""
    if not context.added_algebra:
        raise event
    field,factor=context.field,list(event.factor)
    base=field.base
    if not 2<=len(factor)<len(field.modulus):
        raise event
    if not base.is_zero(base.sub(factor[-1],base.one())):
        raise event
    _quotient,remainder=pd(base,field.modulus,factor)
    if any(not base.is_zero(value) for value in remainder):
        raise event
    return tuple(factor)


class NarrowFrontendAttempt(dict):
    """Same-attempt bundle; its driver invalidates it before restriction replay."""
    def __init__(self,driver,data):
        super().__init__(data)
        self.driver=driver

    def invalidate(self):
        number=self.get('attempt_number')
        self.clear()
        self.update(status='INVALIDATED',attempt_number=number,
                    invalidated='all frontend field/context/rank/determinant/repair/chart state')


def _raw_cache(n,k,raw):
    if type(n) is not int or type(k) is not int or n<3 or not 1<=k<n or 2*k>=n:
        raise ValueError('raw narrow frontend requires integers n>=3, 1<=k<n, 2k<n')
    keys=tuple((outer,inner) for outer in range(-k,k+1) for inner in range(1-n,n))
    if any(key not in raw for key in keys):
        raise ValueError('missing original raw lag coefficient')
    return keys,tuple(raw[key] for key in keys)


class RawNarrowFrontend:
    """Own original-native ingress and all frontend restriction attempts.

    prepare() finishes the actual frontend. If a future downstream operation
    raises Restart, its global owner must discard ALL its downstream state,
    then call discard_and_replay(event,stage); this redoes the frontend from
    this immutable original-native cache. This class does not yet run those
    carrier/correction/query stages or claim a complete global solver scope.
    """
    def __init__(self,base,n,k,raw):
        self.base,self.n,self.k=base,n,k
        self.raw_keys,self.original_native_cache=_raw_cache(n,k,raw)
        self.restriction=None
        self.restarts=[]
        self.attempts=0
        self._retained_degree=None
        self.current=None

    def _record_restriction(self,receipt):
        old,new=receipt['old_degree'],receipt['new_degree']
        if not 1<=new<old or (self._retained_degree is not None and old!=self._retained_degree):
            raise ArithmeticError('global frontend restriction did not strictly lower current degree')
        if len(self.restarts)+1>receipt['initial_degree']-1:
            raise ArithmeticError('global frontend restriction exceeded degree-drop bound')
        self.restriction,self._retained_degree=receipt['factor'],new
        self.restarts.append(dict(receipt,attempt=self.attempts,
            invalidated='all attempt frontend state; downstream owner must discard every dependent object'))

    @staticmethod
    def _restriction_receipt(context,event,stage):
        factor=_retained_factor(context,event)
        return {'old_degree':context.field.degree,'new_degree':len(factor)-1,
                'factor':factor,'guard_index':event.guard_index,'stage':stage,
                'initial_degree':context.metadata['initial_extension_degree']}

    def _attempt(self):

        
        context=setup_banded_label_algebra(self.base,self.original_native_cache,
                                           self.n,self.k,self.restriction)
        field=context.field
        stage='raw_symbol_and_normal_rank'
        try:
            raw=dict(zip(self.raw_keys,context.coefficients))
            stats={'rank':{},'minor_determinant':{},'repair':{},'repaired_determinant':{}}
            minor=narrow_raw_minor(field,self.n,self.k,raw,stats=stats['rank'])
            stage='normalized_maximum_minor_determinant'
            delta=determinant_from_maximum_minor(field,minor,stats=stats['minor_determinant'])
            stage='actual_toeplitz_repair'
            repair=repair_toeplitz_symbol(context,minor['symbol'],minor,delta,stats=stats['repair'])
            common={'status':repair['status'],'field':field,'context':context,
                    'n':self.n,'k':self.k,'raw':raw,
                    'original_native_cache':self.original_native_cache,
                    'raw_keys':self.raw_keys,'symbol':minor['symbol'],
                    'maximum_minor':minor,'minor_determinant':delta,
                    'normal_rank':minor['normal_rank'],'repair':repair,
                    'stats':stats,'attempt_number':self.attempts,
                    'scope':'raw narrow frontend only; physical determinant and solve remain open',
                    'restart_scope':'whole frontend; future global owner must extend through carrier/correction/queries'}
            if repair['status']=='PHYSICAL_SINGULAR_CERTIFICATE':
                common.update(physical_singular=True,certificate=repair['certificate'])
                return common,None
            if repair['status']!='REPAIRED_SYMBOL':
                raise ArithmeticError('unexpected accepted repair status')
            stage='normalized_repaired_symbol_determinant'

            H=repair['H']

            

            

            identity=list(range(self.n))
            if (minor['normal_rank']==self.n and repair['nullity']==0
                    and minor['rows']==identity and minor['columns']==identity):
                determinant=list(delta)
                stats['repaired_determinant']['same_attempt_full_rank_reuses']=1
            else:
                determinant=NonsingularDeterminant(field,stats['repaired_determinant']).determinant_nonsingular(H)
            stage='grouped_unit_chart_search'
            chart=_GroupedChartSearch(context).select(determinant)
            common.update(status='NARROW_FRONTEND_READY',physical_singular=None,
                          H=H,X=repair['X'],Y=repair['Y'],E0=repair['E0'],
                          repaired_determinant=determinant,chart=chart,
                          c=chart['c'],modulus=chart['modulus'])
            return common,None
        except Restart as event:

            
            return None,self._restriction_receipt(context,event,stage)

    def prepare(self):
        if self.current is not None:
            return self.current
        while True:
            self.attempts+=1
            result,receipt=self._attempt()
            if receipt is not None:
                self._record_restriction(receipt)
                continue
            result.update(attempts=self.attempts,restarts=list(self.restarts))
            self.current=NarrowFrontendAttempt(self,result)
            return self.current

    def discard_and_replay(self,event,stage):
        """Global-owner seam after discarding ALL future downstream state.

        This is restriction replay, not an oracle/callback executor. stage is
        diagnostic text. The driver stores no exception/traceback, but this
        call's live event and external references can retain failed objects.
        Bundle invalidation is logical, not an immediate-release guarantee.
        The final whole owner must leave its minimal failed-attempt frame
        with only a native-factor/integer receipt before fresh full replay.
        """
        if not isinstance(event,Restart) or self.current is None:
            raise ValueError('a current attempt and its actual Restart are required')
        receipt=self._restriction_receipt(self.current['context'],event,stage)
        self.current.invalidate()
        self.current=None
        self._record_restriction(receipt)
        return self.prepare()


def prepare_narrow_frontend(base,n,k,raw):
    """Original native raw lag mapping -> actual same-attempt frontend bundle."""
    return RawNarrowFrontend(base,n,k,raw).prepare()
