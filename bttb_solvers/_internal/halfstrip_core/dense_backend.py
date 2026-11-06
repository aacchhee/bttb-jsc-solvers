# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from contextlib import contextmanager
from contextvars import ContextVar


STRASSEN_LEAVES=frozenset((1,2,4,8,16))
_STATE=ContextVar('halfstrip_core_dense_backend',default=None)


def selection():
    state=_STATE.get()
    return ('strassen',1) if state is None else state[:2]


def entry_policy():
    state=_STATE.get()
    return 'off' if state is None else state[3]


def affinity():
    return selection()+(entry_policy(),)


def record(key,amount=1):
    state=_STATE.get()
    if state is not None:
        ledger=state[2]
        ledger[key]=ledger.get(key,0)+amount


def record_max(key,value):
    state=_STATE.get()
    if state is not None:
        ledger=state[2]
        ledger[key]=max(ledger.get(key,0),value)


def metadata():
    backend,leaf=selection()
    return {'dense_backend':backend,'dense_exponent':3 if backend=='classical' else 'log2(7)',
            'strassen_leaf':leaf if backend=='strassen' else None,
            'scalar_backend':'uniform_allfield_fast','polynomial_entry_policy':entry_policy()}


def terminal_method():
    backend,leaf=selection()
    if backend=='classical':method='KN_NATIVE_THREE_PHASE_CLASSICAL_DENSE_SUPPLIED_GENERATORS'
    elif leaf==1:method='KN_NATIVE_THREE_PHASE_STRASSEN_SUPPLIED_GENERATORS'
    else:method='KN_NATIVE_THREE_PHASE_STRASSEN_FIXED_LEAF_'+str(leaf)+'_SUPPLIED_GENERATORS'
    policy=entry_policy()
    return method if policy=='off' else method+'_POLYNOMIAL_ENTRIES_'+policy.upper()


@contextmanager
def dense_scope(backend='strassen',strassen_leaf=1,*,polynomial_entry_policy='off'):
    if type(backend) is not str or backend not in ('strassen','classical'):
        raise ValueError('dense backend must be strassen or classical')
    if type(strassen_leaf) is not int or strassen_leaf not in STRASSEN_LEAVES:
        raise ValueError('Strassen leaf must be one of 1,2,4,8,16')
    if backend=='classical' and strassen_leaf!=1:
        raise ValueError('a Strassen leaf choice requires the Strassen backend')
    if type(polynomial_entry_policy) is not str or polynomial_entry_policy not in ('off','observe','normalize'):
        raise ValueError('polynomial entry policy must be off, observe or normalize')
    ledger={}
    token=_STATE.set((backend,strassen_leaf,ledger,polynomial_entry_policy))
    try:
        yield ledger
    finally:
        _STATE.reset(token)
