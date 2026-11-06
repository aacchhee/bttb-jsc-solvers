# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from bttb_solvers._internal.halfstrip_core.uniform_terminal import _FieldOperations


class NativeBaseFieldView(_FieldOperations):

    
    p = None

    def __init__(self, field):
        super().__init__(field)
        self._base = field  

    def info(self, include_inversions=True):
        return {'kind': 'native_effective', 'adapter': type(self._base).__name__,
                'p': None, 'metadata_source': 'owned view; native metadata unqueried'}
