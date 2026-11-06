# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Field setup and common representation tools.

Role: Shared. Adapt explicit coordinate fields to the guarded native-operation interface used by source/generator acquisition.

Entry points: NativeBaseFieldView.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.uniform_terminal import _FieldOperations


class NativeBaseFieldView(_FieldOperations):

    
    p = None

    def __init__(self, field):
        super().__init__(field)
        self._base = field  

    def info(self, include_inversions=True):
        return {'kind': 'native_effective', 'adapter': type(self._base).__name__,
                'p': None, 'metadata_source': 'owned view; native metadata unqueried'}
