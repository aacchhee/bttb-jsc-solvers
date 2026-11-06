# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Displacement construction trials.

Role: Trial. Acquire wide-band Stein generators, optionally compress them, and solve the resulting physical service through supplied-generator algebra.

Entry points: WidePhysicalSolver; solve_wide.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from bttb_solvers._internal.halfstrip_core.dense_backend import metadata as dense_metadata, terminal_method, affinity
from bttb_solvers._internal.kn_services import Services, METHOD
from bttb_solvers._internal.wide_raw_generator import acquire
from bttb_solvers._internal.generator_compression import compress_generators as _compress_generators


COMPRESSED_METHOD='reference_RAW_STEIN_TWO_PASS_COMPRESSION_KN'
UNCOMPRESSED_METHOD='reference_RAW_STEIN_UNCOMPRESSED_KN'


class SingularPhysicalMatrix(ValueError):
    """The supplied raw physical matrix has a nontrivial kernel."""


class WidePhysicalSolver:
    def __init__(self, F, n, k, raw, *, compress_generators=True):
        if type(compress_generators) is not bool:
            raise ValueError('compress_generators must be a Boolean route choice')
        self._dense_selection=affinity()
        self.F=F
        self.generator=acquire(F,n,k,raw)
        self.stats={}
        self.g,self.h=self.generator.stein(F)
        acquired_width=len(self.g[0])
        pass_ranks=[]
        pass_widths=[]
        if compress_generators:
            first=_compress_generators(F,self.g,self.h)
            self.g,self.h=first.G,first.H
            pass_ranks.append(first.rank)
            pass_widths.append({'input_width':first.input_width,'output_width':first.rank})
            self.stats['generator_compression_passes']=[first.stats]
            del first
            if pass_ranks[-1]:

                
                second=_compress_generators(F,self.h,self.g)
                self.g,self.h=second.H,second.G
                pass_ranks.append(second.rank)
                pass_widths.append({'input_width':second.input_width,'output_width':second.rank})
                self.stats['generator_compression_passes'].append(second.stats)
                del second
        method=COMPRESSED_METHOD if compress_generators else UNCOMPRESSED_METHOD
        self.metadata={'method':method,'terminal_method':terminal_method(),
                       'route':'compressed' if compress_generators else 'uncompressed',
                       'source':'algorithm raw wide / algorithm Stein / KN',
                       'physical_order':n*n,'represented_order':self.generator.order,
                       'stein_width':len(self.g[0]),
                       'acquired_stein_width':acquired_width,
                       'compression_enabled':compress_generators,
                       'compression_pass_ranks':pass_ranks,
                       'compression_pass_widths':pass_widths,
                       'stein_width_before_compression':acquired_width,
                       'stein_width_after_compression':len(self.g[0]),
                       'padding':self.generator.order-n*n,
                       'nonsingularity_certified':False,
                       'homogeneous_setup_calls':1,'forward_calls':0,'transpose_calls':0}
        self.metadata.update(dense_metadata())
        self.terminal=Services(F,self.stats)
        kernel=self.terminal.solve_stein(self.g,self.h,homogeneous=True)
        if kernel['status']!='NULLSPACE':
            raise ArithmeticError('KN homogeneous service did not return a kernel description')

        
        self.metadata['paid_dense_base']=kernel['paid_dense_base']
        self.metadata['setup_phase_degree_budget']=kernel.get('phase_degree_budget')
        if kernel['nullity']:
            error=SingularPhysicalMatrix('singular raw physical matrix')

            
            error.metadata=dict(self.metadata)
            error.statistics=dict(self.stats)
            raise error
        self.metadata['nonsingularity_certified']=True

    def _solve(self, rhs, transposed):
        if affinity()!=self._dense_selection:
            raise ValueError('prepared wide solver must run under its original dense selection')
        represented=self.generator.rhs(self.F,rhs)
        represented.reverse()
        g,h=(self.h,self.g) if transposed else (self.g,self.h)
        answer=self.terminal.solve_stein(g,h,represented)
        self.metadata['transpose_calls' if transposed else 'forward_calls']+=1
        if answer['status']!='CONSISTENT':
            raise ArithmeticError('inconsistent result after certified physical nonsingularity')
        solution=list(reversed(answer['solution']))
        self.metadata['last_phase_degree_budget']=answer.get('phase_degree_budget')
        return self.generator.recover(solution)

    def solve(self, rhs):
        return self._solve(rhs,False)

    def solve_transpose(self, rhs):
        """Independent literal-transpose RHS; swap the same Stein factors."""
        return self._solve(rhs,True)


def prepare_wide(F,n,k,raw,*,compress_generators=True):
    return WidePhysicalSolver(F,n,k,raw,compress_generators=compress_generators)


def solve_wide(F,n,k,raw,forward_rhs,transpose_rhs,*,compress_generators=True):
    """Acquire once, certify nonsingularity, solve both independently supplied RHS."""
    solver=prepare_wide(F,n,k,raw,compress_generators=compress_generators)
    forward=solver.solve(forward_rhs)
    transposed=solver.solve_transpose(transpose_rhs)
    return {'solution':forward,'transpose_solution':transposed,
            'metadata':dict(solver.metadata),'statistics':dict(solver.stats)}
