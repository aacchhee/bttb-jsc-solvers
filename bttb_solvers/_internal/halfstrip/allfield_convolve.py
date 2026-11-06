# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""HIR, shared multiplication and alternate terminal components.

Role: Shared. Select the uniform or Cantor-Kaltofen scalar multiplier for inverse-series and terminal controls.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from copy import deepcopy

from .allfield_poly import make_multiplier, multiply_rectangles, LEAF_CUTOFF
from .ck_poly import make_multiplier as make_ck_multiplier


def select_multiplier(F, kind):
    """Construct one recognized primitive; selection never depends on field values."""
    if kind == "allfield":
        return make_multiplier(F)
    if kind == "ck":
        return make_ck_multiplier(F)
    raise ValueError("multiplier must be allfield or ck")


class AllFieldConvolutionBackend:
    kind = "allfield_fft"
    achieved = True

    def __init__(self,F,cnt=None,multiplier="allfield"):
        self.F,self.cnt = F,cnt
        self.multiplier_kind = multiplier
        self.kind = "ck_fft" if multiplier == "ck" else "allfield_fft"
        self._multiply = select_multiplier(F,multiplier)
        self.merge_calls = 0
        self.kronecker_max_input = 0

    def summary(self):
        return {"kind":self.kind,"requested":self.kind,
                "root_searches":0,"root_orders_required":[],
                "radix":self._multiply.stats["radix"],
                "leaf_cutoff":LEAF_CUTOFF,"zero_skipping":False,
                "achieved":True,"cubic_ledger_achieved":True,
                "formal_primitive":("recursive cyclotomic quotient CK" if self.multiplier_kind == "ck" else "uniform synthetic cyclotomic-ring FFT"),
                "scalar_multiplier_kind":self.multiplier_kind,
                ("sharp_multiplication_bound" if self.multiplier_kind == "ck" else "coarse_multiplication_bound"):self._multiply.operation_bound + " field operations",
                "bound_scope":"selected scalar multiplication primitive; proof receipt separate from telemetry",
                "two_dimensional_route":"no-carry Kronecker scalar multiplication; stride 3n-2",
                "kernel_transform_sharing":False,
                "multiplication":deepcopy(self._multiply.stats),
                "multiplication_stats_scope":"cumulative callable telemetry; may include shared terminal products",
                "merge_calls":self.merge_calls,"kronecker_max_input":self.kronecker_max_input,
                "reason":("recursive field-specialized cyclotomic quotient CK in every characteristic; fixed small exact leaves only" if self.multiplier_kind == "ck" else "recursive synthetic-ring FFT in every characteristic; fixed small exact leaves only")}

    def linear(self,a,b):
        result = self._multiply(a,b)
        if a and b and self.cnt is not None:
            self.cnt.charge_conv(len(a)+len(b)-1,engine=self.kind,transforms=0)
        return result

    def cyclic(self,a,b):
        if len(a)!=len(b):
            raise ValueError("cyclic convolution needs equal lengths")
        n = len(a)
        result = [self.F.zero()]*n
        for k,value in enumerate(self.linear(a,b)):
            result[k%n] = self.F.add(result[k%n],value)
        return result

    def linear_corr(self,h,v):
        if len(h)!=len(v):
            raise ValueError("linear correlation needs equal lengths")
        n = len(h)
        return self.linear(list(reversed(h)),v)[n-1:2*n-1] if n else []

    def cyclic_corr(self,a,v):
        if len(a)!=len(v):
            raise ValueError("cyclic correlation needs equal lengths")
        n = len(a)
        return self.cyclic([a[(-i)%n] for i in range(n)],v)

    corr = linear_corr

    def merge_window(self,sources,kernel,n,first,last,left):
        """Return the physical merge window and False for kernel cache sharing."""
        if n<1 or left<0 or left>len(sources) or first<0 or last<first:
            raise ValueError("invalid merge window dimensions")
        if any(len(row)!=n for row in sources[:left]):
            raise ValueError("source row width must equal n")
        if any(row is not None and len(row)!=2*n-1 for row in kernel):
            raise ValueError("kernel row width must equal 2n-1")
        zero = self.F.zero()
        if not left or not kernel:
            return {tau:[zero]*n for tau in range(first,last)},False

        K = [[zero]*(2*n-1)]
        K += [list(row) if row is not None else [zero]*(2*n-1) for row in kernel[1:]]
        stride = 3*n-2
        self.kronecker_max_input = max(self.kronecker_max_input,
            stride*(left-1)+n,stride*(len(kernel)-1)+2*n-1)
        C = multiply_rectangles(self.F,[list(row) for row in sources[:left]],K,self._multiply)
        self.merge_calls += 1
        return {tau:C[tau][n-1:2*n-1] if tau<len(C) else [zero]*n
                for tau in range(first,last)},False

