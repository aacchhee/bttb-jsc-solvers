# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from fractions import Fraction
from contextlib import nullcontext

from bttb_solvers._internal.banded_bttb.counter import OpCounter as _BandedOpCounter


STAGES = (
    "setup",        
    "leading",      
    "merge",        
    "apply_inv",    
    "terminal",     
    "reconstruct",  
    "det",          
    "transpose",    
    "mark",         
    "chart",        
)


class HalfStripCounter(_BandedOpCounter):
    """`banded_bttb.counter.OpCounter` with algorithm's stage list and the
    convolution model columns."""

    def __init__(self, *, count_ops=True):
        _BandedOpCounter.__init__(self)
        self.count_ops = count_ops
        self.model_conv_cells = 0
        self.convolutions = 0
        self.model_conv2d_cells = 0
        self.model_conv2d_cells_unpadded = 0
        self.kernel_conv2d_cells = 0
        self.conv2d_calls = 0
        self.leading_applies = 0
        self.convolutions_on_apply = 0
        self.backend_counts = {
            "ntt_1d_convolutions": 0,
            "ntt_1d_transforms": 0,
            "ntt_2d_merges": 0,
            "ntt_2d_transforms": 0,
            "ck_1d_convolutions": 0,
            "ck_2d_merges": 0,
            "allfield_1d_convolutions": 0,
            "allfield_2d_merges": 0,
            "schoolbook_1d_convolutions": 0,
            "schoolbook_2d_merges": 0,
        }
        self.backend_by_stage = {}
        if not count_ops:

            
            self.c_add = self.c_mul = self.c_inv = self._forbid_scalar_tick
            self.charge_solve = self.charge_conv = self.charge_apply_inv = self.charge_conv2d = self._omit_model_charge
            self.stage = self._lean_stage

    def _forbid_scalar_tick(self, *args, **kwargs):
        raise AssertionError('lean timing field must not invoke scalar counter ticks')

    def _omit_model_charge(self, *args, **kwargs):
        pass

    def _lean_stage(self, name):
        if name not in STAGES:
            raise ValueError('unknown stage %r' % name)
        return nullcontext(self)


    def _backend_stage(self):
        name = self._stack[-1][0] if self._stack else "(none)"
        return self.backend_by_stage.setdefault(name, {
            "ntt_1d_convolutions": 0,
            "ntt_1d_transforms": 0,
            "ntt_2d_merges": 0,
            "ntt_2d_transforms": 0,
            "ck_1d_convolutions": 0,
            "ck_2d_merges": 0,
            "allfield_1d_convolutions": 0,
            "allfield_2d_merges": 0,
            "schoolbook_1d_convolutions": 0,
            "schoolbook_2d_merges": 0,
        })

    def _charge_backend(self, key, amount=1):
        self.backend_counts[key] += amount
        self._backend_stage()[key] += amount

    def charge_conv(self, length, engine="schoolbook_oracle", transforms=0):
        """One 1-D convolution (linear, cyclic, or a correlation) of length
        `length`.  The model charge is the soft-O cell count, i.e. `length`
        with the log suppressed; the log is stated in the report, never
        multiplied in, so that nobody can read a soft-O as an O."""
        self.model_conv_cells += length
        self.convolutions += 1
        if engine == "ntt":
            self._charge_backend("ntt_1d_convolutions")
            self._charge_backend("ntt_1d_transforms", transforms)
        elif engine == "ck_fft":
            self._charge_backend("ck_1d_convolutions")
        elif engine == "allfield_fft":
            self._charge_backend("allfield_1d_convolutions")
        else:
            self._charge_backend("schoolbook_1d_convolutions")

    def charge_apply_inv(self):
        self.leading_applies += 1

    def charge_conv2d(self, responses, nt, ns, kernel_cells=0, raw=None,
                      engine="schoolbook_oracle", kernel_fresh=False):
        """One divide-and-conquer merge: `responses` right-hand sides through a
        2-D transform padded to nt x ns, charged as the round-17 prototype
        charges it (2 * responses * nt * ns data cells, forward and inverse).

        `raw = (t, s)` is the UNPADDED output extent of the same merge.  Both
        columns are kept because power-of-two padding is jagged: it can double a
        column across one doubling of n and so inflate a fitted exponent by up to
        1, which is an artefact of the padding rule and not of the algorithm.
        The padded column is the one that mirrors algorithm (d) and the round-17
        prototype; the unpadded one is the one to fit."""
        self.model_conv2d_cells += 2 * responses * nt * ns
        if raw is not None:
            self.model_conv2d_cells_unpadded += 2 * responses * raw[0] * raw[1]
        self.kernel_conv2d_cells += kernel_cells
        self.conv2d_calls += 1
        if engine == "ntt":
            self._charge_backend("ntt_2d_merges")
            self._charge_backend("ntt_2d_transforms",
                                 2 * responses + (1 if kernel_fresh else 0))
        elif engine == "ck_fft":
            self._charge_backend("ck_2d_merges")
        elif engine == "allfield_fft":
            self._charge_backend("allfield_2d_merges")
        else:
            self._charge_backend("schoolbook_2d_merges")


    def stage(self, name):
        """Same semantics as the banded counter's `stage`: stages do not nest,
        entering one suspends the enclosing one, every field operation lands in
        exactly one stage.  Reimplemented only because the banded version
        validates against ITS module-level STAGES tuple."""
        if name not in STAGES:
            raise ValueError("unknown stage %r; the list is fixed: %r" % (name, STAGES))
        return _StageCtx(self, name)


    def report(self, n=None, w=None):
        rep = {
            "measured": {"add": self.add, "mul": self.mul, "inv": self.inv,
                         "total": self.total()},
            "model": {"solve_n2": self.model_solve_n2, "solves": self.solves,
                      "conv_cells": self.model_conv_cells,
                      "convolutions": self.convolutions,
                      "leading_applies": self.leading_applies,
                      "conv2d_data_cells": self.model_conv2d_cells,
                      "conv2d_data_cells_unpadded": self.model_conv2d_cells_unpadded,
                      "conv2d_kernel_cells": self.kernel_conv2d_cells,
                      "merges": self.conv2d_calls},
            "backend": dict(self.backend_counts),
            "backend_by_stage":
                {s: dict(v) for s, v in sorted(self.backend_by_stage.items())},
            "stages": {s: dict(self.stages[s]) for s in self.stages},
            "counted": "Field.add/sub/neg -> add, Field.mul -> mul, Field.inv -> inv; "
                       "div = 1 inv + 1 mul; pow counts the muls it performs.",
            "not_counted": "eq, is_zero, comparisons, index arithmetic, copies.",
            "model_vs_measured": "MEASURED field operations and the MODEL cell "
                                 "ledger are never added. The backend and "
                                 "backend_by_stage maps say whether those cells "
                                 "were executed by NTTs, uniform all-field trees, or the "
                                 "schoolbook correctness oracle.",
            "convolutions_per_leading_apply":
                (Fraction(self.convolutions_on_apply, self.leading_applies)
                 if self.leading_applies else None),
            "status": "MEASURED (never PROVED).",
        }
        if not self.count_ops:
            rep['status'] = 'DISABLED for lean timing; operation counts unavailable'
            rep['operation_counts_valid'] = False
        for s in rep["stages"]:
            d = rep["stages"][s]
            d["ops"] = d["add"] + d["mul"] + d["inv"]
            if n:
                d["ops_over_n3"] = Fraction(d["ops"], n ** 3)
            if n and w:
                d["ops_over_wn3"] = Fraction(d["ops"], w * n ** 3)
        return rep


class _StageCtx(object):
    def __init__(self, cnt, name):
        self.cnt, self.name = cnt, name

    def __enter__(self):
        cnt = self.cnt
        if cnt._stack:
            outer, osnap = cnt._stack[-1]
            cnt._accumulate(outer, osnap)
        cnt._stack.append((self.name, cnt._snapshot()))
        return cnt

    def __exit__(self, *exc):
        cnt = self.cnt
        nm, sn = cnt._stack.pop()
        cnt._accumulate(nm, sn)
        if cnt._stack:
            outer, _ = cnt._stack[-1]
            cnt._stack[-1] = (outer, cnt._snapshot())
        return False
