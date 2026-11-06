# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from contextlib import contextmanager
from fractions import Fraction


STAGES = (
    "setup",      
    "musearch",   
    "invert",     
    "combine",    
    "corners",    
    "inner",      
    "apply",      
    "det",        
    "transpose",  
    "mark",       
    "repair",     
)


class OpCounter:
    """Measured field operations, plus the model charge for Toeplitz solves."""

    def __init__(self):
        self.add = 0
        self.mul = 0
        self.inv = 0
        self.model_solve_n2 = 0
        self.solves = 0
        self.stages = {}
        self._stack = []          


    def c_add(self, k=1):
        self.add += k

    def c_mul(self, k=1):
        self.mul += k

    def c_inv(self, k=1):
        self.inv += k

    def charge_solve(self, n):
        """The R-1 model charge for one all-branch n x n Toeplitz solve."""
        self.model_solve_n2 += n * n
        self.solves += 1


    def _snapshot(self):
        return (self.add, self.mul, self.inv, self.model_solve_n2, self.solves)

    def _accumulate(self, name, snap):
        d = self.stages.setdefault(
            name, {"add": 0, "mul": 0, "inv": 0, "model_solve_n2": 0, "solves": 0})
        cur = self._snapshot()
        for i, key in enumerate(("add", "mul", "inv", "model_solve_n2", "solves")):
            d[key] += cur[i] - snap[i]

    @contextmanager
    def stage(self, name):
        """`with cnt.stage('invert'): ...` attributes everything inside to that
        stage.  Stages do not nest: entering one SUSPENDS the enclosing stage
        (its delta so far is banked) and leaving RESUMES it, so every field
        operation lands in exactly one stage and none is counted twice."""
        if name not in STAGES:
            raise ValueError("unknown stage %r; the list is fixed: %r" % (name, STAGES))
        if self._stack:
            outer, osnap = self._stack[-1]
            self._accumulate(outer, osnap)
        snap = self._snapshot()
        self._stack.append((name, snap))
        try:
            yield self
        finally:
            nm, sn = self._stack.pop()
            self._accumulate(nm, sn)
            if self._stack:
                outer, _ = self._stack[-1]
                self._stack[-1] = (outer, self._snapshot())


    def total(self):
        """Measured field operations: add + mul + inv."""
        return self.add + self.mul + self.inv

    def report(self, n=None, m=None, k=None):
        rep = {
            "measured": {"add": self.add, "mul": self.mul, "inv": self.inv,
                         "total": self.total()},
            "model": {"solve_n2": self.model_solve_n2, "solves": self.solves},
            "stages": {s: dict(self.stages[s]) for s in self.stages},
            "counted": "Field.add/sub/neg -> add, Field.mul -> mul, Field.inv -> inv; "
                       "div = 1 inv + 1 mul; pow counts the muls it performs.",
            "not_counted": "eq, is_zero, comparisons, index arithmetic, copies.",
            "model_vs_measured": "two columns, NEVER added: model charges each "
                                 "Toeplitz solve n^2 for historical comparability; "
                                 "measured counts the implemented R-1' primitive.",
            "status": "MEASURED (never PROVED).",
        }
        for s in rep["stages"]:
            d = rep["stages"][s]
            d["ops"] = d["add"] + d["mul"] + d["inv"]

            if n:
                d["ops_over_n3"] = Fraction(d["ops"], n ** 3)
            if n and m and k:
                d["ops_over_kmn2"] = Fraction(d["ops"], k * m * n * n)
        return rep
