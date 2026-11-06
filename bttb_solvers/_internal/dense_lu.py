# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Physical dense baseline.

Role: Main. Owned native FLINT matrices, checked ABI binding, one physical LU, batched literal-transpose/forward solves and determinant extraction.

Entry points: NativeAPI; OwnedMat; NativeLU.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import platform
import sys

U = C.c_ulong
S = C.c_long


class NMod(C.Structure):
    _fields_ = [('n', U), ('ninv', U), ('norm', U)]


class Mat(C.Structure):
    _fields_ = [('entries', C.POINTER(U)), ('r', S), ('c', S),
                ('stride', S), ('mod', NMod)]


class SingularPhysicalMatrix(ArithmeticError):
    """LU rank check refused a singular physical matrix; no solves attempted."""


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    default=str).encode()).hexdigest()


def permutation_sign(perm):
    """Sign in linear time, also rejecting malformed native permutations."""
    n = len(perm)
    seen = bytearray(n)
    membership = bytearray(n)
    for value in perm:
        if not 0 <= value < n or membership[value]:
            raise ArithmeticError('invalid native LU permutation')
        membership[value] = 1
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            while not seen[i]:
                seen[i] = 1
                i = perm[i]
    return -1 if (n - cycles) % 2 else 1


class NativeAPI:
    def __init__(self):
        import flint
        if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
            raise RuntimeError('dense LU requires pinned Linux x86_64 wheel')
        if C.sizeof(U) != 8 or C.sizeof(S) != 8 or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError('dense LU requires LP64 ABI')
        if flint.__version__ != '0.9.0' or str(flint.__FLINT_VERSION__) != '3.6.0':
            raise RuntimeError('dense LU requires python-flint 0.9.0 / FLINT 3.6.0')
        vendor = Path(flint.__file__).resolve().parent.parent
        candidates = sorted((vendor / 'python_flint.libs').glob('libflint*.so*'))
        if len(candidates) != 1:
            raise RuntimeError('expected exactly one bundled libflint, got ' + str(candidates))
        self.path = candidates[0].resolve()
        self.lib = C.CDLL(str(self.path))
        native_version = (C.c_char * 6).in_dll(self.lib, 'flint_version').value.decode()
        if native_version != '3.6.0':
            raise RuntimeError('bundled native FLINT version mismatch: ' + native_version)
        ptr = C.POINTER(Mat)
        self._bind('nmod_mat_init', [ptr, S, S, U])
        self._bind('nmod_mat_clear', [ptr])
        self._bind('nmod_mat_lu', [C.POINTER(S), ptr, C.c_int], S)
        self._bind('nmod_mat_transpose', [ptr, ptr])
        self._bind('nmod_mat_solve_tril', [ptr, ptr, ptr, C.c_int])
        self._bind('nmod_mat_solve_triu', [ptr, ptr, ptr, C.c_int])
        self._bind('flint_set_num_threads', [C.c_int])
        self._bind('flint_get_num_threads', [], C.c_int)
        self.lib.flint_set_num_threads(1)
        if self.lib.flint_get_num_threads() != 1:
            raise RuntimeError('native FLINT did not accept one thread')
        self.provenance = dict(python_flint_version=flint.__version__, flint_version=native_version,
            native_library=self.path.name, native_library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            matrix_struct_bytes=C.sizeof(Mat), native_threads=1)

    def _bind(self, name, args, result=None):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = args, result


_API = None


def get_api():
    global _API
    if _API is None:
        _API = NativeAPI()
    return _API


class OwnedMat:
    def __init__(self, api, r, c, p):
        self.api, self.mat, self.closed = api, Mat(), True
        api.lib.nmod_mat_init(C.byref(self.mat), r, c, p)
        self.closed = False
        if self.mat.r != r or self.mat.c != c or self.mat.stride < c or self.mat.mod.n != p:
            self.close()
            raise RuntimeError('native matrix layout check failed')

    def close(self):
        if not self.closed:
            self.api.lib.nmod_mat_clear(C.byref(self.mat))
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class NativeLU:
    def __init__(self, N, p, api=None):
        if N < 1 or p != 998244353:
            raise ValueError('production dense LU is scoped to N>=1 and F998244353')
        self.api, self.N, self.p = api or get_api(), N, p
        self.storage = OwnedMat(self.api, N, N, p)
        self.perm = (S * N)()
        self.factorization_calls = 0
        self.ready = False

    @property
    def mat(self):
        return self.storage.mat

    @classmethod
    def from_rows(cls, rows, p=998244353, api=None):
        out = cls(len(rows), p, api)
        try:
            for i, row in enumerate(rows):
                if len(row) != out.N:
                    raise ValueError('square matrix required')
                for j, value in enumerate(row):
                    out.mat.entries[i * out.mat.stride + j] = int(value) % p
            return out
        except BaseException:
            out.close()
            raise

    @classmethod
    def from_problem(cls, problem, api=None):
        if problem['field'].get('kind') != 'prime':
            raise ValueError('prime field required')
        p, n = int(problem['field']['p']), int(problem['dimensions']['n'])
        kind, rows = problem['raw']['kind'], problem['raw']['rows']
        if kind not in ('banded', 'halfstrip'):
            raise ValueError('unsupported raw matrix kind')
        width = problem['dimensions']['k' if kind == 'banded' else 'w']
        expected_rows = 2*width+1 if kind == 'banded' else n+width
        if n < 1 or width < 0 or len(rows) != expected_rows:
            raise ValueError('invalid physical dimensions or lag count')
        out = cls(n*n, p, api)
        try:

            
            blocks = []
            for row in rows:
                if len(row) != 2*n-1:
                    raise ValueError('invalid lag row length')
                blocks.append((U * (n*n))(*(int(row[n-1+a-b]) % p
                                           for a in range(n) for b in range(n))))
            dest = C.addressof(out.mat.entries.contents)
            for t in range(n):
                for d, block in enumerate(blocks):
                    u = t - d + width
                    if 0 <= u < n:
                        source = C.addressof(block)
                        for a in range(n):
                            offset = ((t*n+a) * out.mat.stride + u*n) * C.sizeof(U)
                            C.memmove(dest + offset, source + a*n*C.sizeof(U), n*C.sizeof(U))
            return out
        except BaseException:
            out.close()
            raise

    def factorize(self):
        if self.factorization_calls:
            raise RuntimeError('refusing repeated LU of one fixture')
        self.factorization_calls += 1

        result = int(self.api.lib.nmod_mat_lu(self.perm, C.byref(self.mat), 1))
        if result != self.N:
            raise SingularPhysicalMatrix('native LU rank check refused physical matrix')
        self.ready = True

    def solve(self, vectors, transpose=False):
        if not self.ready:
            raise RuntimeError('successful factorization required')
        vectors = list(vectors)
        if any(len(v) != self.N for v in vectors):
            raise ValueError('wrong RHS length')
        R = len(vectors)
        if not R:
            return []

        
        with OwnedMat(self.api, self.N, R, self.p) as B, \
             OwnedMat(self.api, self.N, R, self.p) as Y, \
             OwnedMat(self.api, self.N, R, self.p) as X:
            for i in range(self.N):
                source = i if transpose else self.perm[i]
                for j, v in enumerate(vectors):
                    B.mat.entries[i*B.mat.stride+j] = int(v[source]) % self.p
            lib = self.api.lib
            if transpose:

                
                lib.nmod_mat_transpose(C.byref(self.mat), C.byref(self.mat))
                try:
                    lib.nmod_mat_solve_tril(C.byref(Y.mat), C.byref(self.mat), C.byref(B.mat), 0)
                    lib.nmod_mat_solve_triu(C.byref(X.mat), C.byref(self.mat), C.byref(Y.mat), 1)
                finally:
                    lib.nmod_mat_transpose(C.byref(self.mat), C.byref(self.mat))
            else:
                lib.nmod_mat_solve_tril(C.byref(Y.mat), C.byref(self.mat), C.byref(B.mat), 1)
                lib.nmod_mat_solve_triu(C.byref(X.mat), C.byref(self.mat), C.byref(Y.mat), 0)
            answer = [[0]*self.N for _ in range(R)]
            for i in range(self.N):
                target = self.perm[i] if transpose else i
                for j in range(R):
                    answer[j][target] = int(X.mat.entries[i*X.mat.stride+j])
            return answer

    def determinant(self):
        if not self.ready:
            raise RuntimeError('successful factorization required')
        result = permutation_sign(list(self.perm))
        for i in range(self.N):
            result = result * int(self.mat.entries[i*self.mat.stride+i]) % self.p
        return result % self.p

    def close(self):
        self.ready = False
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

