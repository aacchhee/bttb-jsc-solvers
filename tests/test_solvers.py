# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small exact checks against independently built dense FLINT matrices."""
import random
import unittest
from flint import nmod_mat
import bttb_solvers as s
from bttb_solvers.matrices import dense_matrix,action,banded_to_halfstrip


def banded(n=4,k=1,p=s.PRIME):
    rng=random.Random(17)
    rows=[[rng.randrange(p) for _ in range(2*n-1)] for _ in range(2*k+1)]
    rows[0][:n-1]=[0]*(n-1);rows[0][n-1]=1;rows[0][n]=1
    return rows


class Solvers(unittest.TestCase):
    def check_prepared(self,prepared,rows,n,w,kind='banded',p=s.PRIME):
        A=dense_matrix(rows,n,w,kind=kind,p=p)
        b=list(range(1,n*n+1))
        with prepared as q:
            x=q.solve(b);y=q.solve_transpose(b)
            ref=[int(v) for v in A.solve(nmod_mat(n*n,1,b,p)).entries()]
            reft=[int(v) for v in A.transpose().solve(nmod_mat(n*n,1,b,p)).entries()]
            self.assertEqual(x,ref);self.assertEqual(y,reft)
            self.assertEqual(action(rows,n,w,x,kind=kind,p=p),b)
            self.assertEqual(action(rows,n,w,y,kind=kind,p=p,transpose=True),b)
            return q

    def test_banded_routes(self):
        rows=banded()
        for backend in ('python','flint'):
            for fn in (s.prepare_cc,s.prepare_cc_r,s.prepare_bfm,s.prepare_gbe):
                with self.subTest(algorithm=fn.__name__,backend=backend):
                    self.check_prepared(fn(rows,4,1,backend=backend),rows,4,1)
        self.check_prepared(s.prepare_bfm(rows,4,1,product='strassen'),rows,4,1)

    def test_halfstrip_routes(self):
        rows=banded_to_halfstrip(banded(),4,1)
        for fn in (s.prepare_hir,s.prepare_hpr):
            with self.subTest(algorithm=fn.__name__):
                q=fn(rows,4,1)
                self.assertEqual(q.determinant(),int(dense_matrix(rows,4,1,kind='halfstrip').det()))
                self.check_prepared(q,rows,4,1,kind='halfstrip')
        for fn in (s.solve_hsc,s.solve_hsc_q0):
            with self.subTest(algorithm=fn.__name__):
                b=list(range(1,17));result=fn(rows,4,1,b)
                self.assertEqual(action(rows,4,1,result['solution'],kind='halfstrip'),b)
                self.assertEqual(result['determinant'],int(dense_matrix(rows,4,1,kind='halfstrip').det()))

    def test_failed_leading_block(self):
        rows=banded_to_halfstrip(banded(),4,1)
        rows[0]=[0]*7;rows[0][4]=1
        self.check_prepared(s.prepare_hpr(rows,4,1),rows,4,1,kind='halfstrip')
        result=s.solve_hsc(rows,4,1,list(range(1,17)))
        self.assertEqual(action(rows,4,1,result['solution'],kind='halfstrip'),list(range(1,17)))
        with self.assertRaises((ValueError,ArithmeticError)):
            s.solve_hsc_q0(rows,4,1,list(range(1,17)))
        with self.assertRaises(Exception):s.prepare_hir(rows,4,1)

    def test_zero_determinant(self):
        rows=[[0]*7 for _ in range(5)]
        for fn in (s.solve_hsc,s.solve_hsc_q0):
            if fn is s.solve_hsc_q0:continue
            r=fn(rows,4,1)
            self.assertEqual(r['determinant'],0);self.assertTrue(r['singular'])

    def test_dense_lu_reuses_factorization(self):
        rows=banded()
        q=s.prepare_dense_lu(rows,4,1)
        self.assertEqual(q.state.factorization_calls,1)
        b=list(range(1,17))
        self.assertEqual(q.solve_many([b,b]),[q.solve(b),q.solve(b)])
        self.assertEqual(q.solve_many([b,b],transpose=True),[q.solve_transpose(b),q.solve_transpose(b)])
        self.assertEqual(q.state.factorization_calls,1)
        self.assertEqual(q.determinant(),int(dense_matrix(rows,4,1).det()))
        self.check_prepared(q,rows,4,1)
        with self.assertRaises(RuntimeError):q.solve(b)

    def test_extension_routes(self):
        rows=banded(n=3)
        F=s.extension_field(s.PRIME,2);root=s.root_of_unity(F,3)
        for fn in (s.prepare_cc_e,s.prepare_cc_e_r):
            with self.subTest(algorithm=fn.__name__):
                self.check_prepared(fn(rows,3,1,field=F,root=root),rows,3,1)

    def test_dimension_guards(self):
        with self.assertRaises(ValueError):s.prepare_cc([[0]*3]*3,2,1)
        with self.assertRaises(ValueError):s.prepare_bfm([[0]*7]*3,4,0)
        with self.assertRaises(ValueError):s.prepare_hpr([[0]*7]*3,4,1)


if __name__=='__main__':unittest.main()
