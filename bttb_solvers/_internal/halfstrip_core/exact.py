# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Main HSC and HPR series/mosaic implementation.

Role: Shared. Native guarded field operations, elementary elimination, scalar polynomials and sparse Laurent-polynomial matrix helpers.

Navigation: docs/CODE_GUIDE.md and docs/SOLVERS.md (repository root).
"""
from fractions import Fraction
from itertools import product

class Restart(Exception):
    def __init__(self, expression, factor, guard_index):
        self.expression, self.factor, self.guard_index = expression, factor, guard_index
        super().__init__('proper coefficient-algebra restriction')

class PrimeField:
    def __init__(self,p): self.p=p; self.guards=0; self.inverse_guards=[]
    def zero(self): return 0
    def one(self): return 1
    def from_int(self,x): return x%self.p
    def add(self,a,b): return (a+b)%self.p
    def sub(self,a,b): return (a-b)%self.p
    def neg(self,a): return -a%self.p
    def mul(self,a,b): return a*b%self.p
    def is_zero(self,a): self.guards+=1; return a%self.p==0
    def inv(self,a):
        if self.is_zero(a): raise ZeroDivisionError
        self.inverse_guards.append({'denominator':a,'unit':True})
        return pow(a,self.p-2,self.p)
    def eq(self,a,b): return self.is_zero(self.sub(a,b))
    def polynomial_leaf(self,a,b):
        """Exact fixed-64 convolution; integer digits cannot carry.

        This is an adapter optimization of the multiplier's fixed leaf, not
        a growing schoolbook cutoff or an assumption about roots in F_p.
        """
        if not a or not b:return []
        if max(len(a),len(b))>64:raise ValueError('fixed polynomial leaf exceeded')
        return self._packed_product(a,b)
    def polynomial_native(self,a,b):
        """Bounded native accelerator, not the generic recursive cutoff.

        The deliberate fixed 4096-coefficient bound changes constants only;
        beyond it the caller uses the uniform all-field route. Leaf stays 64.
        """
        if not a or not b:return []
        if max(len(a),len(b))>4096:return None
        return self._packed_product(a,b)
    def _packed_product(self,a,b):
        bits=max(1,(min(len(a),len(b))*(self.p-1)**2).bit_length())
        aa=bb=0
        for x in reversed(a):aa=(aa<<bits)+(x%self.p)
        for x in reversed(b):bb=(bb<<bits)+(x%self.p)
        product=aa*bb;mask=(1<<bits)-1;out=[]
        for _ in range(len(a)+len(b)-1):
            out.append((product&mask)%self.p);product>>=bits
        return out
    def info(self,include_inversions=True): return {'kind':'prime','p':self.p,'guards':self.guards,'inverse_guards':list(self.inverse_guards) if include_inversions else {'count':len(self.inverse_guards)}}

class RationalField:
    p=0
    def __init__(self): self.guards=0; self.inverse_guards=[]
    def zero(self): return Fraction(0)
    def one(self): return Fraction(1)
    def from_int(self,x): return Fraction(x)
    def add(self,a,b): return a+b
    def sub(self,a,b): return a-b
    def neg(self,a): return -a
    def mul(self,a,b): return a*b
    def is_zero(self,a): self.guards+=1; return a==0
    def inv(self,a):
        if self.is_zero(a): raise ZeroDivisionError
        self.inverse_guards.append({'denominator':a,'unit':True})
        return 1/a
    def eq(self,a,b): return self.is_zero(self.sub(a,b))
    def info(self,include_inversions=True): return {'kind':'rational','guards':self.guards,'inverse_guards':list(self.inverse_guards) if include_inversions else {'count':len(self.inverse_guards)}}

def trim(F,a):
    a=list(a)
    while a and F.is_zero(a[-1]): a.pop()
    return a

def pa(F,a,b):
    out=[F.zero() for _ in range(max(len(a),len(b)))]
    for i in range(len(out)): out[i]=F.add(a[i] if i<len(a) else F.zero(),b[i] if i<len(b) else F.zero())
    return trim(F,out)
def ps(F,a,b): return pa(F,a,[F.neg(x) for x in b])
def pm(F,a,b):
    if not a or not b:return []
    out=[F.zero() for _ in range(len(a)+len(b)-1)]
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]=F.add(out[i+j],F.mul(x,y))
    return trim(F,out)
def pd(F,a,b):
    a=trim(F,a);b=trim(F,b)
    if not b:raise ZeroDivisionError
    q=[F.zero() for _ in range(max(0,len(a)-len(b)+1))];bi=F.inv(b[-1])
    while len(a)>=len(b):
        k=len(a)-len(b); c=F.mul(a[-1],bi);q[k]=c
        for j,x in enumerate(b):a[k+j]=F.sub(a[k+j],F.mul(c,x))
        a=trim(F,a)
    return trim(F,q),a
def pgcd(F,a,b):
    a=trim(F,a);b=trim(F,b)
    while b:a,b=b,pd(F,a,b)[1]
    if not a:return []
    c=F.inv(a[-1]);return [F.mul(x,c) for x in a]
def pxgcd(F,a,b):
    r0,r1=trim(F,a),trim(F,b);s0,s1=[F.one()],[]
    while r1:
        q,r=pd(F,r0,r1);r0,r1=r1,r;s0,s1=s1,ps(F,s0,pm(F,q,s1))
    c=F.inv(r0[-1]);return [F.mul(x,c) for x in r0],[F.mul(x,c) for x in s0]

class EtaleField:
    def __init__(self,base,modulus):
        self.base=base; self.modulus=trim(base,modulus);self.degree=len(self.modulus)-1
        self.p=base.p;self.guards=0;self._classification={};self._inverse={};self._products={};self.inverse_guards=[]
    def reduce(self,a):
        a=pd(self.base,list(a),self.modulus)[1] if len(a)>=len(self.modulus) else trim(self.base,a)
        return tuple(a+[self.base.zero()]*(self.degree-len(a)))
    def zero(self):return (self.base.zero(),)*self.degree
    def one(self):return self.embed(self.base.one())
    def embed(self,a):return (a,)+(self.base.zero(),)*(self.degree-1)
    def from_int(self,a):return self.embed(self.base.from_int(a))
    def add(self,a,b):return tuple(self.base.add(x,y) for x,y in zip(a,b))
    def sub(self,a,b):return tuple(self.base.sub(x,y) for x,y in zip(a,b))
    def neg(self,a):return tuple(self.base.neg(x) for x in a)
    def mul(self,a,b):
        key=(a,b)
        try:hash(key)
        except TypeError:return self.reduce(pm(self.base,a,b))
        if key not in self._products:self._products[key]=self.reduce(pm(self.base,a,b))
        return self._products[key]
    def is_zero(self,a):
        self.guards+=1;a=self.reduce(a)
        if all(self.base.is_zero(x) for x in a):return True
        try:hash(a);cacheable=True
        except TypeError:cacheable=False
        if cacheable:
            if a not in self._classification:self._classification[a]=pgcd(self.base,a,self.modulus)
            g=self._classification[a]
        else:g=pgcd(self.base,a,self.modulus)
        if len(g)>1:raise Restart(a,g,self.guards)
        return False
    def inv(self,a):
        if self.is_zero(a):raise ZeroDivisionError
        self.inverse_guards.append({'denominator':a,'unit':True,'gcd':[self.base.one()]})
        try:hash(a)
        except TypeError:
            g,s=pxgcd(self.base,a,self.modulus);return self.reduce(s)
        if a not in self._inverse:
            g,s=pxgcd(self.base,a,self.modulus);self._inverse[a]=self.reduce(s)
        return self._inverse[a]
    def eq(self,a,b):return self.is_zero(self.sub(a,b))
    def info(self,include_inversions=True):return {'kind':'etale','base':self.base.info(False),'modulus':self.modulus,'degree':self.degree,'guards':self.guards,'inverse_guards':list(self.inverse_guards) if include_inversions else {'count':len(self.inverse_guards)}}

def power(F,a,n):
    out=F.one()
    while n:
        if n&1:out=F.mul(out,a)
        a=F.mul(a,a);n//=2
    return out

def irreducible_modulus(p,e):
    """Frozen lexicographic monic enumeration and exhaustive divisor certificate."""
    F=PrimeField(p); attempts=[]
    for c in product(range(p),repeat=e):
        f=list(c)+[1];witness=None
        for d in range(1,e//2+1):
            for v in product(range(p),repeat=d):
                g=list(v)+[1]
                if not pd(F,f,g)[1]:witness=g;break
            if witness:break
        attempts.append({'candidate':f,'divisor':witness})
        if witness is None:return f,attempts
    raise ArithmeticError('finite irreducible enumeration exhausted')

def base_factory(name):
    if name=='Q':return RationalField()
    if name=='F4':return EtaleField(PrimeField(2),[1,1,1])
    return PrimeField(int(name[1:]))

def zeros(F,m,n):return [[F.zero() for _ in range(n)] for _ in range(m)]
def eye(F,n):
    a=zeros(F,n,n)
    for i in range(n):a[i][i]=F.one()
    return a
def transpose(a):return [list(r) for r in zip(*a)] if a else []
def mm(F,a,b):
    if not a:return []
    n=len(b[0]) if b else 0;out=zeros(F,len(a),n)
    for i,row in enumerate(a):
        for k,x in enumerate(row):
            if F.is_zero(x):continue
            for j,y in enumerate(b[k]):out[i][j]=F.add(out[i][j],F.mul(x,y))
    return out

def echelon(F,a,augment=None):
    a=[list(r)+(list(augment[i]) if augment is not None else []) for i,r in enumerate(a)]
    m=len(a);n=len(a[0])-(len(augment[0]) if augment else 0) if m else 0
    piv=[];det=F.one();swaps=0;r=0
    for j in range(n):
        k=next((k for k in range(r,m) if not F.is_zero(a[k][j])),None)
        if k is None:continue
        if k!=r:a[k],a[r]=a[r],a[k];swaps+=1
        pivot=a[r][j];det=F.mul(det,pivot);c=F.inv(pivot)
        a[r]=[F.mul(x,c) for x in a[r]]
        for k in range(m):
            if k==r:continue
            c=a[k][j]
            if F.is_zero(c):continue
            a[k]=[F.sub(x,F.mul(c,y)) for x,y in zip(a[k],a[r])]
        piv.append(j);r+=1
        if r==m:break
    if swaps%2:det=F.neg(det)
    return a,piv,det

def determinant(F,a):
    if not a:return F.one()
    _,p,d=echelon(F,a)
    return d if len(p)==len(a) else F.zero()
def inverse_and_determinant(F,a):
    """One charged elimination supplies both inverse and determinant."""
    if not a:return [],F.one()
    out,p,d=echelon(F,a,eye(F,len(a)))
    if len(p)!=len(a):raise ZeroDivisionError('singular matrix')
    return [r[len(a):] for r in out],d

def inverse(F,a):
    if not a:return []
    out,p,_=echelon(F,a,eye(F,len(a)))
    if len(p)!=len(a):raise ZeroDivisionError('singular matrix')
    return [r[len(a):] for r in out]
def kernel(F,a):
    if not a:return []
    out,p,_=echelon(F,a);n=len(a[0]);free=[j for j in range(n) if j not in p];basis=[]
    for j in free:
        v=[F.zero() for _ in range(n)];v[j]=F.one()
        for i,k in enumerate(p):v[k]=F.neg(out[i][j])
        basis.append(v)
    return transpose(basis)


def clean(F,a,lo=None,hi=None):
    return {k:v for k,v in a.items() if (lo is None or k>=lo) and (hi is None or k<hi) and not F.is_zero(v)}
def sa(F,a,b,lo=None,hi=None):
    c=dict(a)
    for k,v in b.items():c[k]=F.add(c.get(k,F.zero()),v)
    return clean(F,c,lo,hi)
def sn(F,a):return {k:F.neg(v) for k,v in a.items()}
def ss(F,a,b,lo=None,hi=None):return sa(F,a,sn(F,b),lo,hi)
def sm(F,a,b,lo=None,hi=None):
    c={}
    for i,x in a.items():
        for j,y in b.items():
            k=i+j
            if (lo is not None and k<lo) or (hi is not None and k>=hi):continue
            c[k]=F.add(c.get(k,F.zero()),F.mul(x,y))
    return clean(F,c,lo,hi)
def scale(F,a,c):return clean(F,{k:F.mul(v,c) for k,v in a.items()})
def shift(a,k):return {i+k:v for i,v in a.items()}
def sinv(F,a,precision):
    a0=a.get(0,F.zero());iv=F.inv(a0);out={0:iv}
    for k in range(1,precision):
        s=F.zero()
        for j,x in a.items():
            if 0<j<=k:s=F.add(s,F.mul(x,out.get(k-j,F.zero())))
        v=F.neg(F.mul(iv,s))
        if not F.is_zero(v):out[k]=v
    return out

def pzeros(m,n):return [[{} for _ in range(n)] for _ in range(m)]
def peye(F,n):
    a=pzeros(n,n)
    for i in range(n):a[i][i]={0:F.one()}
    return a
def pmat(F,a):return [[({0:x} if not F.is_zero(x) else {}) for x in row] for row in a]
def pmm(F,a,b,lo=None,hi=None):
    if not a:return []
    n=len(b[0]) if b else 0;out=pzeros(len(a),n)
    for i,row in enumerate(a):
        for k,x in enumerate(row):
            if not x:continue
            for j,y in enumerate(b[k]):
                if y:out[i][j]=sa(F,out[i][j],sm(F,x,y,lo,hi),lo,hi)
    return out
def pcut(F,a,lo=None,hi=None):return [[clean(F,x,lo,hi) for x in row] for row in a]
def coefficient(F,a,k):return [[x.get(k,F.zero()) for x in row] for row in a]
def pack(value):
    if isinstance(value,Fraction):return {'Q':[value.numerator,value.denominator]}
    if isinstance(value,tuple):return {'E':[pack(x) for x in value]}
    if isinstance(value,list):return [pack(x) for x in value]
    if isinstance(value,dict):return {str(k):pack(v) for k,v in value.items()}
    return value
def unpack(value):
    if isinstance(value,list):return [unpack(x) for x in value]
    if isinstance(value,dict):
        if set(value)=={'Q'}:return Fraction(*value['Q'])
        if set(value)=={'E'}:return tuple(unpack(x) for x in value['E'])
        return {k:unpack(v) for k,v in value.items()}
    return value
