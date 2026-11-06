# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from .exact import (pzeros, peye, pmat, pcut, transpose, clean, sa, ss,
                      sn, scale, shift)
from .poly import sm, sinv, pmm


def series_dot(F,a,b,lo=None,hi=None):
    out={}
    for x,y in zip(a,b):out=sa(F,out,sm(F,x,y,lo,hi),lo,hi)
    return out


def valuation(a,cap):
    return min((k for k in a if 0<=k<cap),default=cap)


def module_rows(F,G,N):
    """Select <=width original rows spanning G over K[z]/z^N.

    Rectangular minimum-valuation elimination retains full N precision:
    quotient ambiguity z^(N-v) multiplies a pivot row divisible by z^v.
    Only width rounds and O(width^2 len(G)) series operations are used.
    Original selected rows span because all row subtractions have analytic
    coefficients; already selected pivots can reconstruct each residual row.
    """
    if not G:return []
    width=len(G[0]);A=[[clean(F,x,0,N) for x in row] for row in G]
    remaining=list(range(len(G)));columns=list(range(width));selected=[]
    while remaining and columns:
        best=None
        for i in remaining:
            for j in columns:
                v=valuation(A[i][j],N)
                if best is None or (v,i,j)<best:best=(v,i,j)
        v,i,j=best
        if v==N:break
        selected.append(i);remaining.remove(i);columns.remove(j)
        inverse=sinv(F,shift(A[i][j],-v),N-v)
        for r in remaining:
            quotient=sm(F,shift(A[r][j],-v),inverse,0,N-v)
            for k in columns:
                A[r][k]=ss(F,A[r][k],sm(F,quotient,A[i][k],0,N),0,N)
            A[r][j]={}
    return selected


def collision_list(c):
    return [{'row':i,'column':j,'value':x} for (i,j),x in sorted(c.items())]


def joint_compiler(F,c,n,w,*,validate=False):
    L=n+w;h=n*w;P=L+3*h;q=c['q'];V=c['V'];BA=c['BA'];S=c['S'];ext=list(range(n+q));active=list(range(n+q,n+2*q));labels=c['selected_labels'];op=[None]*n+[F.inv(x) for x in labels]*2
    if not q:
        return {'P':P,'q':0,'delta':[],'k':0,'Us':[],'Vs':[],'dU':F.one(),'dV':F.one(),'B':[[] for _ in range(n)],'G':[],'W':pcut(F,c['R'],0,L),'W_nonzero':pcut(F,c['R'],0,L),'pivots':[],'depth_terms':[],'overcap':False,'labels':op,'coverage':{'unequal_label_pivot':False,'nonzero_lift_tail':False}}
    A=list(reversed(c['H']));GA=pzeros(n,2);RA=pzeros(n,2)
    GA[0][0]={0:F.one()};RA[0][1]={0:F.one()}
    for i in range(1,n):GA[i][1]=A[i-1][0];RA[i][0]=sn(F,A[0][i-1])

    
    VtBA=c.get('VtBA')
    if VtBA is None:VtBA=transpose(c['BV'])
    EBA=BA+pzeros(q,n)+VtBA
    tGA=pmm(F,EBA,GA,0,P);tRA=pmm(F,EBA,RA,0,P)
    GG=pzeros(n+2*q,4);HH=pzeros(n+2*q,4)
    for i in range(n+2*q):
        GG[i][0]=sn(F,tGA[i][0]);GG[i][1]=sn(F,tGA[i][1]);GG[i][3]=sn(F,EBA[i][0]);HH[i][0]=tRA[i][0];HH[i][1]=tRA[i][1];HH[i][2]=EBA[i][0]
    for i in range(q):GG[n+q+i][2]={0:op[n+q+i]};HH[n+q+i][3]={0:op[n+q+i]}

    collisions={}
    for j in range(n):collisions[(0,j)]=dict(BA[0][j])
    for i in range(q):
        p=n+i;a=n+q+i
        collisions[(p,p)]={};collisions[(p,a)]={0:F.one()};collisions[(a,p)]={0:F.one()};collisions[(a,a)]=sn(F,S[i][i])
    initial={'GA':GA,'RA':RA,'G':[[dict(x) for x in r] for r in GG],'Hgen':[[dict(x) for x in r] for r in HH],'collisions':collision_list(collisions)} if validate else None
    rows=list(active);cols=list(active);Csave=[];Dsave=[];delta=[];pivots=[];K=d=0;unequal=False;dU=F.one();dV=F.one();oracle_rows=0
    def entry(i,j,precision):
        if (i,j) in collisions:return clean(F,collisions[(i,j)],None,precision)
        if i>=n and j>=n:
            v=series_dot(F,GG[i],HH[j],None,precision)
            diff=F.sub(op[i],op[j])
            if F.is_zero(diff):raise ArithmeticError('missing collision')
            return scale(F,v,F.inv(diff))
        raise ArithmeticError('entry requires cross recurrence')
    def colcross(j,ids,precision):
        out={};rhs=[series_dot(F,GG[i],HH[j],None,precision) for i in range(n)];beta=op[j];bi=F.inv(beta);following={}
        for i in range(n-1,-1,-1):
            out[i]=scale(F,ss(F,following,rhs[i],None,precision),bi);following=out[i]
        for i in ids:
            if i>=n:out[i]=entry(i,j,N if i in rows else precision)
        return out
    def rowcross(i,ids,precision):
        out={};alpha=op[i];ai=F.inv(alpha);following={}
        for j in range(n-1,-1,-1):
            rhs=series_dot(F,GG[i],HH[j],None,precision);out[j]=scale(F,sa(F,rhs,following,None,precision),ai);following=out[j]
        for j in ids:
            if j>=n:out[j]=entry(i,j,N if j in cols else precision)
        return out
    while rows:
        N=P-K;Rprec=N-d;best=None

        
        selected=module_rows(F,[GG[i] for i in rows],N)
        oracle_rows+=len(selected)
        for ri in selected:
            i=rows[ri]
            for cj,j in enumerate(cols):
                if F.eq(op[i],op[j]):continue
                e=entry(i,j,N);v=valuation(e,L)
                if best is None or (v,ri,cj)<best[:3]:best=(v,ri,cj,e)
        rowpos={i:r for r,i in enumerate(rows)};colpos={j:s for s,j in enumerate(cols)}
        for (i,j),e in collisions.items():
            if i not in rowpos or j not in colpos:continue
            ri=rowpos[i];cj=colpos[j];v=valuation(e,L)
            if best is None or (v,ri,cj)<best[:3]:best=(v,ri,cj,clean(F,e,0,N))
        if best is None:best=(L,0,0,{})
        nu,ri,cj,m=best;lower=K+len(rows)*nu
        if lower>h:
            return {'P':P,'q':q,'overcap':True,'lower_bound':lower,'K':K,'remaining':len(rows),'minimum':nu,'pivots':pivots,'initial':initial,'labels':op,'coverage':{'unequal_label_pivot':unequal,'nonzero_lift_tail':False}}
        if nu==L:
            for i,j in zip(rows,cols):
                C=colcross(j,ext,Rprec);D=rowcross(i,ext,Rprec)
                Csave.append([clean(F,C[t],0,Rprec) for t in ext]);Dsave.append([clean(F,D[t],0,Rprec) for t in ext]);delta.append(L)
            K+=len(rows)*L;break
        i=rows[ri];j=cols[cj];a=sn(F,m);u=shift(a,-nu);u_inv=sinv(F,u,N-nu);a_inv=shift(u_inv,-nu)
        allrows=ext+rows;allcols=ext+cols;C=colcross(j,allrows,Rprec);D=rowcross(i,allcols,Rprec)
        negatives=[{'side':side,'id':t,'coefficients':{e:v for e,v in x.items() if e<0}} for side,seq in [('C',C),('D',D)] for t,x in seq.items() if any(e<0 for e in x)]
        if negatives:raise ArithmeticError('forbidden negative cross coefficients')

        g=[dict(x) for x in GG[i]];hh=[dict(x) for x in HH[j]]
        newK=K+nu;newd=max(d,nu);newN=P-newK;newR=newN-newd
        if validate:
            beforeG=[[dict(x) for x in r] for r in GG];beforeH=[[dict(x) for x in r] for r in HH];beforeColl=collision_list(collisions)
        if ri%2:dU=F.neg(dU)
        if cj%2:dV=F.neg(dV)
        dU=F.mul(dU,u_inv.get(0,F.zero()))
        Csave.append([clean(F,C[t],0,Rprec) for t in ext]);Dsave.append([sm(F,u_inv,D[t],0,min(Rprec,N-nu)) for t in ext]);delta.append(nu)
        for r in allrows:
            if r==i:continue
            precision=newR if r in ext else newN
            scaled_cross=sm(F,C[r],a_inv,None,precision)
            for k in range(4):GG[r][k]=sa(F,GG[r][k],sm(F,scaled_cross,g[k],None,precision),None,precision)
        for s in allcols:
            if s==j:continue
            precision=newR if s in ext else newN
            scaled_cross=sm(F,D[s],a_inv,None,precision)
            for k in range(4):HH[s][k]=sa(F,HH[s][k],sm(F,scaled_cross,hh[k],None,precision),None,precision)
        for key in list(collisions):
            r,s=key
            if r==i or s==j:del collisions[key];continue
            precision=newN if r in rows and s in cols else newR
            collisions[key]=sa(F,collisions[key],sm(F,sm(F,C[r],a_inv,None,precision),D[s],None,precision),None,precision)

        
        unequal=unequal or not F.eq(op[i],op[j])
        if validate:
            pivot={'row_id':i,'col_id':j,'row_position':ri,'col_position':cj,'nu':nu,'K_before':K,'d_before':d,'N':N,'Rprec':Rprec,'a':a,'u_inv':u_inv,'C':[[t,C[t]] for t in allrows],'D':[[t,D[t]] for t in allcols],'g':g,'h':hh,'G_before':beforeG,'Hgen_before':beforeH,'collisions_before':beforeColl,'G_after':[[dict(x) for x in r] for r in GG],'Hgen_after':[[dict(x) for x in r] for r in HH],'collisions_after':collision_list(collisions),'precision_after':{'K':newK,'d':newd,'N':newN,'Rprec':newR},'forbidden_negative':negatives,'unequal_labels':not F.eq(op[i],op[j])}
            pivots.append(pivot)
        else:
            pivots.append({'row_id':i,'col_id':j,'nu':nu,
                           'row_position':ri,'col_position':cj})
        rows.remove(i);cols.remove(j);K,d=newK,newd

    
    B=[[Csave[j][i] for j in range(q)] for i in range(n)];Vs=[[Csave[j][n+i] for j in range(q)] for i in range(q)]
    Us=[[Dsave[i][n+j] for j in range(q)] for i in range(q)];G=[list(reversed(Dsave[i][:n])) for i in range(q)]
    dmax=max(delta,default=0);lift=L+dmax;B=pcut(F,B,0,lift);G=pcut(F,G,0,lift)

    
    X=pzeros(n,n)
    for j in range(n):X[0][j]=clean(F,collisions[(0,j)],None,L)
    for i in range(1,n):
        for j in range(n):
            Dentry=series_dot(F,GG[i-1],HH[j],None,L)
            X[i][j]=sa(F,X[i-1][j+1] if j<n-1 else {},Dentry,None,L)
    Wnz=[list(reversed(r)) for r in X];W=[[dict(x) for x in r] for r in Wnz];terms=[]
    for ch,de in enumerate(delta):
        if de!=L:continue
        term=pzeros(n,n)
        for r in range(n):
            for s in range(n):term[r][s]=shift(sm(F,B[r][ch],G[ch][s],0,2*L),-L);W[r][s]=sa(F,W[r][s],term[r][s],-dmax,L)
        if validate:terms.append({'channel':ch,'coefficients':term})
    tail=any(k>=L for mat in [B,G] for row in mat for x in row for k in x)
    return {'P':P,'q':q,'overcap':False,'k':K,'delta':delta,'Us':Us,'Vs':Vs,'dU':dU,'dV':dV,'B':B,'G':G,'W_nonzero':(Wnz if validate else None),'depth_terms':terms,'W':W,'pivots':pivots,'initial':initial,'final_generators':({'G':GG,'Hgen':HH,'collisions':collision_list(collisions)} if validate else None),'oracle_rows':oracle_rows,'labels':op,'coverage':{'unequal_label_pivot':unequal,'nonzero_lift_tail':tail}}

