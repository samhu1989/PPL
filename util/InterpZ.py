from __future__ import print_function;
from __future__ import division;
import numpy as np;
import sys;
import math;
import scipy as sp;

def fequal(a,b,th=1e-2):
    return abs(a-b) < th;

class Point(object):
    def __init__(self, x=0, y=0, z = None):
        self.x = x;
        self.y = y;
        self.z = z;    
        
    def dist(p0,p1):
        return math.sqrt((p0.x - p1.x)*(p0.x-p1.x)+(p0.y - p1.y)*(p0.y-p1.y));

def interpZInBetween(pb,pm,pe):
    d1 = Point.dist(pb,pm);
    d2 = Point.dist(pe,pm);
    d3 = Point.dist(pb,pe);
    if fequal( d1 + d2 , d3 ):
        return ( pb.z*d1 + pe.z*d2 ) / ( d1 + d2 );
    return None;
        
class Line(object):
    def getCrossPoint(l1,l2):
        d = l1.a * l2.b - l2.a * l1.b;
        if fequal(d,0.0):
            return None;
        p = Point();
        p.x = (l1.b * l2.c - l2.b * l1.c)*1.0 / d;
        p.y = (l1.c * l2.a - l2.c * l1.a)*1.0 / d;
        return p;
        
    def getInterpCrossPoint(self,l2):
        p = Line.getCrossPoint(self,l2);
        if p is None:
            return None;
        zv = interpZInBetween(l2.p1,p,l2.p2);
        if zv is not None:
            p.z = zv;
            return p;
        return None;
        
    def __init__(self, p1, p2):
        self.p1 = p1;
        self.p2 = p2;
        self.__get_param__();
        self.length = Point.dist(self.p1,self.p2);

    def __get_param__(self):
        self.a = self.p1.y - self.p2.y;
        self.b = self.p2.x - self.p1.x;
        self.c = self.p1.x*self.p2.y - self.p2.x*self.p1.y;
    
def interpZ(fxyz,x,y):
    pm = Point(float(x),float(y));
    pa = Point(fxyz[0,0],fxyz[0,1],fxyz[0,2]);
    pb = Point(fxyz[1,0],fxyz[1,1],fxyz[1,2]);
    pc = Point(fxyz[2,0],fxyz[2,1],fxyz[2,2]);
    pd = Point(fxyz[3,0],fxyz[3,1],fxyz[3,2]);
    lam = Line(pa,pm);
    lbc = Line(pb,pc);
    lcd = Line(pc,pd);
    pambc = lam.getInterpCrossPoint(lbc);
    if pambc is not None:
        zv = interpZInBetween(pa,pm,pambc);
        if zv is not None:
            return zv;
    pamcd = lam.getInterpCrossPoint(lcd);
    if pamcd is not None:
        zv = interpZInBetween(pa,pm,pamcd);
        if zv is not None:
            return zv;
    return None;

def areaFromL(a,b,c):
    p = 0.5*(a+b+c);
    v = p*(p-a)*(p-b)*(p-c)
    if isinstance(v, float):
        return np.sqrt(v);
    else:
        v[v>0] = np.sqrt(v[v>0]);
        return v;
    
def areaFromV(A,B,x,y):
    a = np.sqrt( (A[0] - B[0])**2 + ( A[1] - B[1] )**2 );
    b = np.sqrt( (x-B[0])**2 + (y - B[1])**2 );
    c = np.sqrt( (x-A[0])**2 + (y - A[1])**2 );
    return areaFromL(a,b,c);
"""
def areaFromV(A,B,x,y):
    return np.abs((A[0] * B[1] + B[0] * y + x * A[1] - B[0] * y - x * B[1] - A[0] * y) / 2.0);
"""
def orderZ(fxy):
    zorder = np.zeros([8],np.float32);
    hull = sp.spatial.qhull.Delaunay(fxy).convex_hull;
    hullidx =  np.sort(np.unique(hull));
    zorder[hullidx] = np.array([ x for x in range(0,len(hullidx)) ],dtype=np.float32);
    c = np.mean(fxy[hullidx,:]);
    inidx = []; 
    for i in range(8):
        if i not in hullidx:
            inidx.append(i);
    dists = np.square(fxy[inidx,:] - c).sum(axis=1);
    inidx_order = np.argsort(dists);
    order = 7;
    for i in inidx_order:
        zorder[inidx[i]] = float(order);
        order -= 1;
    return zorder;

def fillZ(depth,i,fxyz,h,w):
    fz = fxyz[:,2];
    fxy = fxyz[:,0:2];
    y,x = np.mgrid[0:h,0:w];
    #print(depth.shape);
    A = fxy[0,:];
    B = fxy[1,:];
    C = fxy[2,:];
    D = fxy[3,:];
    aAMB = areaFromV(A,B,x,y);
    aBMC = areaFromV(B,C,x,y);
    aCMD = areaFromV(C,D,x,y);
    aDMA = areaFromV(D,A,x,y);
    aABC = areaFromV(A,B,C[0],C[1]);
    aCDA = areaFromV(A,D,C[0],C[1]);
    d = (aAMB+aBMC+aCMD+aDMA) - (aABC+aCDA);
    mask = fequal(aAMB+aBMC+aCMD+aDMA,aABC+aCDA,5.0);
    mZ = np.max(np.abs(fz));
    depth[i,mask] = mZ;