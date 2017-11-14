
import math
import random
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from scipy import *
from scipy.linalg import norm, pinv
from sklearn.cluster import KMeans 
from matplotlib import pyplot as plt
def frange2(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0
    else: start += 0.0 # force it to be a float

    if inc == None:
        inc = 1.0

    count = int((end - start) / inc)
    if start + count * inc != end:
        # need to adjust the count.
        # AFAIKT, it always comes up one short.
        count += 1

    L = [None,] * count
    for i in xrange(count):
        L[i] = start + i * inc

    return L
def sphere(x):
    sum=0
    for i in range(N):
        sum=sum+math.pow(x[i]-1,2)
    return sum
def distant(x):
    y=sphere(x)
    return math.pow(y,0.5)
def normalI():
    no=np.random.normal(0,np.ones(N))
    return no
def normalXI(x):
    
    no=np.random.normal(0,np.full(N,x,dtype=np.float))
    
    return no
def normalX(x):
    no=np.random.normal(0,x)
    return no
def normal():
    no=np.random.normal(0,1)
    return no
sigmarange=frange2(0.001,4.1,0.25)
h=[]
h1=[]
h2=[]
for varerror in frange2(3.001,3.5,1.5):
    for varsigma in sigmarange:
        p=0
        p1=0
        p2=0
        print(varsigma)
        for i in range(0,np.power(10,4)):
        
            c=0
            while(1):
                y2=varsigma*normal()-(np.power(varsigma,2)/2.0)
                y=y2-varerror*normal()
                c=c+1.0
                if(y>0):
                    break
            #rint(p/i)
            p=p+c
            if(y2>0):
                p1=p1+1
                p2=p2+y2
        p=p/np.power(10.0,4)    
        h.append(1.0/p)
        h1.append(p1/np.power(10.0,4))
        h2.append(p2/np.power(10.0,4))
        
plt.figure(4)

#plt.plot(sigmarange,h,'-sg',sigmarange,h1,'.g',sigmarange,h2,'*g')
plt.plot(sigmarange,h,'*g')