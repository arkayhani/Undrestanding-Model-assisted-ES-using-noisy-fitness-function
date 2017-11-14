
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
sigmarange=frange2(2.001,2.1,0.25)
errorrange=frange2(0.501,10.1,0.25)
h=[]
for varerror in errorrange:
    for varsigma in sigmarange:
        p=0
        print(varerror)
        for i in range(0,100000):
            c=0
            while(1):
                c+=1
                y2=varsigma*normal()-(np.power(varsigma,2)/2.0)
                y=y2-varerror*normal()
                if(y>0):
                    break
            #if(y2>0):
                p=p+1.0/c
        h.append(p/100000.0)
        

plt.figure(2)
plt.plot(errorrange,h,'r*--')