import math
import random
import numpy as np
import scipy
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from scipy import *
from scipy.linalg import norm, pinv
from sklearn.cluster import KMeans 
from matplotlib import pyplot as plt
from scipy import integrate

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
def boundx(l,r,z):
    return [0,np.inf]
def boundy(x,l,r,z):
    return [-x,np.inf]
def cdf(x):
 #return np.exp(-x*t) / t**N
 return 0.5*(scipy.special.erf(x/math.sqrt(2.0))+1.0)
def pq(x,l,r):
    return (1/(sqrt(2.0*pi)*l))*(math.pow(e,-(math.pow((x+(math.pow(l,2.0)/2.0))/l,2.0)/2.0)))

def pe(x,l,r):
    return (1.0/(sqrt(2.0*pi)*r))*(math.pow(e,-(math.pow(x/r,2.0)/2.0)))
    
def f1(y,x,l,r,z):
    return y*pq(y,l,r)*pe(x,l,r)/z
def f2(y,x,l,r):
    return pe(x,l,r)*y*pq(y,l,r)/(integrate.quad(f5, -x,np.inf,args=(l,r))[0])#*(1.0/cdf(y/r))#
def f3(y,l,r,z):
    return y*pq(y,l,r)/z
def f4(y,l,r):
    return pq(y,l,r)
def f5(y,l,r):
    return pq(y,l,r)
#def f(x,l,r):
# return (1/sqrt(2*pi))*(l*x-(math.pow(l,2)/2))*(math.pow(e,-(math.pow(x,2)/2)))*cdf((l*x-math.pow(l,2)/2)/r)
a=1.001
b=frange2(0.001,100,0.25)
h=[]
for i in b:
    #h.append(integrate.nquad(f3, [[0,np.inf],[-np.inf,np.inf]],args=(i,b))[0])
    #h.append(integrate.nquad(f1, [boundy,boundx],args=(i,b))[0])
    #z=(integrate.nquad(f5, [boundy,[-np.inf,0]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,+np.inf],[-np.inf,0]],args=(i,b,1))[0])
    #z1=(integrate.nquad(f5, [boundy,[0,np.inf]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,+np.inf],[0,np.inf]],args=(i,b,1))[0])
    #z1=(integrate.nquad(f5, [[0,np.inf],[0,np.inf]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,np.inf],[0,np.inf]],args=(i,b,1))[0])
    #print(z)
    #h.append(integrate.quad(f3,0,np.inf,args=(i,b,z))[0])
    h.append(integrate.nquad(f2, [[0,np.inf],[-np.inf,np.inf]],args=(a,i))[0])
    #h.append(integrate.nquad(f1, [boundy,boundx],args=(i,b,z))[0])
plt.figure(2)
plt.plot(b,h,"--g")
print(h)
print(-e)