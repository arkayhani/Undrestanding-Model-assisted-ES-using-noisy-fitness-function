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
def boundy2(x,l,r):
    return [max(-x,0),np.inf]
def boundy(x,l,r,z):
    return [-x,np.inf]
def cdf(x):
 #return np.exp(-x*t) / t**N
 return 0.5*(scipy.special.erf(x/math.sqrt(2.0))+1.0)
#def pq(x,l,r):
#    return (1/(np.sqrt(2.0*pi)*l))*(np.power(e,-(np.power((x+(np.power(l,2.0)/2.0))/l),2.0)/2.0))
def pq(x,l,r):
    return (1/(np.sqrt(2.0*pi)*l))*(np.power(e,-(np.power((x+(np.power(l,2.0)/2.0))/l,2.0)/2.0)))
def pq2(z,x,l,r):
    return (math.pow(e,-(np.power((x+(math.pow(l,2.0)/2.0))/l,2.0)/2.0)+(math.pow((z+(math.pow(l,2.0)/2.0))/l,2.0)/2.0)))
def pq3(z,x,l,r):
    return (np.power(e,0.5*(z-x)*(z+x+np.power(l,2))*(1/(np.power(l,2)))))

def pe(x,l,r):
    return (1.0/(np.sqrt(2.0*pi)*r))*(np.power(e,-(np.power(x/r,2.0)/2.0)))
    
def f1(y,x,l,r,z):
    return y*pq(y,l,r)*pe(x,l,r)/z
def f2(z,y,x,l,r):
    return (pe(x,l,r)*y*pq3(z,y,l,r))#*(1.0/cdf(y/r))#
def f3(y,l,r,z):
    return y*pq(y,l,r)/z
def f4(y,l,r):
    return pq(y,l,r)
def f5(y,l,r):
    return pq(y,l,r)
def fun2(y,l,r):
    return y*pq(y,l,r)
def fun4(y,x,l,r):
    return pe(x,l,r)*y*pq(y,l,r)
def fun1(x,l,r):
    return pq(x,l,r)
def fun1a(x,l,r):
    return (scipy.special.erf((2*np.inf+np.power(l,2))/(np.power(2,1.5)*l))/2)-(scipy.special.erf((2*(-x)+np.power(l,2))/(np.power(2,1.5)*l))/2)
def fun1a2(x,l,r):
    return ((scipy.special.erf((2*np.inf+np.power(l,2))/(np.power(2,1.5)*l))/2)-(scipy.special.erf((2*(max(0,-x))+np.power(l,2))/(np.power(2,1.5)*l))/2))
def fun3(x,l,r):
    return (pe(x,l,r)*(integrate.quad(fun1,-x,np.inf,args=(l,r))[0]))
def fun32(x,l,r):
    return (pe(x,l,r)*(integrate.quad(fun1,max(0,-x),np.inf,args=(l,r))[0]))
def fun3a(x,l,r):
    return (pe(x,l,r)*fun1a(x,l,r))
def fun3a2(x,l,r):
    return x*(pe(x,l,r)*fun1a2(x,l,r))
def fun3anew(x,l,r):
    return (l*x-np.power(l,2)/2)*np.power(e,-0.5*np.power(x,2))*cdf((l*x-np.power(l,2)/2)/r)
def fun2anew(x,l,r):
    return np.power(e,-0.5*np.power(x,2))*cdf((l*x-np.power(l,2)/2)/r)*(1.0/(np.sqrt(2*pi)))
def fun2anew2(x,l,r):
    return (l*x-np.power(l,2)/2)*np.power(e,-0.5*np.power(x,2))*cdf((l*x-np.power(l,2)/2)/r)*(1.0/(np.sqrt(2*pi)))
#def f(x,l,r):
# return (1/sqrt(2*pi))*(l*x-(math.pow(l,2)/2))*(math.pow(e,-(math.pow(x,2)/2)))*cdf((l*x-math.pow(l,2)/2)/r)
a=frange2(0.001,4.1,0.25)
b=frange2(0.001,8.1,1)
h=[]
h1=[]
h2=[]
h3=[]
prints=[]
prints2=[]
for j in b:  
        h=[]
        h1=[]
        h2=[]
        h3=[] 
        h4=[]         
        for i in a:
    #h.append(integrate.nquad(f3, [[0,np.inf],[-np.inf,np.inf]],args=(i,b))[0])
    #h.append(integrate.nquad(f1, [boundy,boundx],args=(i,b))[0])
    #z=(integrate.nquad(f5, [boundy,[-np.inf,0]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,+np.inf],[-np.inf,0]],args=(i,b,1))[0])
    #z1=(integrate.nquad(f5, [boundy,[0,np.inf]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,+np.inf],[0,np.inf]],args=(i,b,1))[0])
    #z1=(integrate.nquad(f5, [[0,np.inf],[0,np.inf]],args=(i,b,1))[0]/integrate.nquad(f5, [[-np.inf,np.inf],[0,np.inf]],args=(i,b,1))[0])
    #print(z)
    #h.append(integrate.quad(fun2, 0,np.inf,args=(i,b))[0])
            h.append((integrate.quad(fun2anew, -np.inf,np.inf,args=(i,j))[0]))#*(1.0/(np.sqrt(2*pi)*i)))
            #h1.append((integrate.quad(fun2anew, i/2,np.inf,args=(i,j))[0])*(1.0/(np.sqrt(2*pi)*i)))
            h4.append(cdf(-np.power(i,2)/(2*np.sqrt(np.power(i,2)+np.power(j,2)))))
            h2.append((integrate.quad(fun2anew2, i/2,np.inf,args=(i,j))[0])/(integrate.quad(fun2anew, -np.inf,np.inf,args=(i,j))[0]))
        prints.append(h)
        prints2.append(h2)
        #prints.append(h1)
        #prints.append(h2)
   # h3.append((integrate.nquad(fun4, [boundy2,[-np.inf,np.inf]],args=(i,b))[0])/(integrate.quad(fun3a, -np.inf,np.inf,args=(i,b))[0]))
    #h.append(integrate.quad(f3,0,np.inf,args=(i,b,z))[0])
   # h.append(integrate.nquad(f2, [boundy,boundy2,[-np.inf,np.inf]],args=(i,b))[0])
    #h.append(integrate.nquad(f1, [boundy,boundx],args=(i,b,z))[0])
plt.figure(3)
#plt.plot(a,prints[0],"*b",a,prints[1],"-*y",a,prints[2],"-*b",a,prints[3],"-*b",a,prints2[1],"-*r")
plt.plot(a,prints2[0],"-y",a,prints2[1],"-r",a,prints2[2],"-g",a,prints2[4],"-b",a,prints2[8],"-c")
print(h)
print(h1)
print(np.inf/5-np.inf)
print(-e)

