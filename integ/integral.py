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
def cdf(x):
 #return np.exp(-x*t) / t**N
 return 0.5*(scipy.special.erf(x/math.sqrt(2))+1)
def f(x,l,r):
 return (1/sqrt(2*pi))*(l*x-(math.pow(l,2)/2))*(math.pow(e,-(math.pow(x,2)/2)))*cdf((l*x-math.pow(l,2)/2)/(r))
a=1.001
b=frange2(0.001,100,0.5)
h=[]
for i in b:
    h.append(integrate.quad(f, a/2, np.inf,args=(a,i))[0])
plt.figure(2)
plt.plot(b,h,"--r")