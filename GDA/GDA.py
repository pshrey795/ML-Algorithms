import math
from operator import matmul
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys

from numpy.lib.function_base import cov

def GaussianDiscriminantAnalysis(x,y):

    m = x.shape[0]

    y1 = y
    sum1 = np.sum(y)
    y0 = 1 - y
    sum0 = m - sum1

    mean0 = np.matmul(x.T,y0) / sum0
    mean1 = np.matmul(x.T,y1) / sum1

    phi = sum1/m   
    var = x - np.matmul(y0,mean0.T) - np.matmul(y1,mean1.T)
    var0 = var * y0
    var1 = var * y1
    cov = np.matmul(var.T,var)/m
    cov0 = np.matmul(var0.T,var0)/sum0
    cov1 = np.matmul(var1.T,var1)/sum1

    return phi, mean0, mean1, cov0, cov1, cov
    

def main():

    #Setting up matrices/vectors for linear regression
    x1 = np.loadtxt("q4x.dat", ndmin=2, usecols=0)
    x2 = np.loadtxt("q4x.dat", ndmin=2, usecols=1)
    y_inp = np.loadtxt("q4y.dat", dtype=str, ndmin=2) 
    y = np.array([],ndmin=2)

    for s in y_inp:
        if s=="Alaska":
            y = np.append(y,np.array([0],ndmin=2),axis=1)
        else:
            y = np.append(y,np.array([1],ndmin=2),axis=1)

    y = y.T

    #Normalising the data
    x1 -= np.mean(x1)
    x1 /= np.sqrt(np.var(x1))
    x2 -= np.mean(x2)
    x2 /= np.sqrt(np.var(x2))
    x = np.append(x1,x2,axis=1)

    #Setting up parameters for learning
    epsilon = 1e-10

    phi, mu0, mu1, sigma0, sigma1, sigma = GaussianDiscriminantAnalysis(x,y)

    print(phi)
    print(mu0)
    print(mu1)
    print(sigma0)
    print(sigma1)
    print(sigma)

if __name__ == "__main__":
    main()


 