import math
from operator import matmul
from os import confstr
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
    
def plot_data(x1,x2,y):
    x1_0 = x1 * (1-y)
    x2_0 = x2 * (1-y)
    x1_1 = x1 * y
    x2_1 = x2 * y
    plt.scatter(x1_0,x2_0,c="r",label="Alaska")
    plt.scatter(x1_1,x2_1,c="g",label="Canada")
    plt.title("Original Data")
    plt.xlabel("x\u2081")
    plt.ylabel("x\u2082")
    plt.legend()
    plt.savefig("Graph(b).png")
    plt.close()

def plot_linear_separator(mu0,mu1,sigma,x1,x2,y):
    x1_0 = x1 * (1-y)
    x2_0 = x2 * (1-y)
    x1_1 = x1 * y
    x2_1 = x2 * y
    plt.scatter(x1_0,x2_0,c="r",label="Alaska")
    plt.scatter(x1_1,x2_1,c="g",label="Canada")
    sigma_inv = np.linalg.inv(sigma)
    constant = np.matmul(np.matmul(mu0.T,sigma_inv),mu0) - np.matmul(np.matmul(mu1.T,sigma_inv),mu1)
    linear = (2 * np.matmul(mu1.T-mu0.T,sigma_inv)).T
    x1_pred = np.linspace(-2,2,100)
    x2_pred = (-1) * (constant[0][0] + linear[0] * x1_pred) / (linear[1])
    plt.plot(x1_pred,x2_pred,c="b",label="Linear Separator")
    plt.title("Linear Separation in GDA")
    plt.xlabel("x\u2081")
    plt.ylabel("x\u2082")
    plt.legend()
    plt.savefig("Graph(c).png")

def plot_quad_separator(mu0,mu1,sigma0,sigma1,x1,x2,y):
    x1_0 = x1 * (1-y)
    x2_0 = x2 * (1-y)
    x1_1 = x1 * y
    x2_1 = x2 * y
    epsilon = 1e-2
    # plt.scatter(x1_0,x2_0,c="r",label="Alaska")
    # plt.scatter(x1_1,x2_1,c="g",label="Canada")
    det0 = np.linalg.det(sigma0)
    det1 = np.linalg.det(sigma1)
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    x1_pred = np.linspace(-3,3,1000)
    x2_pred = np.linspace(-3,3,1000)
    X1, X2 = np.meshgrid(x1_pred,x2_pred)
    constant = math.log(det1/det0) + np.matmul(np.matmul(mu1.T,sigma1_inv),mu1) - np.matmul(np.matmul(mu0.T,sigma0_inv),mu0)
    linear = 2 * (np.matmul(mu0.T,sigma0_inv) - np.matmul(mu1.T,sigma1_inv))
    quad = sigma1_inv - sigma0_inv
    x1_pred = np.array([])
    x2_pred = np.array([])
    for i in range(1000):
        for j in range(1000):
            x = np.array([X1[i][j], X2[i][j]]).reshape((2,1))
            k = (constant + np.matmul(linear,x) + np.matmul(np.matmul(x.T,quad),x))[0][0]
            if abs(k)<epsilon:
                x1_pred = np.append(x1_pred,X1[i][j])
                x2_pred = np.append(x2_pred,X2[i][j])
    plt.plot(x1_pred,x2_pred,c="blue")
    plt.title("Quadratic Separation in GDA")
    plt.xlabel("x\u2081")
    plt.ylabel("x\u2082")
    plt.legend()
    plt.savefig("Graph(e).png")
    plt.close()

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

    #Plotting
    plot_data(x1,x2,y)
    plot_linear_separator(mu0,mu1,sigma,x1,x2,y)
    plot_quad_separator(mu0,mu1,sigma0,sigma1,x1,x2,y)

if __name__ == "__main__":
    main()


 