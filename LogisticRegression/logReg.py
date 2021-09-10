import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys

def f(x):
    a = math.exp(x)
    b = (1+a)**2
    return a/b

def g(x):
    a = math.exp(x)
    return 1/(1+a)

def cost(a):
    m = np.size(a)
    return np.matmul(a.T,a)/m

def gradient(x,y,t):
    hypothesis = (-1) * np.matmul(x,t)
    #e = np.array([g(h) for h in hypothesis],ndmin=2)
    e = np.exp(hypothesis)
    e = 1/(1+e) 
    diff = y - e
    return np.matmul(x.T,diff)

def calc(x,t):
    hypothesis = (-1) * np.matmul(x,t)
    m = hypothesis.shape[0]
    #e = np.array([f(h) for h in hypothesis],ndmin=0)
    e = np.exp(hypothesis)
    e = e/(np.square(1+e))
    return np.diag(e.T[0])

def newtonMethod(x,y,epsilon):
    #Inserting intercept term
    x = np.insert(x,0,1,axis=1)
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))

    #Convergence when cost between consecutive iterations changes less than epsilon
    
    itr = 0
    prevCost = -1
    nextCost = 1
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost
        
        #Code for computing Hessian and difference for updating by Newton's Method 
        z = calc(x,theta)
        hessian = np.matmul(np.matmul(x.T,z),x)                   
        diff = np.matmul(np.linalg.inv(hessian),gradient(x,y,theta)) 
        theta = theta + diff
        k = cost(diff)
        nextCost = k[0][0]
        itr += 1

    return theta, itr
    

def main():

    #Setting up matrices/vectors for linear regression
    x1 = np.loadtxt("logisticX.csv", delimiter=",", ndmin=2, usecols=0)
    x2 = np.loadtxt("logisticX.csv", delimiter=",", ndmin=2, usecols=1)
    y = np.loadtxt("logisticY.csv", delimiter=",", ndmin=2)

    #Normalising the data
    x1 -= np.mean(x1)
    x1 /= np.sqrt(np.var(x1))
    x2 -= np.mean(x2)
    x2 /= np.sqrt(np.var(x2))
    x = np.append(x1,x2,axis=1)

    #Setting up parameters for learning
    epsilon = 1e-10

    theta, iterations = newtonMethod(x,y,epsilon)

    print(theta)
    print(iterations)

if __name__ == "__main__":
    main()


 