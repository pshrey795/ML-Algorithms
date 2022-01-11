import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys

#Cost function 
def cost(a):
    m = np.size(a)
    return np.matmul(a.T,a)/m

#Gradient function for Newton's Formula
def gradient(x,y,t):
    hypothesis = (-1) * np.matmul(x,t)
    e = np.exp(hypothesis)
    e = 1/(1+e) 
    diff = y - e
    return np.matmul(x.T,diff)

#Auxiliary function for Hessian calculation
def calc(x,t):
    hypothesis = (-1) * np.matmul(x,t)
    m = hypothesis.shape[0]
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

        #Parameter update by Newton's Method
        theta = theta + diff
        k = cost(diff)
        nextCost = k[0][0]
        itr += 1

    return theta, itr
    
#Plotting the given data and the separator learned in training
def plot_logistic_curve(x1,x2,y,theta):

    #Separating the given examples corresponding to different binary values
    x1_0 = x1 * (1-y)
    x2_0 = x2 * (1-y)
    x1_1 = x1 * y
    x2_1 = x2 * y

    #Generating points for plotting the learned separator
    x_pred = np.linspace(-2,2,100)
    y_pred = (-1) * (theta[0] + theta[1]*x_pred) / (theta[2])

    #Scatter plots for 0 and 1
    plt.scatter(x1_0,x2_0,c="r",label="0")
    plt.scatter(x1_1,x2_1,c="g",label="1")

    #Line plot for the separator
    plt.plot(x_pred,y_pred,c="b",label="Decision Boundary")
    plt.xlabel("x\u2081")
    plt.xlabel("x\u2082")
    plt.legend()
    plt.savefig("Graph.png")
    plt.close()

def main():

    plotCurve = False

    #Setting up matrices/vectors for logistic regression
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

    #Resultant values after training(Q-3A)
    theta, iterations = newtonMethod(x,y,epsilon)
    print("Final value of learned parameters: "+str(theta))
    print("Total number of iterations taken: "+str(iterations))

    #Plotting(Q-3B)
    if(plotCurve):
        plot_logistic_curve(x1,x2,y,theta)

if __name__ == "__main__":
    main()


 