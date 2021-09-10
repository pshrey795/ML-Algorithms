import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys

#Cost function(MSE) i.e. J(theta)
def cost(t,x,y):
    diff = y - np.matmul(x,t)
    m = np.size(y)
    return (np.matmul(diff.T,diff))/(2*m)

def gradient(x,y,t):
    diff = y - np.matmul(x,t)
    m = np.size(y)
    return np.matmul(x.T,diff)/m

def stochasticGradientDescent(x,y,epsilon,eta,batch_size):
    #Inserting intercept term
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))

    #Training using Batch Gradient Descent
    prevCost = -1.0
    nextCost = cost(theta,x,y)

    #Batch details
    epoch = 0
    batch_num = 0

    #Convergence when cost between consecutive iterations changes less than epsilon
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost
        batch_num = 0
        nextCost = 0

        while batch_num<m:
            #Parameter update using gradient update
            i = batch_num
            j = min(i+batch_size,m)
            actual_size = min(batch_size,m-i)
            theta += eta * gradient(x[i:j],y[i:j],theta)
            nextCost += cost(theta,x[i:j],y[i:j])
            batch_num += batch_size
        
        epoch += 1
        nextCost /= m

    return theta, epoch, nextCost, cost(theta,x,y)
    

def main():

    #Sampling 1 million data points
    m = 1000000
    x1 = np.random.normal(3,2,m).reshape((m,1))
    x2 = np.random.normal(-1,2,m).reshape((m,1))
    x = np.ones((m,1))
    x = np.append(x,x1,axis=1)
    x = np.append(x,x2,axis=1)
    theta_init = np.array([[3],[1],[2]])
    y = np.matmul(x,theta_init) + np.random.normal(0,np.sqrt(2),m).reshape((m,1))

    #Setting up parameters for learning
    epsilon = 1e-10
    eta = 1e-3
    batch_size = 1000000

    theta, epoch, finalError, finalCost = stochasticGradientDescent(x,y,epsilon,eta,batch_size)

    print(theta)
    print(epoch)
    print(finalError)
    print(finalCost)


if __name__ == "__main__":
    main()





