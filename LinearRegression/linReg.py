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
    return np.matmul(x.T,diff)

def gradientDescent(x,y,epsilon,eta):
    #Inserting intercept term
    x = np.insert(x,0,1,axis=1)
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))
    theta1 = np.array([0])
    theta0 = np.array([0])

    #Training using Batch Gradient Descent
    prevCost = -1.0
    nextCost = cost(theta,x,y)
    costVector = np.array([nextCost])
    itr = 0

    #Convergence when cost between consecutive iterations changes less than epsilon
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost

        #Parameter update using gradient update
        theta += eta * gradient(x,y,theta) / m
        
        itr += 1
        nextCost = cost(theta,x,y)

    return theta, itr, nextCost
    

def main():

    #Setting up matrices/vectors for linear regression
    x = np.loadtxt("linearX.csv", delimiter=",", ndmin=2)
    y = np.loadtxt("linearY.csv", delimiter=",", ndmin=2)

    #Normalising the data
    x -= np.mean(x)
    x /= np.sqrt(np.var(x))

    #Setting up parameters for learning
    epsilon = 1e-15
    eta = 1e-3

    theta, iterations, finalCost = gradientDescent(x,y,epsilon,eta)

    #Final paramter values
    print(theta)                
    print(iterations)                  #Number of iterations
    print(finalCost)                   #Final cost

    #Comparing given data and predicted data (1(b))
    # x_p = x.T[0].reshape((100,1))
    # plt.scatter(x_p,y,c="r",label="Given Data")
    # plt.plot(x_p,(np.matmul(x,theta)),c="b",label="Hypothesis Prediction")
    # plt.title("Given v/s Predicted")
    # plt.xlabel("Acidity")
    # plt.ylabel("Wine Density")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()


 