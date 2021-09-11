import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits import mplot3d as plt3d
import sys

from numpy.core.defchararray import index

#Cost function(MSE) i.e. J(theta)
def cost(t,x,y):
    diff = y - np.matmul(x,t)
    m = np.size(y)
    return (np.matmul(diff.T,diff))/(2*m)

def gradient(x,y,t):
    diff = y - np.matmul(x,t)
    return np.matmul(x.T,diff)

def gradientDescent(x,y,epsilon,eta):
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))
    thetaVector = theta

    #Training using Batch Gradient Descent
    prevCost = -1.0
    nextCost = cost(theta,x,y)
    itr = 0

    #Convergence when cost between consecutive iterations changes less than epsilon
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost

        #Parameter update using gradient update
        theta += eta * gradient(x,y,theta) / m
        thetaVector = np.append(thetaVector,theta,axis=1)
        itr += 1
        nextCost = cost(theta,x,y)

    return thetaVector.T, itr, nextCost
    
def plot_curve(x,y,theta):
    x_p = x.T[1]
    plt.scatter(x_p,y,c="r",label="Given Data")
    plt.plot(x_p,(np.matmul(x,theta)),c="b",label="Hypothesis Prediction")
    plt.title("Given v/s Predicted")
    plt.xlabel("Acidity")
    plt.ylabel("Wine Density")
    plt.legend()
    plt.savefig("Graph(b).png")
    plt.close()

def plot_mesh(x,y,theta):
    frequency = 20
    a = np.linspace(-0.5,2,frequency)
    b = np.linspace(-1.5,1,frequency)
    X, Y = np.meshgrid(a,b)
    Z = X + Y
    for i in range(frequency):
        for j in range(frequency):
            Z[i][j] = cost(np.array([X[i][j],Y[i][j]]).reshape((2,1)),x,y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')  
    surf_plot = ax.plot_surface(X,Y,Z,label="Cost Function")
    surf_plot._facecolors2d = surf_plot._facecolor3d
    surf_plot._edgecolors2d = surf_plot._edgecolor3d
    plt.title("Cost v/s Parameters")
    ax.set_xlabel("theta\u2080")
    ax.set_ylabel("theta\u2081")
    ax.set_zlabel("Cost Value")
    ax.legend()
    plt.savefig("Graph(c).png")
    plt.close()

def plot_contour(x,y,theta):
    frequency = 20
    a = np.linspace(-0.5,2,frequency)
    b = np.linspace(-1.5,1,frequency)
    X, Y = np.meshgrid(a,b)
    Z = X + Y
    for i in range(frequency):
        for j in range(frequency):
            Z[i][j] = cost(np.array([X[i][j],Y[i][j]]).reshape((2,1)),x,y) 
    plt.contour(X,Y,Z)
    plt.plot(theta.T[0],theta.T[1],c="black")
    plt.title("Cost v/s Parameters")
    plt.xlabel("theta\u2080")
    plt.ylabel("theta\u2081")
    plt.savefig("Graph(d).png")
    plt.close()

def main():

    #Setting up matrices/vectors for linear regression
    x = np.loadtxt("linearX.csv", delimiter=",", ndmin=2)
    y = np.loadtxt("linearY.csv", delimiter=",", ndmin=2)

    #Normalising the data
    x -= np.mean(x)
    x /= np.sqrt(np.var(x))
    #Inserting intercept term
    x = np.insert(x,0,1,axis=1)

    #Setting up parameters for learning
    epsilon = 1e-10
    eta = 1e-3

    thetaVector, iterations, finalCost = gradientDescent(x,y,epsilon,eta)

    theta = thetaVector[-1]
    #Final paramter values
    print(theta)                
    print(iterations)                  #Number of iterations
    print(finalCost)                   #Final cost

    #Plotting
    plot_curve(x,y,theta)
    plot_mesh(x,y,thetaVector)
    plot_contour(x,y,thetaVector)

if __name__ == "__main__":
    main()