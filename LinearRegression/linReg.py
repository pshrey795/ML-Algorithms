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

#Gradient calculation
def gradient(x,y,t):
    diff = y - np.matmul(x,t)
    return np.matmul(x.T,diff)

#Linear Regression using gradient descent
def gradientDescent(x,y,epsilon,eta):

    #Number of examples
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))
    thetaVector = theta
    prevCost = -1.0
    nextCost = cost(theta,x,y)
    costVector = np.array([nextCost])
    itr = 0

    #Convergence when cost between consecutive iterations changes less than epsilon
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost

        #Parameter update using gradient update
        theta += eta * gradient(x,y,theta) / m
        thetaVector = np.append(thetaVector,theta,axis=1)
        itr += 1
        nextCost = cost(theta,x,y)
        costVector = np.append(costVector,nextCost)

    return thetaVector.T, costVector, itr, nextCost
    
#Plotting the given parameters and the linear hypothesis
def plot_curve(x,y,theta):
    x_p = x.T[1]
    plt.scatter(x_p,y,c="r",label="Given Data")
    plt.plot(x_p,(np.matmul(x,theta)),c="b",label="Hypothesis Prediction")
    plt.title("Given v/s Predicted")
    plt.xlabel("Acidity")
    plt.ylabel("Wine Density")
    plt.legend()
    plt.savefig("Graph.png")
    plt.close()

#Plotting the mesh/surface for the cost function and animating the movement of parameters
def plot_mesh(x,y,theta,costVector):
    frequency = 200
    t = theta[-1]
    a = np.linspace(t[0]-1.5,t[0]+1.5,frequency)
    b = np.linspace(t[1]-1.5,t[1]+1.5,frequency)
    X, Y = np.meshgrid(a,b)
    Z = X + Y
    for i in range(frequency):
        for j in range(frequency):
            Z[i][j] = cost(np.array([X[i][j],Y[i][j]]).reshape((2,1)),x,y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')  
    surf_plot = ax.plot_surface(X,Y,Z,label="Cost Function",alpha=0.5)
    surf_plot._facecolors2d = surf_plot._facecolor3d
    surf_plot._edgecolors2d = surf_plot._edgecolor3d

    animPlot, = ax.plot([],[],[],color="red",lw=2,label="Path of parameters")

    def nextFrame(i):
        currTheta = theta[0:i].T
        animPlot.set_data(currTheta[0],currTheta[1])
        animPlot.set_3d_properties(costVector[0:i])
        return animPlot,

    animate = anim.FuncAnimation(fig,nextFrame,frames=theta.shape[0],interval=0.2,repeat=False,blit=True)
    nextFrame(theta.shape[0])

    plt.title("Cost v/s Parameters(Learning Rate = ")
    ax.set_xlabel("theta\u2080")
    ax.set_ylabel("theta\u2081")
    ax.set_zlabel("Cost Value")
    ax.legend()
    plt.savefig("Graph.png")
    plt.show()
    plt.close(fig)


#Plotting the contour of the cost function and tracking the path of the paramaters
def plot_contour(x,y,theta):
    fig = plt.figure()

    frequency = 200
    a = np.linspace(-theta[-1][0],2*theta[-1][0],frequency)
    b = np.linspace(-theta[-1][1],2*theta[-1][1],frequency)
    X, Y = np.meshgrid(a,b)
    Z = X + Y
    for i in range(frequency):
        for j in range(frequency):
            Z[i][j] = cost(np.array([X[i][j],Y[i][j]]).reshape((2,1)),x,y) 
    plt.contour(X,Y,Z,levels=25)

    animPlot, = plt.plot([],[],color="black",label="Path of Parameters",lw = 2)

    def nextFrame(i):
        currTheta = theta[0:i].T
        animPlot.set_data(currTheta[0],currTheta[1])
        return animPlot,

    animate = anim.FuncAnimation(fig,nextFrame,frames=theta.shape[0],interval=0.2,repeat=False,blit=True)
    nextFrame(theta.shape[0])

    plt.title("Cost v/s Parameters(Learning Rate=0.01)")
    plt.xlabel("theta\u2080")
    plt.ylabel("theta\u2081")
    plt.savefig("Graph.png")
    plt.show()
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
    eta = 0.01

    plotCurve = False
    plotMesh = False
    plotCont = False

    #Final values of the parameter after training(Q-1A)
    thetaVector, costVector, iterations, finalCost = gradientDescent(x,y,epsilon,eta)
    theta = thetaVector[-1]
    print("Learning Rate: "+str(eta))
    print("Final parameter values: "+str(theta))                
    print("Total number of iterations taken for training: "+str(iterations))                  
    print("Final value of the cost function: "+str(finalCost))

    #Plotting

    #Plotting the given data and hypothesis(Q-1B)
    if(plotCurve):
        plot_curve(x,y,theta)

    #Plotting mesh and animating the movement of parameters(Q-1C)
    if(plotMesh):
        plot_mesh(x,y,thetaVector,costVector)

    #Plotting contours and animating the movement of parameters for different values of learning rate(Q-1D/E) 
    if(plotCont):
        plot_contour(x,y,thetaVector)

if __name__ == "__main__":
    main()