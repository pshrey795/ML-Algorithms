from os import error
from mpl_toolkits.mplot3d.axes3d import Axes3D
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

#Gradient calculation
def gradient(x,y,t):
    diff = y - np.matmul(x,t)
    m = np.size(y)
    return np.matmul(x.T,diff)/m

#Linear Regression using Stochastic Gradient Descent
def stochasticGradientDescent(x,y,epsilon,eta,batch_size):
    #Inserting intercept term
    m = x.shape[0]

    #For tracking the parameter values and the corresponding error values throughout the course of the algorithm
    theta = np.zeros((x.shape[1],1))
    thetaVector = [np.zeros((x.shape[1],1)).T[0]]

    #Initial cost values
    prevCost = -1.0
    nextCost = cost(theta,x,y)

    #Batch details
    epoch = 0
    batch_num = 0

    #Convergence when cost between consecutive epochs changes less than epsilon, always checked at the end
    #of epoch for consistency
    while(abs(nextCost-prevCost)>epsilon or prevCost<0):
        prevCost = nextCost
        batch_num = 0
        nextCost = 0

        #Updates using round-robin fashion
        while batch_num<m:
            i = batch_num
            j = min(i+batch_size,m)
            actual_size = min(batch_size,m-i)

            #Parameter update using the current batch
            theta += eta * gradient(x[i:j],y[i:j],theta)
            thetaVector.append(np.copy(theta).T[0])

            #Updated cost value
            nextCost += cost(theta,x[i:j],y[i:j])

            #Updating the batch counter
            batch_num += batch_size
        
        epoch += 1
        nextCost /= m

    return np.array(thetaVector), epoch, nextCost, cost(theta,x,y)

#Testing the model on given test data for a given batch value
def testModel(theta):

    #Input Data
    x_test_1 = np.loadtxt("q2test.csv",delimiter=",",ndmin=2, skiprows=1, usecols=0)
    x_test_2 = np.loadtxt("q2test.csv",delimiter=",",ndmin=2, skiprows=1, usecols=1)
    y_test = np.loadtxt("q2test.csv",delimiter=",",ndmin=2, skiprows=1, usecols=2)

    #Error Calculation for given theta, which depends on batch_size
    x_test = np.ones((y_test.shape[0],1))
    x_test = np.append(x_test,x_test_1,axis=1)
    x_test = np.append(x_test,x_test_2,axis=1)
    error = cost(theta,x_test,y_test)
    print(error)

#Plotting the movement of parameters as a function of number of iterations
def trackMovement(thetaVector):

    #New 3D co-ordinate system
    fig = plt.figure()
    ax = plt.axes(projection='3d',xlim=(-1,4),ylim=(-2,4),zlim=(-3,3))

    #Sub plot for animating the movement of paramaters
    animPlot, = ax.plot([],[],[],color="red",lw=2,label="Path of parameters")

    #Frame update function
    def nextFrame(i):
        currTheta = thetaVector[0:i].T
        animPlot.set_data(currTheta[0],currTheta[1])
        animPlot.set_3d_properties(currTheta[2])
        return animPlot,

    #Function for animation
    animate = anim.FuncAnimation(fig,nextFrame,frames=thetaVector.shape[0],interval=0.001,repeat=False,blit=True)
    nextFrame(thetaVector.shape[0])
    
    plt.title("Movement of Parameters")
    ax.set_xlabel("theta\u2080")
    ax.set_ylabel("theta\u2081")
    ax.set_zlabel("theta\u2082")
    ax.legend()
    plt.savefig("Graph_1000000.png")
    plt.show()
    plt.close(fig)

def main():

    doTest = False
    plotCurve = False

    #Sampling 1 million data points(Q-2A)
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

    #Resultant parameters after training(Q-2B)
    thetaVector, epoch, finalError, finalCost = stochasticGradientDescent(x,y,epsilon,eta,batch_size)

    theta = thetaVector[-1]
    print("Final values of theta: "+str(theta))
    print("Total number of epochs: "+str(epoch))
    print("Final value of error: "+str(finalError))
    print("Final value of cost: "+str(finalCost))

    #Testing the trained model on test data(Q-2C)
    if(doTest):
        testModel(theta)

    #Plotting(Q-2D)
    if(plotCurve):
        trackMovement(thetaVector)

if __name__ == "__main__":
    main()





