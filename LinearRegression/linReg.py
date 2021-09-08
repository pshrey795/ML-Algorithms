import numpy as np
import matplotlib.pyplot as plt

#Setting up matrices/vectors for linear regression
x = np.loadtxt("linearX.csv", delimiter=",", ndmin=2)

#Normalising the data
x -= np.mean(x)
x /= np.sqrt(np.var(x))
m = np.size(x)
x = np.insert(x,1,1,axis=1)
y = np.loadtxt("linearY.csv", delimiter=",", ndmin=2)

#Setting up parameters for learning
theta = np.array([[0.0],[0.0]], ndmin=2)
epsilon = 1e-15
eta = 1e-2

#Cost function(MSE) i.e. J(theta)
def cost(t):
    diff = y - np.matmul(x,t)
    acc = 0
    for i in range(m):
        acc += (diff[i]*diff[i])/(2*m)
    return acc

#Training using Batch Gradient Descent
prevCost = -1.0
nextCost = cost(theta)
itr = 0

#For tracking the parameters for plotting

#Convergence when cost between consecutive iterations changes less than epsilon
while(abs(nextCost-prevCost)>epsilon or prevCost<0):
    prevCost = nextCost

    #Parameter update using gradient update
    diff = (y - np.matmul(x,theta))
    for i in range(m):
        coeff = (eta*diff[i])/(m)
        theta += coeff * (x[i].T).reshape((2,1))
    
    nextCost = cost(theta)
    itr += 1

#Final paramter values
print(theta)                
print(itr)                  #Number of iterations
print(nextCost)             #Final cost

x_p = x.T[0].reshape((100,1))

#Comparing given data and predicted data (1(b))
plt.scatter(x_p,y,c="r",label="Given Data")
plt.plot(x_p,(np.matmul(x,theta)),c="b",label="Hypothesis Prediction")
plt.title("Given v/s Predicted")
plt.xlabel("Acidity")
plt.ylabel("Wine Density")
plt.legend()
plt.show()



 