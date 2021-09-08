import numpy as np

x = np.loadtxt("linearX.csv", delimiter=",")
x -= np.mean(x)
x /= np.sqrt(np.var(x))
y = np.loadtxt("linearY.csv", delimiter=",")

theta = np.array([0,0])
eta = 0.001
epsilon = 1e-20
n = np.size(x)

def sum(t):
    acc = 0
    for i in range(0,n):
        error = y[i] - np.dot(t,np.array([x[i],1]))
        acc += (error*error)/(2*n)
    return acc

k = 0
prevSum = -1
nextSum = sum(theta)
while abs(prevSum-nextSum)>epsilon or prevSum<0 :
    prevSum = nextSum
    t = theta
    for i in range(0,n):
        x_new = np.array([x[i],1.0])
        error = eta * (y[i] - np.dot(t,x_new))
        theta = np.add(theta,error * x_new)
    nextSum = sum(theta)
    print(nextSum)
    k += 1

print(k)
print(theta)