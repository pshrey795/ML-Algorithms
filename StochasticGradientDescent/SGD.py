import numpy as np
import matplotlib.pyplot as plt

#Sampling 1 million examples using normal distribution Q(2a)
x1 = np.random.normal(3,2,1e6)
x2 = np.random.normal(-1,2,1e6)
y = np.random.normal(4,np.sqrt(2),1e6)

#Setting up parameters for learning
theta = np.array([0,0,0])
eta = 0.01
epsilon = 1e-10
m = 1e6

