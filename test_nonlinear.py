import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    
    x_t = np.reshape(x, (x.size,1))
    #print(x_t)
    Xp = np.repeat(x_t, p+1, 1)
    for i in range(p+1):
        Xp[:,i] = np.power(Xp[:,i], i)
        
    print(Xp.shape)
    return Xp
    
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')
    
x = np.array([[2],
              [3]])
p = 7
print(Xtest[:,2])
print(mapNonLinear(Xtest[:,2],p))