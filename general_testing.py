import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

x = np.array([[0.,1.,2.,3.],
              [4.,5.,6.,7.]])
y = np.array([[1.],
              [0.]])
lambd = 1.

def learnRidgeRegression(X,y,lambd):
    
    X_t = np.transpose(X)
    print(X)
    print(X_t)
    I = np.identity(X[0,:].size)
    lI = I * lambd
    X_t_X = np.matmul(X_t, X)
    par = lI + X_t_X
    print(par)
    left = np.linalg.inv(par)
    print(left)
    right = np.matmul(X_t, y)
    print(right)
    w = np.matmul(left, right)
    
    return w

print(learnRidgeRegression(x,y,lambd))