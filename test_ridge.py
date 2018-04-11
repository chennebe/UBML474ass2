import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    rmse = (np.sum(np.square(np.subtract(ytest,np.dot(Xtest, w)))))/ Xtest.shape[0]
    #print(rmse)
    return rmse
    
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    xTranspose = np.matrix.transpose(X)

    first = np.matmul(xTranspose, X)

    sec = np.linalg.matrix_power(first, -1)

    third = np.matmul(sec, xTranspose)

    w = np.matmul(third, y)                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD   
    
    """w = np.ones([X[0,:].size, 1])
    Xw = np.matmul(X,w)
    y_Xw = y - Xw
    y_Xw_t = np.transpose(y_Xw)
    left = (.5) * np.matmul(y_Xw_t, y_Xw)
    
    w_t = np.transpose(w)
    w_t_w = np.matmul(w_t, w)
    right = (lambd/2.) * w_t_w
    
    Jw = left + right
                                                                                              
    return Jw"""
    
    X_t = np.transpose(X)
    I = np.identity(X[0,:].size)
    lI = I * lambd
    X_t_X = np.matmul(X_t, X)
    par = lI + X_t_X
    left = np.linalg.inv(par)
    right = np.matmul(X_t, y)
    w = np.matmul(left, right)
    
    return w
    
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
    
jkl = np.argmin(mses3)
print(jkl)
print(mses3[6])
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()