import numpy as np


class LinearReg(object):
    def __init__(self, indim=1, outdim=1):
        self.indim=indim
        self.outdim=outdim 
        self.W=np.zeros((self.indim+1, self.outdim))
        pass

    def fit(self, X, T):
        # implement the .fit() using the simple least-square closed-form solution:
        # W = (X^T X)^-1 X^T T,T是列向量,W直接就算出来了，迭代呢？这个模型误差很大
        phi=np.hstack((np.ones((X.shape[0],1)),X))
        self.W=np.linalg.inv(phi.T @ phi) @ phi.T @ T
        # HINT: 
        #   extend the input features before fitting to it.
        #   compute the weight matrix of shape [indim+1, outdim]
        pass

    def predict(self, X):
        # implement the .predict() using the parameters learned by .fit()
        phi=np.hstack([np.ones((X.shape[0],1)),X])
        return phi @ self.W #列向量
        pass


def second_order_basis(X):#X先二阶交叉，再正则化
    # we will perform a simple implementation
    # using the broadcasting mechanism in numpy.
    # HINT:
    #   np.triu_indices(): returns the indices of the upper triangular matrix
    XT=X.T
    n=np.triu_indices(X.shape[1])
    for i in range(1,X.shape[0]+1):
      Xa=XT[:,[i-1]] @ X[[i-1],:]
      m=Xa[n]
      if i==1 :
       X1=m
      else:
       X1=np.vstack((X1,m))
    return X1
    pass
