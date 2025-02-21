import numpy as np


def sigmoid(x):
    # the sigmoid function
    return 1.0/(1+np.e**(-x))
    pass


class LogisticReg(object):
    def __init__(self, indim=1):
            # initialize the parameters with all zeros
            # w: shape of [d+1, 1]
            self.indim=indim
            self.w=np.zeros(self.indim+1)
            pass
    
    def set_param(self, weights, bias):
        # helper function to set the parameters
        # NOTE: you need to implement this to pass the autograde.
        # weights: vector of shape [d, ]
        # bias: scaler
        self.w=np.hstack((np.array([bias]),weights))
        pass
    
    def get_param(self):
        # helper function to return the parameters
        # NOTE: you need to implement this to pass the autograde.
        # returns:
            # weights: vector of shape [d, ]
            # bias: scaler
        return self.w[1:],self.w[0]
        pass

    def compute_loss(self, X, t):
        # compute the loss
        # X: feature matrix of shape [N, d]
        # t: input label of shape [N, ]
        # NOTE: return the average of the log-likelihood, NOT the sum.
 
        # extend the input matrix

        # compute the loss and return the loss
        l=0
        for i in range(1,X.shape[0]+1):
           x=np.hstack((np.array([1]),X[i-1,:]))
           l=l-np.log(sigmoid(t[i-1]*(self.w @ x)))
        return l/X.shape[0] 
        pass


    def compute_grad(self, X, t):
        # X: feature matrix of shape [N, d]
        # grad: shape of [d, 1]
        # NOTE: return the average gradient, NOT the sum.
        l=0
        for i in range(1,X.shape[0]+1):
           x=np.hstack((np.array([1]),X[i-1,:]))
           l=l-(1-sigmoid(t[i-1]*(self.w @ x)))*t[i-1]*x
        return l/X.shape[0] 
        pass


    def update(self, grad, lr=0.001):
        # update the weights
        # by the gradient descent rule
        self.w=self.w-lr*grad
        pass


    def fit(self, X, t, lr=0.001, max_iters=1000, eps=1e-7):
        # implement the .fit() using the gradient descent method.
        # args:
        #   X: input feature matrix of shape [N, d]
        #   t: input label of shape [N, ]
        #   lr: learning rate
        #   max_iters: maximum number of iterations
        #   eps: tolerance of the loss difference 
        # TO NOTE: 
        #   extend the input features before fitting to it.
        #   return the weight matrix of shape [indim+1, 1]

        loss = 1e10
        for epoch in range(max_iters):
            # compute the loss 
            new_loss = self.compute_loss(X, t)

            # compute the gradient
            grad = self.compute_grad(X, t)

            # update the weight
            self.update(grad, lr=lr)

            # decide whether to break the loop
            if np.abs(new_loss - loss) < eps:
                return self.w


    def predict_prob(self, X):
        # implement the .predict_prob() using the parameters learned by .fit()
        # X: input feature matrix of shape [N, d]
        #   NOTE: make sure you extend the feature matrix first,
        #   the same way as what you did in .fit() method.
        # returns the prediction (likelihood) of shape [N, ]
        y=np.zeros(X.shape[0])
        for i in range(1,X.shape[0]+1):
           x=np.hstack((np.array([1]),X[i-1,:]))
           y[i-1]=sigmoid(self.w @ x)
           
        return y

        pass

    def predict(self, X, threshold=0.5):
        # implement the .predict() using the .predict_prob() method
        # X: input feature matrix of shape [N, d]
        # returns the prediction of shape [N, ], where each element is -1 or 1.
        # if the probability p>threshold, we determine t=1, otherwise t=-1
        y=self.predict_prob(X)
        t=np.ones(X.shape[0])
        t=-1*t
        t[y>threshold]=1
        return t
        pass
