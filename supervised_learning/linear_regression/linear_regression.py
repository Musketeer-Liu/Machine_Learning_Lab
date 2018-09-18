from __future__ import print_function, division


import math
import numpy as np


from utilities.data_manipulation import train_test_split 
from utilities.data_operation import mean_squared_error 
from utilities import plotting




class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    # squared error
    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha*loss
    
    # gradient
    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)
    
    def grad(self, w):
        return self.alpha * w


class LinearRegression():
    """
    Parameters:
    ----------
    n_iterations: int
        The number of training iterations the algorithm will tune the weights for. 
    learning_rate: float
        The step length that will be used when updating the weights.
    regularization: l1_regularization or l2_regularization or None
        Regularization strategy    
    gradient: boolean
        True or false depending if gradient descent should be used when training.
    """

    def __init__(self, n_iterations=3000, learning_rate=0.00005, regularization=None, gradient=True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization: 
            self.regularization = self.regularization
        else: 
            self.regularization = lambda x: 0
            self.regularization.gradient = lambda x: 0


    def initialize_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)


    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.insert(y, (m_samples, 1))
        self.training_errors = []

        if self.gradient == True:
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                loss = np.mean(0.5 * (y_pred - y) ** 2) + self.regularization(self.w)
                self.training_errors.append(loss)
                # X.T.dot(y_pred - y) -- calculate gradient
                w_grad = X.T.dot(y_pred - y) + self.regularization.grad(self.w)
                self.w = self.w - self.learning_rate * w_grad
        else:
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.w = X_T_X_I_X_T_X_T_y


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self, w)
        return y_pred




