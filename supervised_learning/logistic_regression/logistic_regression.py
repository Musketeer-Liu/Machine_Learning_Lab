from __future__ import print_function, division


import math
import numpy as np


from utilities.data_manipulation import train_test_split 
from utilities.data_operation import mean_squared_error 
from utilities import plotting




def sigmoid(x):
    return  1/ (1 + np.exp(-x))


class LogisticRegression():
    """
    Logistic Regeression Classifer
    ----------
    Parameters:
    ----------
    n_iterations:        
        The number of training iterations
    learning_rate: float
        The step length that will be taken when following the negative gradient
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. 
        If false then we use batch optimization by least squares.
    """

    def __init__(self, n_iterations=4000, learning_rate=0.1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate


    def initialize_weights(self, n_features):
        # Setup parameter in [-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)
    

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)

        # Add a feature col x1, x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # gradient training for n_iterations round
        for i in range(self.n_iterations):
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_grad = X.T.dot(y_pred - y)
            self.w = self.w - self.learning_rate * w_grad

    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)


