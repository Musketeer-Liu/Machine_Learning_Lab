from __future__ import division


import sys
import math
import numpy as np




def calculate_entropy(y):
    'Return the entroy of label y'
    log2 = lambda x: math.log(x) / math.log(2)
    
    entropy = 0
    unique_labels = np.unique(y)
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)

    return entropy


def mean_squared_error(y_true, y_pred):
    'Return mean squared error between prediction and true value'
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance(X):
    'Return variance of the features X'
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance


def calculate_std_dev(X):
    'Return standard_deviation of features X'
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(v1, v2):
    'Return l2 distance between 2 vectors'
    distance = 0
    for i in range(len(v1)):
        distance += pow((v1[i] - v2[i]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    'Return accuracy after comparing prediction and true value'
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_covariance_matrix(X, M=None):
    'Return covariance matrix for dataset X'
    if M is None: M = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(M - M.mean(axis=0))
    return np.array(covariance_matrix, dtype = float)


def calculate_correlation_matrix(X, M=None):
    'Return correlation matrix for dataset X'
    if M is None: M = X

    covariance = calculate_covariance_matrix(X)
    X_std_dev = np.expand_dims(calculate_std_dev(X), 1)
    y_std_dev = np.expand_dims(calculate_std_dev(M), 1)

    correlation_matrix = np.divide(covariance, X_std_dev.dot(y_std_dev.T))
    return np.array(correlation_matrix, dtype=float)


