from __future__ import division
from itertools import combinations_with_replacement


import sys
import math
import numpy as np




def shuffle_data(X, y, seed=None):
    'Random shuffle of X and y samples'
    if seed: np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
    'Batch Generator'
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        start, end = i, min(i+batch_size, n_samples)
        # for Supervised Learning
        if y: yield X[start:end], y[start:end]
        # for Unsupervised Learning
        else: yield X[start:end]


def divide_on_feature(X, feature_i, threshold):
    'Divide dataset based on wheter feature index >= given threshold'
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree+1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    
    combinations = index_combinations()
    n_output_features = len(combinations)
    # create X_new with random value,litte faster than np.zeros
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


def get_random_subsets(X, y, n_subsets, replacements=True):
    'Return radom subsets with replacements of the data'
    n_samples = np.shape(X)[0]
    # Concatenate X and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Use 100 of training samples with replacement or 50% without replacements
    subsample_size = n_samples if replacements else int(n_samples//2)
    
    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size = np.shape(range(subsample_size)),
            replace = replacements
        )
        X, y = X_y[idx][:, :-1], X_y[idx][:, -1]
        subsets.append([X, y])
    
    return subsets


def make_diagonal(X):
    'Convert a vector into an diagonal matrix'
    matrix = np.zeros(len(X), len(X))
    for i in range(len(matrix[0])):
        matrix[i, i] = X[i]
    return matrix


def normalize(X, axis=-1, order=2):
    'Normalize dataset X'
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2==0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    'Standardize the dataset X'
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    'Split the data into train and test sets'
    if shuffle: X, y = shuffle_data(X, y, seed)

    # Split train / test dataset according to test_size
    pivot = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[pivot:], X[:pivot]
    y_train, y_test = y[pivot:], y[:pivot]

    return X_train, X_test, y_train, y_test


def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    'Split the data into k sets of train-test data'
    if shuffle: X, y = shuffle_data(X, y)
    
    left_overs = {}
    n_samples = len(y)
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs["X"], left_overs["y"] = X[-n_left_overs:], y[-n_left_overs:]
        X, y = X[:-n_left_overs], y[:-n_left_overs]

    sets = []
    X_split, y_split = np.split(X, k), np.split(y, k)
    for i in range(k):
        X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i+1:], axis=0)
        X_test, y_test = X_split[i], y_split[i]
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets)


def to_categorical(X, n_col=None):
    'One-hot encoding of nominal values'
    if not n_col: n_col = np.amax(X) + 1
    
    one_hot = np.zeros((X.shape[0], n_col))
    one_hot[np.arange(X.shape[0]), X] = 1

    return one_hot


def to_nominal(X):
    'Conversion from one-hot encoding to nominal'
    return np.argmax(X, axis=1)








    