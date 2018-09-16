from __future__ import division
import numpy as np


from data_operation import accuracy_score




class Loss(object):
    # def __init__(self):
    #     pass

    def loss(self, y_true, y_pred):
        return NotImplementedError()
    
    def gradient(self, y_true, y_pred):
        return NotImplementedError()
    
    def accuracy(self, y_true, y_pred):
        return 0


class SquareLoss(Loss):
    # def __init__(self):
    #     pass
    
    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)
    
    def gradient(self, y_true, y_pred):
        return -(y_true - y_pred)


class SoftLoss(Loss):
    # def __init__(self):
    #     pass

    def gradient(self, y_true, y_pred):
        return y_true - y_pred


class CrossEntropy(Loss):
    # def __init__(self):
    #     pass

    def loss(self, y_true, y_pred):
        # Avoid divission by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)

    def accuracy(self, y_true, y_pred):
        return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    def gradient(self, y_true, y_pred):
        # Avoid divission by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true/y_pred + (1-y_true)/(1-y_pred)

