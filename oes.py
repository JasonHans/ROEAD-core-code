#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from math import exp, log, sqrt
import numpy as np

# parameters
L = 4.  # learning rate
G = 40.  # smoothing parameter for adaptive learning rate
delta = 1000  # L1 regularization, 
D = 0  # number of weights to use


class oes(object):
    ''' Our main algorithm: OES
    '''

    def __init__(self, L, G, delta, D):
        # parameters
        self.L = L
        self.G = G
        self.delta = delta
        # feature related parameters
        self.D = D

        # model
        # z: weights
        # w: lazy weights
        self.z = np.zeros(D)
        self.w = np.zeros(D) 

    def predict(self, x, iter):
        ''' 

            INPUT:
                x: features

            OUTPUT:
                wTx
        '''

        # parameters
        L = self.L
        G = self.G
        delta = self.delta

        # model
        D = self.D
        z = self.z
        w = self.w

        # wTx is the inner product of w and x
        wTx = 0.
        for i in range(D):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]


            if sign * z[i] <= (1/delta):
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                w[i] =(L/(G*sqrt(iter))) * (sign * (1/delta) - z[i])
        wTx = np.dot(x, w)
        # cache the current w for update stage
        self.w = w
        return wTx

    def predictY(self, p):
        if p > 0:
            y_pred = 1
        else:
            y_pred = -1
        return y_pred

    def update(self, x, y_pred, y, iter):
        ''' Update model using x, p, y

            INPUT:


            MODIFIES:
                self.z: weights
        '''

        # parameter
        L = self.L

        # model
        z = self.z
        w = self.w
        D = self.D

        # update z
        if y_pred == y:
            for i in range(D):
                g = 0  # gradient 计算
                z[i] += g - ((G * (sqrt(iter) - sqrt(iter - 1))) / L) * w[i]
        else:
            for i in range(D):
                g = -y * x[i]  # gradient 计算
                z[i] += g - ((G*(sqrt(iter)-sqrt(iter-1)))/L) * w[i]

    def log_loss(self, y_pred, y, wTx):
        ''' FUNCTION: Bounded loss
        '''
        if y_pred == y:
            loss = 0
        else:
            loss = 1 - y * wTx
        return loss


