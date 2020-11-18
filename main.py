import logging
from sklearn import metrics
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse  # 处理稀疏矩阵
from scipy.sparse import csr_matrix
from math import exp, log, sqrt
import random
import time
from utils import load_data
from oes import oes
from _datetime import datetime


if __name__ == "__main__":
    # main()

    (x_train, y_train), (x_test, y_test) = load_data("hdfs")

    # parameters
    L = 4  # learning rate 
    G = 40.  # max feature vector entry, smoothing parameter for adaptive learning rate   
    delta = 1000  # 1/delta, L1 regularization, smaller value means more regularized
    D = x_train.shape[1]  # number of weights to use
	
    learner = oes(L, G, delta, D)
    w = np.random.randn(D).reshape(D, 1)  
    EPOCH = 3
    loss = 0
    count = 0
    count1 = 0
    y_predict = np.zeros(x_train.shape[0])
    y_test_pred = np.zeros(x_test.shape[0])


    for epoch in range(1, EPOCH):

        for i in range(x_train.shape[0]):
            p = learner.predict(x_train[i], i+1+(EPOCH-2)*x_train.shape[0])
            y_pred = learner.predictY((p))
            loss += learner.log_loss(y_pred, y_train[i], p)
            count += 1
            learner.update(x_train[i], y_pred, y_train[i], i+1+(EPOCH-2)*x_train.shape[0])
            y_predict[i] = y_pred
            if count % 10 == 0:
                #### Regret / T
                value = str(loss / count)
                #### Loss
                #value = str(loss)

                w = learner.w
                test_loss = 0
                for i in range(x_test.shape[0]):
                    p1 = learner.predict(x_test[i], i+1+(EPOCH-2)*x_train.shape[0])
                    y_pred1 = learner.predictY(p1)
                    y_test_pred[i] = y_pred1
                    test_loss += learner.log_loss(y_pred1, y_test[i], p1)
                    count1 += 1
                value2 = str(test_loss / count1)

    print("Train Precision:")
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_train, y_predict, average='weighted')
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

    print("Test Precision:")
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

