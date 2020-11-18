#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing


def load_data(opt="hdfs"):
    """Load train and test dataset

    Keywords arguments:
        opt: dataset name: hdfs

    Returns:
        (x_train, y_train): train data features and labels
        (x_test, y_test): test data features and labels
    """
    if opt == "hdfs":

        X = pd.read_csv("feature/x.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
        Y = pd.read_csv("feature/y.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
        x = X.values
        y = Y.values
        x = np.array(X)
        y = np.array(Y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    else:
        logging.error("Unknown dataset!!")

    logging.info("train data shape: {}".format(x_train.shape))
    logging.info("test data shape: {}".format(x_test.shape))
    return (x_train, y_train), (x_test, y_test)





