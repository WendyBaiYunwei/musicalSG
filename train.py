from keras import backend as K
from keras_preprocessing import sequence
from keras.layers import *
from keras.models import Model
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

def train():
    train = np.load('dataset/train.npy', allow_pickle=True)

    x = np.zeros((1801 * 185, 32))
    y = np.zeros((1801 * 185))
    n = 0
    for i in range(1801):
        # pad the current song's bars
        for j in range(185):
            if j < len(train[i][1]) and 1 in train[i][1][j]:
                y[n] = train[i][1][j].index(1)
            else:
                y[n] = 24
            
            for k in range(32):
                if j < len(train[i][0]) and k < len(train[i][0][j]) and 1 in train[i][0][j][k]:
                    x[n][k] = train[i][0][j][k].index(1)
                else:
                    x[n][k] = 12
            n += 1

    train = np.load('dataset/test.npy', allow_pickle=True)

    test_x = np.zeros((train.shape[0] * 185, 32))
    test_y = np.zeros((train.shape[0] * 185))

    n = 0
    for i in range(train.shape[0]):
        for j in range(185):
            if j < len(train[i][1]) and 1 in train[i][1][j]:
                test_y[n] = train[i][1][j].index(1)
            else:
                test_y[n] = 24
            
            for k in range(32):
                if j < len(train[i][0]) and k < len(train[i][0][j]) and 1 in train[i][0][j][k]:
                    test_x[n][k] = train[i][0][j][k].index(1)
                else:
                    test_x[n][k] = 12
            n += 1

    clf = DecisionTreeClassifier()
    clf = clf.fit(x, y)
    y_pred = clf.predict(test_x)

    print("Done!")

    print(accuracy_score(test_y, y_pred))

if __name__ == '__main__':
    train()