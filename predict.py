# the numpy array X to predict must be in "./dataset/X.npy"
# models must be in "./models/RFC.pkl" or "./models/LSTM.pkl"
# in command line terminal, call "python predict.py A B" eg. "python predict.py LSTM 1"
# A: LSTM or RFC
# B: 1 or 0 (1 means X.npy is already padded like in train.py, 0 means X.npy is not yet padded)
# Output: y_pred.npy will be saved as "./dataset/y_pred.npy"

from keras import backend as K
from keras_preprocessing import sequence
from keras.layers import *
from keras.models import Model, Sequential
import numpy as np
import time
import os
import sys
import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from fastai.imports import *
from fastai.structured import *
from IPython.display import display


def RFC():
    m = joblib.load("models/RFC.pkl")
    return m

def LSTM():
    m = joblib.load("models/LSTM.pkl")
    return m

def preprop_load():
    # assume X.npy is already preprocessed
    x = np.load('dataset/X.npy', allow_pickle=True)
    return x

def notpreprop_load():
    # if X.npy is not preprocessed
    x_raw = np.load('dataset/X.npy', allow_pickle=True)
    
    x = np.zeros((len(x_raw) * 185, 32))
    #y = np.zeros((1801 * 185))
    n = 0
    for i in range(len(x_raw)):
        # pad the current song's bars
        for j in range(185):
           # if j < len(train[i][1]) and 1 in train[i][1][j]:
           #     y[n] = train[i][1][j].index(1)
           # else:
           #     y[n] = 24
            
            for k in range(32):
                if j < len(x_raw[i][0]) and k < len(x_raw[i][0][j]) and 1 in x_raw[i][0][j][k]:
                    x[n][k] = x_raw[i][0][j][k].index(1)
                else:
                    x[n][k] = 12
            n += 1
    rem = []
    for i in range(len(x)):
        if all(x[i] == 12):
            rem.append(i)
    xn = np.delete(x, rem, 0)
    return xn

def predict(m, x):
    y_predict = m.predict(x)
    np.save("dataset/pred_y.npy", y_predict)
    return 0

if __name__ == '__main__':
    model_type = sys.argv[1]
    pad = sys.argv[2]
    
    if pad == "1":
        x = preprop_load()
    elif pad == "0":
        x = notpreprop_load()
    else:
        print("Error: Padded is either 1 or 0")

    if model_type == "RFC":
        m = RFC()
    elif model_type == "LSTM":
        m = LSTM()
    else:
        print("Error: Model is either RFC or LSTM")
    
    predict(m, x)
    print("Saved to \"dataset/pred_y.npy\"")