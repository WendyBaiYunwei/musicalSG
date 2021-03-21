from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import time
import os


def train():
    # zero padding
    input_vec = sequence.pad_sequences(np.load('dataset/input_vector.npy', allow_pickle=True))
    target_vec = np.load('dataset/target_vector.npy')

    input_dim = input_vec.shape[2]
    output_dim = target_vec.shape[1]

    clf=RandomForestClassifier(n_estimators=100)

    clf.fit(input_vec,target_vec)

    print("Done training")

    # y_pred=clf.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    train()