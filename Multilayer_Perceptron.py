# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
import feature_extraction


#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Activation



def reformat_data(Acc, Gyr):
    Acc_reformat = []
    for entry in Acc:
        Acc_reformat.append(entry)

    return Acc_reformat


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    Acc_sen = 0.000244
    Gyr_sen = 0.07
    raw_data = pd.read_pickle('FallAllD.pkl')
    raw_data['Acc'] = raw_data['Acc']*Acc_sen
    raw_data['Gyr'] = raw_data['Gyr'] * Gyr_sen
    sample_freq = 238

    data = feature_extraction.get_features_alt(raw_data,{},{})
    #data = feature_extraction.get_features(raw_data, {}, {},238)
    le = preprocessing.LabelEncoder()
    #raw_data['Device'] = le.fit_transform(raw_data['Device'])
    #feat = data[['W_Acc_max','W_Gyr_max','W_Acc_freq','W_Gyr_freq']]
    feat = data[['W_Acc_max', 'W_Acc_min','W_Acc_mean','W_Acc_std','W_Acc_range', 'W_Gyr_max', 'W_Gyr_min','W_Gyr_mean','W_Gyr_std','W_Gyr_range']]
    feat = feature_extraction.flatten_frame_alt(feat)
    #feat = feature_extraction.flatten_frame(feat)
    #[data, act] = feature_extraction.process_raw(raw_data)
    #feat = data
    act = data['Activity_ID']
    x_train, X_test, y_train, y_test = train_test_split(feat, act, test_size=0.2, random_state=150)
    clf = MLPClassifier(hidden_layer_sizes=(15,3),
                        random_state=19,
                        verbose=True,
                        learning_rate_init=0.01)

    clf.fit(x_train,y_train)

    ypred = clf.predict(X_test)
    score = [accuracy_score(y_test,ypred),f1_score(y_test,ypred),recall_score(y_test,ypred),precision_score(y_test,ypred)]
    print(ypred)
    print(score)

    with open('MLP_model', 'wb') as f:
        pickle.dump(clf,f)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
