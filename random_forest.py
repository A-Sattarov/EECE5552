import numpy as np
import h5py
import pickle
import io
import pandas as pd
from scipy import fft
from feature_extraction import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, recall_score, precision_score
from itertools import chain

f = pd.read_pickle('FallAllD.pkl')
c = pd.read_pickle('FallClas.pkl')
# f = pd.read_hdf('FallAllD.h5', 'df')
df = pd.DataFrame(f, columns = ['SubjectID', 'Device', 'ActivityID', 'TrialNo', 'Acc', 'Gyr'])
dc = pd.DataFrame(c, columns = ['Fall'])

# f = h5py.File('FallAllD.h5', 'r')
features = get_features_alt(df,{},{})
train_size = int(len(features)*0.8)
rand_indices = list(range(0,len(features)))
np.random.shuffle(rand_indices)
train_indices = rand_indices[0:train_size]
test_indices = rand_indices[train_size:]
# train_data = features.iloc[train_indices,[1,2,3,4,5]].values
features_flat = []
# Include gyro
# for i in range(len(features)):
#     features_flat.append(np.concatenate((features.iloc[i][1],features.iloc[i][2],features.iloc[i][3],
#     features.iloc[i][4],features.iloc[i][5],features.iloc[i][6],features.iloc[i][7],features.iloc[i][8],
#     features.iloc[i][9],features.iloc[i][10])))

# Only Accel
for i in range(len(features)):
    features_flat.append(np.concatenate((features.iloc[i][1],features.iloc[i][2],features.iloc[i][3],
    features.iloc[i][4],features.iloc[i][5])))
train_data = [features_flat[i] for i in train_indices]
test_data = [features_flat[i] for i in test_indices]

train_class = dc.iloc[train_indices].Fall
test_class = dc.iloc[test_indices].Fall.values
# print(test_class)
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(train_data,train_class)

results = np.round(regressor.predict(test_data))
score = f1_score(test_class,results, average='weighted')
rec = recall_score(test_class,results)
prec = precision_score(test_class,results)
counter = 0

print(results)
for i in range(len(results)):
    if (test_class[i] == results[i]):
        counter += 1
print(counter/len(results))
print(score)
print(rec)
print(prec)
