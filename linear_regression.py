import numpy as np
import h5py
import pickle
import io
import pandas as pd

f = pd.read_pickle('FallAllD.pkl')
# f = pd.read_hdf('FallAllD.h5', 'df')
# df = pd.DataFrame(f, columns = ['SubjectID', 'Device', 'ActivityID', 'TrialNo', 'Acc', 'Gyr', 'Mag', 'Bar'])

f.to_csv("data/data.csv")

print(pd.read_csv('data/data.csv'))
