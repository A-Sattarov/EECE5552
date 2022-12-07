import numpy as np
import h5py
import pickle
import io
from feature_extraction import *
import pandas as pd

f = pd.read_pickle('FallAllD.pkl')

flat_frame, Act_ID = process_raw(f)

print(flat_frame)
