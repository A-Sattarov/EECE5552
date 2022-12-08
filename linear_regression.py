import numpy as np
from feature_extraction import *
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

'''
SubjectID      - the number of the person from whom the data were collected (15 in total)
Device         - from what type of device the data were collected from (Waist, Wrist, and Neck)
ActivityID     - the ID of the activity that the person was conducting (135 in total)
TrialNo        - the number of the trial that the activity was performed by the subject
Acc            - Acceleration data
Gyr            - Gyroscope data
'''

# This function gets properties and outputs x y z values of these properties as a dictionary
# Acc=True means you want acceleration data. Acc=False means you want Gyro data
def categorize_data(SubjectID, Device, ActivityID, TrialNo, Acc):
    df = pd.read_pickle('FallAllD.pkl')
    for i in range(len(df)):
        df.Acc[i] = {'x': df.Acc[i][:, 0], 'y': df.Acc[i][:, 1], 'z': df.Acc[i][:, 2]}
        df.Gyr[i] = {'x': df.Gyr[i][:, 0], 'y': df.Gyr[i][:, 1], 'z': df.Gyr[i][:, 2]}
    f = df.groupby("SubjectID", group_keys=True)[['Device', 'ActivityID', 'TrialNo', 'Acc', 'Gyr']].apply(lambda x: x.set_index(['Device', 'ActivityID', 'TrialNo']))
    df = f.loc[(f.index.get_level_values('SubjectID') == SubjectID) & (f.index.get_level_values('Device') == Device) & (f.index.get_level_values('ActivityID') == ActivityID) & (f.index.get_level_values('TrialNo') == TrialNo)]
    return df.reset_index().Acc[0] if Acc == True else df.reset_index().Gyr[0]


data = categorize_data(SubjectID=1, Device='Waist', ActivityID=13, TrialNo=1, Acc=True)
print(data)
