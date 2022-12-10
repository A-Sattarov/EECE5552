import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from feature_extraction import *
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

def normalize(data):
    return np.subtract(data, np.mean(data))

data_wrist = categorize_data(SubjectID=1, Device='Wrist', ActivityID=15, TrialNo=1, Acc=True)
data_waist = categorize_data(SubjectID=1, Device='Waist', ActivityID=15, TrialNo=1, Acc=True)
data_neck = categorize_data(SubjectID=1, Device='Neck', ActivityID=15, TrialNo=1, Acc=True)

test_wrist = categorize_data(SubjectID=1, Device='Wrist', ActivityID=15, TrialNo=2, Acc=True)

'''fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(data_wrist.get('x'), data_wrist.get('y'), data_wrist.get('z'), edgecolor='black', color='blue', label='Wrist', s=12)
ax1.scatter(data_waist.get('x'), data_waist.get('y'), data_waist.get('z'), edgecolor='black', color='orange', label='Waist', s=12)
ax1.scatter(data_neck.get('x'), data_neck.get('y'), data_neck.get('z'), edgecolor='black', color='red', label='Neck', s=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax1.grid()
plt.show()'''

'''fig2, ax2 = plt.subplots()
plt.plot(np.arange(len(data_neck.get('x'))), data_neck.get('x'), label='X')
plt.plot(np.arange(len(data_neck.get('y'))), data_neck.get('y'), label='Y')
plt.plot(np.arange(len(data_neck.get('z'))), data_neck.get('z'), label='Z')
ax2.set_axisbelow(True)
ax2.minorticks_on()
ax2.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
ax2.set_xlabel('Index')
ax2.set_ylabel('Acceleration')
ax2.legend()
plt.show()'''

# Normalize data (i.e. subtract the mean value from the data set to shift the graph vertically down)
normal_wrist = np.vstack((normalize(data_wrist.get('x')), normalize(data_wrist.get('y')), normalize(data_wrist.get('z')))).T
normal_waist = np.vstack((normalize(data_waist.get('x')), normalize(data_waist.get('y')), normalize(data_waist.get('z')))).T
normal_neck = np.vstack((normalize(data_neck.get('x')), normalize(data_neck.get('y')), normalize(data_neck.get('z')))).T
normal_test = np.vstack((normalize(test_wrist.get('x')), normalize(test_wrist.get('y')), normalize(test_wrist.get('z')))).T # test data of the wrist

'''fig3, ax3 = plt.subplots()
plt.plot(np.arange(len(normal_wrist)), normal_wrist[:, 0], label='X')
plt.plot(np.arange(len(normal_wrist)), normal_wrist[:, 1], label='Y')
plt.plot(np.arange(len(normal_wrist)), normal_wrist[:, 2], label='Z')
ax3.set_axisbelow(True)
ax3.minorticks_on()
ax3.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
ax3.set_xlabel('Index')
ax3.set_ylabel('Acceleration')
ax3.legend()
plt.show()'''

# Implement KNN
'''
Each class stands for different action
class 0 - regular activity
class 1 - fall
'''
classes = np.concatenate((np.zeros(1600, dtype=int), np.ones(1000, dtype=int), np.zeros(len(normal_wrist) - 2600, dtype=int))) # from 1600 to 2600 a fall occurred
kdata = list(zip(normal_wrist[:, 0], normal_wrist[:, 1], normal_wrist[:, 2]))
knn = KNeighborsClassifier(n_neighbors=120)
knn.fit(kdata, classes)

ktest = list(zip(normal_test[:, 0], normal_test[:, 1], normal_test[:, 2]))
prediction = knn.predict(ktest)

fig4, ax4 = plt.subplots(2)
fig4.suptitle('Train and Test data of wrist. SubjectID = 1, ActivityID=15')
ax4[0].scatter(np.arange(len(normal_wrist)), normal_wrist[:, 0], c=classes, label='X train')
ax4[1].scatter(np.arange(len(normal_test)), normal_test[:, 0], c=prediction, label='X test')
ax4[0].legend()
ax4[1].legend()
ax4[0].set_title("Trial No. 1")
ax4[1].set_title("Trial No. 2")
plt.show()

