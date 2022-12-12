import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from feature_extraction import get_features_alt

# Import data, get features
raw_data = pd.read_pickle('FallAllD.pkl')
features = get_features_alt(raw_data, 0, 0)

# Split data into test and train datasets
# 0 - fall; 1 - not a fall
train, test = train_test_split(features, test_size=0.2)

# Initiate KNN, flatten the data, fit the data
knn = KNeighborsClassifier(n_neighbors=70)

# Turn the data into a multidimensional array, then flatten it. Avoid max value and min value features
ktrain = list(zip(train.loc[:, 'W_Acc_std'].str[0].to_numpy(), train.loc[:, 'W_Acc_std'].str[1].to_numpy(), train.loc[:, 'W_Acc_std'].str[2].to_numpy(), train.loc[:, 'W_Acc_range'].str[0].to_numpy(), train.loc[:, 'W_Acc_range'].str[1].to_numpy(), train.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), train.loc[:, 'W_Gyr_std'].str[0].to_numpy(), train.loc[:, 'W_Gyr_std'].str[1].to_numpy(), train.loc[:, 'W_Gyr_std'].str[2].to_numpy(), train.loc[:, 'W_Gyr_range'].str[0].to_numpy(), train.loc[:, 'W_Gyr_range'].str[1].to_numpy(), train.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy()))
ktest = list(zip(test.loc[:, 'W_Acc_std'].str[0].to_numpy(), test.loc[:, 'W_Acc_std'].str[1].to_numpy(), test.loc[:, 'W_Acc_std'].str[2].to_numpy(), test.loc[:, 'W_Acc_range'].str[0].to_numpy(), test.loc[:, 'W_Acc_range'].str[1].to_numpy(), test.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), test.loc[:, 'W_Gyr_std'].str[0].to_numpy(), test.loc[:, 'W_Gyr_std'].str[1].to_numpy(), test.loc[:, 'W_Gyr_std'].str[2].to_numpy(), test.loc[:, 'W_Gyr_range'].str[0].to_numpy(), test.loc[:, 'W_Gyr_range'].str[1].to_numpy(), test.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy()))

# Fit the data
knn.fit(ktrain, train.Activity_ID.to_numpy().astype(int))

# Predict the test data
prediction = knn.predict(ktest)

# Calculate accuracy
calc = np.zeros(len(prediction))
for i in range(len(prediction)):
    calc[i] = 1 if prediction[i]==test.Activity_ID.to_numpy()[i].astype(int) else 0

accuracy = np.sum(calc)/len(calc)
print("Accuracy is: ", accuracy, " k-factor is: ", 35)


'''
# Run this to see a range of accuracies for various k-numbers
for j in range(1, 500, 5):
    # Initiate KNN, flatten the data, fit the data
    knn = KNeighborsClassifier(n_neighbors=int(j))

    # Turn the data into a multidimensional array, then flatten it
    # All features - not efficient because some features only decrease accuracy
    # ktrain = list(zip(train.loc[:, 'W_Acc_std'].str[0].to_numpy(), train.loc[:, 'W_Acc_std'].str[1].to_numpy(), train.loc[:, 'W_Acc_std'].str[2].to_numpy(), train.loc[:, 'W_Acc_range'].str[0].to_numpy(), train.loc[:, 'W_Acc_range'].str[1].to_numpy(), train.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), train.loc[:, 'W_Acc_max'].str[0].to_numpy(), train.loc[:, 'W_Acc_max'].str[1].to_numpy(), train.loc[:, 'W_Acc_max'].str[2].to_numpy(), train.loc[:, 'W_Acc_min'].str[0].to_numpy(), train.loc[:, 'W_Acc_min'].str[1].to_numpy(), train.loc[:, 'W_Acc_min'].str[2].to_numpy(), train.loc[:, 'W_Gyr_std'].str[0].to_numpy(), train.loc[:, 'W_Gyr_std'].str[1].to_numpy(), train.loc[:, 'W_Gyr_std'].str[2].to_numpy(), train.loc[:, 'W_Gyr_range'].str[0].to_numpy(), train.loc[:, 'W_Gyr_range'].str[1].to_numpy(), train.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy(), train.loc[:, 'W_Gyr_max'].str[0].to_numpy(), train.loc[:, 'W_Gyr_max'].str[1].to_numpy(), train.loc[:, 'W_Gyr_max'].str[2].to_numpy(), train.loc[:, 'W_Gyr_min'].str[0].to_numpy(), train.loc[:, 'W_Gyr_min'].str[1].to_numpy(), train.loc[:, 'W_Gyr_min'].str[2].to_numpy()))
    # ktest = list(zip(test.loc[:, 'W_Acc_std'].str[0].to_numpy(), test.loc[:, 'W_Acc_std'].str[1].to_numpy(), test.loc[:, 'W_Acc_std'].str[2].to_numpy(), test.loc[:, 'W_Acc_range'].str[0].to_numpy(), test.loc[:, 'W_Acc_range'].str[1].to_numpy(), test.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), train.loc[:, 'W_Acc_max'].str[0].to_numpy(), train.loc[:, 'W_Acc_max'].str[1].to_numpy(), train.loc[:, 'W_Acc_max'].str[2].to_numpy(), train.loc[:, 'W_Acc_min'].str[0].to_numpy(), train.loc[:, 'W_Acc_min'].str[1].to_numpy(), train.loc[:, 'W_Acc_min'].str[2].to_numpy(), test.loc[:, 'W_Gyr_std'].str[0].to_numpy(), test.loc[:, 'W_Gyr_std'].str[1].to_numpy(), test.loc[:, 'W_Gyr_std'].str[2].to_numpy(), test.loc[:, 'W_Gyr_range'].str[0].to_numpy(), test.loc[:, 'W_Gyr_range'].str[1].to_numpy(), test.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy(), train.loc[:, 'W_Gyr_max'].str[0].to_numpy(), train.loc[:, 'W_Gyr_max'].str[1].to_numpy(), train.loc[:, 'W_Gyr_max'].str[2].to_numpy(), train.loc[:, 'W_Gyr_min'].str[0].to_numpy(), train.loc[:, 'W_Gyr_min'].str[1].to_numpy(), train.loc[:, 'W_Gyr_min'].str[2].to_numpy()))
    ktrain = list(zip(train.loc[:, 'W_Acc_std'].str[0].to_numpy(), train.loc[:, 'W_Acc_std'].str[1].to_numpy(), train.loc[:, 'W_Acc_std'].str[2].to_numpy(), train.loc[:, 'W_Acc_range'].str[0].to_numpy(), train.loc[:, 'W_Acc_range'].str[1].to_numpy(), train.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), train.loc[:, 'W_Gyr_std'].str[0].to_numpy(), train.loc[:, 'W_Gyr_std'].str[1].to_numpy(), train.loc[:, 'W_Gyr_std'].str[2].to_numpy(), train.loc[:, 'W_Gyr_range'].str[0].to_numpy(), train.loc[:, 'W_Gyr_range'].str[1].to_numpy(), train.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy()))
    ktest = list(zip(test.loc[:, 'W_Acc_std'].str[0].to_numpy(), test.loc[:, 'W_Acc_std'].str[1].to_numpy(), test.loc[:, 'W_Acc_std'].str[2].to_numpy(), test.loc[:, 'W_Acc_range'].str[0].to_numpy(), test.loc[:, 'W_Acc_range'].str[1].to_numpy(), test.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), test.loc[:, 'W_Gyr_std'].str[0].to_numpy(), test.loc[:, 'W_Gyr_std'].str[1].to_numpy(), test.loc[:, 'W_Gyr_std'].str[2].to_numpy(), test.loc[:, 'W_Gyr_range'].str[0].to_numpy(), test.loc[:, 'W_Gyr_range'].str[1].to_numpy(), test.loc[:, 'W_Gyr_range'].str[2].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[0].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[1].to_numpy(), train.loc[:, 'W_Gyr_mean'].str[2].to_numpy()))


    # Fit the data
    knn.fit(ktrain, train.Activity_ID.to_numpy().astype(int))

    # Predict the test data
    prediction = knn.predict(ktest)

    # Calculate accuracy
    calc = np.zeros(len(prediction))
    for i in range(len(prediction)):
        calc[i] = 1 if prediction[i]==test.Activity_ID.to_numpy()[i].astype(int) else 0

    accuracy = np.sum(calc)/len(calc)
    print("Accuracy is: ", accuracy, " k-factor is: ", j)'''
