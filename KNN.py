import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from feature_extraction import get_features_alt

# Load KNN configuration
# knn = pickle.loads(knn_config.pkl)

# Import data, get features
raw_data = pd.read_pickle('FallAllD.pkl')
features = get_features_alt(raw_data, 0, 0)

# Split data into test and train datasets
# 0 - fall; 1 - not a fall
train, test = train_test_split(features, test_size=0.1)

# select the K-factor
# k = 15

accuracy = np.array([[0, 0, 0]])
for k in range(1, len(test)):
    # Initiate KNN, flatten the data, fit the data
    knn = KNeighborsClassifier(n_neighbors=k)

    # Turn the data into a multidimensional array, then flatten it. Avoid max value and min value features
    ktrain = list(zip(train.loc[:, 'W_Acc_std'].str[0].to_numpy(), train.loc[:, 'W_Acc_std'].str[1].to_numpy(), train.loc[:, 'W_Acc_std'].str[2].to_numpy(), train.loc[:, 'W_Acc_range'].str[0].to_numpy(), train.loc[:, 'W_Acc_range'].str[1].to_numpy(), train.loc[:, 'W_Acc_range'].str[2].to_numpy(), train.loc[:, 'W_Acc_mean'].str[0].to_numpy(), train.loc[:, 'W_Acc_mean'].str[1].to_numpy(), train.loc[:, 'W_Acc_mean'].str[2].to_numpy(), train.loc[:, 'W_Acc_max'].str[2].to_numpy(), train.loc[:, 'W_Acc_min'].str[2].to_numpy()))
    ktest = list(zip(test.loc[:, 'W_Acc_std'].str[0].to_numpy(), test.loc[:, 'W_Acc_std'].str[1].to_numpy(), test.loc[:, 'W_Acc_std'].str[2].to_numpy(), test.loc[:, 'W_Acc_range'].str[0].to_numpy(), test.loc[:, 'W_Acc_range'].str[1].to_numpy(), test.loc[:, 'W_Acc_range'].str[2].to_numpy(), test.loc[:, 'W_Acc_mean'].str[0].to_numpy(), test.loc[:, 'W_Acc_mean'].str[1].to_numpy(), test.loc[:, 'W_Acc_mean'].str[2].to_numpy(), test.loc[:, 'W_Acc_max'].str[2].to_numpy(), test.loc[:, 'W_Acc_min'].str[2].to_numpy()))

    # Fit the data
    knn.fit(ktrain, train.Activity_ID.to_numpy().astype(int))

    # Predict the test data
    prediction = knn.predict(ktest)

    # Calculate accuracy
    calc = np.zeros(len(prediction))
    for i in range(len(prediction)):
        calc[i] = 1 if prediction[i]==test.Activity_ID.to_numpy()[i].astype(int) else 0

    # The accuracy is:
    # accuracy = np.sum(calc)/len(calc)
    precision = precision_score(test.Activity_ID.to_numpy().astype(int), prediction)
    recall = recall_score(test.Activity_ID.to_numpy().astype(int), prediction)

    # print("Accuracy is: ", accuracy, " k-factor is: ", k)
    # print("F1 score is: ", f1score, "k-factor is: ", k)
    # print("Recall (unweighted) score is: ", f1score)
    accuracy = np.append(accuracy, [[k, recall, precision]], axis=0)

'''print(accuracy)
print(accuracy[:, 0])'''

# Prediction
# print("The Predictions are: ", prediction)

# Accuracy Results
fig1, ax1 = plt.subplots()
ax1.plot(accuracy[:, 0], accuracy[:, 1], color='blue', label='Recall score')
ax1.plot(accuracy[:, 0], accuracy[:, 2], color='red', label='Precision score')
ax1.set_axisbelow(True)
ax1.minorticks_on()
ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
ax1.legend()
plt.title("KNN accuracy for wrist")
plt.xlabel("K parameter")
plt.ylabel("Accuracy")
plt.savefig('KNNfig.jpeg')
plt.show()

# print accuracy results
accuracy = pd.DataFrame(accuracy, columns = ['k', 'recall', 'precision'])
print("The highest accuracy is at: \n", accuracy.sort_values(by=['recall', 'precision'], ascending=False).head(1).to_string(index=False))

'''
Saving the KNN configuration. Undo the for loop and select the K value before saving the configuration
'''
# pickle.dumps(knn, 'knn_config.pkl')
