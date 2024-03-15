import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut

# The 0.90 explained variance ratio calculated by pca.py is 13, but it feels too low, so it's increased to 30
n_low = 30
# Read train_data.npy file
train_data = np.load('dataset/NTU-RGB-D/xview/train_data.npy')
test_data = np.load('dataset/NTU-RGB-D/xview/val_data.npy')
# Read train_label.pkl file
with open('dataset/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)
with open('dataset/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
    test_label = pickle.load(f)

X_train = train_data.reshape(train_data.shape[0],-1)
X_test = test_data.reshape(test_data.shape[0],-1)
y_train = np.array(train_label[1])
y_test = np.array(test_label[1])

train_labels  = train_label[1]
test_labels = test_label[1]


# Extract two classes for comparison
# Get the training sample indexes with labels 10 and 11
index_train_10 = np.where(y_train == 10)[0]
index_train_11 = np.where(y_train == 11)[0]
print(len(index_train_10),len(index_train_10))
# Get the test sample indexes with labels 10 and 11
index_test_10 = np.where(y_test == 10)[0]
index_test_11 = np.where(y_test == 11)[0]
print(len(index_test_10),len(index_test_11))
# Select samples with labels 10 and 11 from the training set and test set
X_train_10_11 = np.concatenate((X_train[index_train_10], X_train[index_train_11]), axis=0)
y_train_10_11 = np.concatenate((y_train[index_train_10], y_train[index_train_11]), axis=0)
X_test_10_11 = np.concatenate((X_test[index_test_10], X_test[index_test_11]), axis=0)
y_test_10_11 = np.concatenate((y_test[index_test_10], y_test[index_test_11]), axis=0)
print(X_train_10_11.shape,y_train_10_11.shape,X_test_10_11.shape,y_test_10_11.shape)


# Perform PCA
pca = PCA(n_components=n_low)
# To ensure consistency of data, dimensionality reduction is performed on both the training and test sets together
X = np.concatenate((X_train_10_11, X_test_10_11), axis=0)
x_pca = pca.fit_transform(X)

# Splitting back according to the original proportion
x_pca_train = x_pca[:X_train_10_11.shape[0]]
x_pca_test = x_pca[X_train_10_11.shape[0]:]
print(x_pca_train.shape,x_pca_test.shape)

# Create an empty list to store accuracies
accuracy_s = []

# Iterate over multiple possible values to find the best neighbor value
for n_neighbor in range(1, 6):
    # Create a KNN classifier and set different numbers of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbor) 
    # Train the KNN model on the dimensionality-reduced training set
    knn_classifier.fit(x_pca_train, y_train_10_11)
    # Perform predictions on the dimensionality-reduced test set
    y_pred = knn_classifier.predict(x_pca_test)
    # Calculate prediction accuracy
    accuracy = accuracy_score(y_test_10_11, y_pred)
    print('n_neighbor=',n_neighbor,' singel acc=',accuracy)
    # Store the accuracy in the list
    accuracy_s.append(accuracy)
    



# Perform K-fold cross-validation
# Create an empty list to store accuracies
accuracy_k = []
# Define K value
k = 5  # Assuming k=5
# Create KFold object
kf = KFold(n_splits=k)
for n_neighbor in range(1, 6):
    accuracy_list = []
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbor) 
    for train_index, test_index in kf.split(x_pca_train):
        X_train_kfold, X_test_kfold = x_pca_train[train_index], x_pca_train[test_index]
        y_train_kfold, y_test_kfold = y_train_10_11[train_index], y_train_10_11[test_index]
        # Train the model
        knn_classifier.fit(X_train_kfold, y_train_kfold)
        # Perform predictions on the test set
        y_pred_kfold = knn_classifier.predict(X_test_kfold)
        # Calculate accuracy
        accuracy = accuracy_score(y_test_kfold, y_pred_kfold)
        # Add accuracy to the list
        accuracy_list.append(accuracy)
    # Calculate average accuracy
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    accuracy_k.append(average_accuracy)
    print("Average Accuracy (K-fold):", average_accuracy)


# Perform Leave-One-Out
# Create an empty list to store accuracies
accuracy_out = []
# Create LeaveOneOut object
loo = LeaveOneOut()
for n_neighbor in range(1, 6):
    accuracy_list = []
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbor) 
    # Perform Leave-One-Out cross-validation
    for train_index, test_index in loo.split(x_pca_train):
        X_train_loo, X_test_loo = x_pca_train[train_index], x_pca_train[test_index]
        y_train_loo, y_test_loo = y_train_10_11[train_index], y_train_10_11[test_index]
        # Train the model
        knn_classifier.fit(X_train_loo, y_train_loo)
        # Perform predictions on the test set
        y_pred_loo = knn_classifier.predict(X_test_loo)
        # Calculate accuracy
        accuracy = accuracy_score(y_test_loo, y_pred_loo)
        # Add accuracy to the list
        accuracy_list.append(accuracy)
    # calcualte average accuracy
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    accuracy_out.append(average_accuracy)
    print("Average Accuracy (leave-one-out):", average_accuracy)


# plot the accuracy
plt.plot(range(1, 6), accuracy_s, label='Test singel Accuracy', marker='o')
plt.plot(range(1, 6), accuracy_k, label='K-fold Accuracy', marker='o')
plt.plot(range(1, 6), accuracy_out, label='Leave One Out Accuracy', marker='o')
plt.xlabel('Number of Neighbors (n_neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neighbors')
plt.legend()
plt.savefig('Accuracy of knns')
plt.show()

