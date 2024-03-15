import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# read file for training datasets and test datasets
train_data = np.load('dataset/NTU-RGB-D/xview/train_data.npy')
test_data = np.load('dataset/NTU-RGB-D/xview/val_data.npy')
# read file for training labels and test labels (opening file in binary mode for reading)
with open ('dataset/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)
with open ('dataset/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
    test_label = pickle.load(f)
    
X_train = train_data.reshape(train_data.shape[0],-1)
X_test = test_data.reshape(test_data.shape[0],-1)
y_train = np.array(train_label[1])
y_test = np.array(test_label[1])
# print above variables
print(train_data.shape, len(train_label[1]), len(train_label[1]))

train_labels  = train_label[1]
test_labels = test_label[1]
# count the number of each class in the training dataset
train_class_counts = {label: train_labels.count(label) for label in set(train_labels)}

# count the number of each class in the testing dataset
test_labels_counts = {label: test_labels.count(label) for label in set(test_labels)}

# Create a list of class labels
classes = list(train_class_counts.keys())
# Create a list of training counts for each class label
train_counts = [train_class_counts[label] for label in classes]
# Create a list of testing counts for each class label, defaulting to 0 if the label is not present in the testing dataset
test_counts = [test_labels_counts.get(label, 0) for label in classes]

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(classes, train_counts, color='blue', alpha=0.5, label='Train')
ax.bar(classes, test_counts, color='red', alpha=0.5, label='Test')

ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('Class Counts in Train and Test Datasets')
ax.legend()

# rotate x-axis labels for better display
plt.xticks(rotation=45)

# show the bar chart
plt.tight_layout()
plt.show()
plt.savefig('Class Counts')
plt.show()

# Extract two classes for comparison
# obtain the index of the training dataset samples with labels 10 and 11
index_train_10 = np.where(y_train == 10)[0]
index_train_11 = np.where(y_train == 11)[0]
print(len(index_train_10),len(index_train_11))
# obtain the index of the testing dataset samples with labels 10 and 11
index_test_10 = np.where(y_test == 10)[0]
index_test_11 = np.where(y_test == 11)[0]
print(len(index_test_10),len(index_test_11))
# select samples with labels 10 and 11 from the training and testing datasets
X_train_10_11 = np.concatenate((X_train[index_train_10], X_train[index_train_11]), axis=0)
X_test_10_11 = np.concatenate((X_test[index_test_10], X_test[index_test_11]), axis=0)
y_train_10_11 = np.concatenate((y_train[index_train_10], y_train[index_train_11]), axis=0)
y_test_10_11 = np.concatenate((y_test[index_test_10], y_test[index_test_11]), axis=0)
print(X_train_10_11.shape, X_test_10_11.shape, y_train_10_11.shape, y_test_10_11.shape)

# Perform PCA on the training dataset
pca = PCA()
X = np.concatenate((X_train_10_11, X_test_10_11), axis=0)
x_pca = pca.fit_transform(X)

# calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
explained_variance_ratio = explained_variance_ratio[:40]
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# plot the explained variance ratio
index_095 = np.argmax(cumulative_explained_variance_ratio >= 0.90)
fig = plt.figure(figsize=(12, 6))
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(len(cumulative_explained_variance_ratio)), cumulative_explained_variance_ratio, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.90, color='red', linestyle='--')
plt.axvline(x=index_095, color='red', linestyle='--')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend(loc='best')
plt.title('Explained Variance Ratio')
plt.savefig('Explained Variance')
plt.show()
print(index_095)