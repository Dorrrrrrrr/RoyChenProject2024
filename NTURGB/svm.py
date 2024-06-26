import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut

# The explained variance of 0.90 calculated by pca.py is 13, but I feel it is too little, so I increased it to 30.
n_low = 30
# Read the train_data.npy file
train_data = np.load('../dataset/NTU-RGB-D/xview/train_data.npy')
test_data = np.load('../dataset/NTU-RGB-D/xview/val_data.npy')
# Read the train_label.pkl file
with open('../dataset/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)
with open('../dataset/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
    test_label = pickle.load(f)

X_train = train_data.reshape(train_data.shape[0],-1)
X_test = test_data.reshape(test_data.shape[0],-1)
y_train = np.array(train_label[1])
y_test = np.array(test_label[1])

train_labels  = train_label[1]
test_labels = test_label[1]


# Take out two types of actions
# Get the training set sample index with labels 5 and 6
index_train_5 = np.where(y_train == 5)[0]
index_train_6 = np.where(y_train == 6)[0]
index_train_36 = np.where(y_train == 36)[0]
index_train_37 = np.where(y_train == 37)[0]
index_train_38 = np.where(y_train == 38)[0]
index_train_39 = np.where(y_train == 39)[0]
print(len(index_train_5),len(index_train_6))
# Get the test set sample index with labels 5 and 6
index_test_5 = np.where(y_test == 5)[0]
index_test_6 = np.where(y_test == 6)[0]
index_test_36 = np.where(y_test == 36)[0]
index_test_37 = np.where(y_test == 37)[0]
index_test_38 = np.where(y_test == 38)[0]
index_test_39 = np.where(y_test == 39)[0]
print(len(index_test_5),len(index_test_6))
# Select samples with labels 5 and 6 from the training set and test set
X_train_5_6_36_37_38_39 = np.concatenate((X_train[index_train_5], X_train[index_train_6], X_train[index_train_36], X_train[index_train_37], X_train[index_train_38], X_train[index_train_39]), axis=0)
y_train_5_6_36_37_38_39 = np.concatenate((y_train[index_train_5], y_train[index_train_6], y_train[index_train_36], y_train[index_train_37], y_train[index_train_38], y_train[index_train_39]), axis=0)
X_test_5_6_36_37_38_39 = np.concatenate((X_test[index_test_5], X_test[index_test_6], X_test[index_test_36], X_test[index_test_37], X_test[index_test_38], X_test[index_test_39]), axis=0)
y_test_5_6_36_37_38_39 = np.concatenate((y_test[index_test_5], y_test[index_test_6], y_test[index_test_36], y_test[index_test_37], y_test[index_test_38], y_test[index_test_39]), axis=0)
print(X_train_5_6_36_37_38_39.shape,y_train_5_6_36_37_38_39.shape,X_test_5_6_36_37_38_39.shape,y_test_5_6_36_37_38_39.shape)


# PCA
pca = PCA(n_components=n_low)
X = np.concatenate((X_train_5_6_36_37_38_39, X_test_5_6_36_37_38_39), axis=0)
x_pca = pca.fit_transform(X)

# Split according to original proportion
x_pca_train = x_pca[:X_train_5_6_36_37_38_39.shape[0]]
x_pca_test = x_pca[X_train_5_6_36_37_38_39.shape[0]:]
print(x_pca_train.shape,x_pca_test.shape)


# Create an empty list to store accuracy
accuracy_results = {'linear': [], 'rbf': [], 'poly': [], 'sigmoid': []}

# Traverse multiple possible kernels
for kernel_type in ['linear', 'rbf', 'poly', 'sigmoid']:
    # Create an SVM classifier and set the kernel type
    svm_classifier = SVC(kernel=kernel_type)
    # Train the SVM model on the dimensionally reduced training set
    svm_classifier.fit(x_pca_train, y_train_5_6_36_37_38_39)
    # Make predictions on the dimensionally reduced test set
    y_pred_svm = svm_classifier.predict(x_pca_test)
    # Calculate prediction accuracy
    accuracy = accuracy_score(y_test_5_6_36_37_38_39, y_pred_svm)
    # Store accuracy in dictionary
    accuracy_results[kernel_type].append(accuracy)
    
# Output the average accuracy for each kernel
for kernel_type, accuracies in accuracy_results.items():
    average_accuracy = np.mean(accuracies)
    print(f"Average Accuracy for {kernel_type} kernel: {average_accuracy}")

average_accuracies = [np.mean(accuracy_results['linear']),
                      np.mean(accuracy_results['rbf']),
                      np.mean(accuracy_results['poly']),
                      np.mean(accuracy_results['sigmoid'])]
# Visualizing accuracy results
kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']
# Draw a histogram
plt.bar(kernel_types, average_accuracies, color='skyblue')
# Add tags and titles
plt.xlabel('Kernel Types')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy of Different Kernel Types')

# Show average accuracy value
for i in range(len(kernel_types)):
    plt.text(i, average_accuracies[i], round(average_accuracies[i], 4), ha='center', va='bottom')
# Save chart and display
plt.savefig('Accuracy of SVMs.png')
plt.show()

