import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# load data
X = pickle.load(open("squats_interpolated.p", 'rb'))
y = pickle.load(open("labels.p", 'rb'))
X = np.array(X).T
mean_X = np.mean(X, axis=0)
X = X - mean_X
y = np.array(y)
pca = PCA(n_components=32)
x_pca = pca.fit_transform(X)
print(x_pca.shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
# Train the classifier
knn.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of knn')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_knn.png')
plt.show()

# Compute the classification report for each class
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_kfold = []

for train_index, test_index in kf.split(x_pca):
    X_train, X_test = x_pca[train_index], x_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = knn.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_kfold.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy_kfold = sum(accuracies_kfold) / len(accuracies_kfold)
print("Average accuracy (k-fold):", average_accuracy_kfold)

# Perform leave-one-out cross-validation
loo = LeaveOneOut()
accuracies_loo = []

for train_index, test_index in loo.split(x_pca):
    X_train, X_test = x_pca[train_index], x_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict the label for the test sample
    y_pred = knn.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_loo.append(accuracy)

# Calculate the average accuracy across all samples
average_accuracy_loo = sum(accuracies_loo) / len(accuracies_loo)
print("Average accuracy (leave-one-out):", average_accuracy_loo)


# fig = plt.figure()
# plt.plot(loss_list, label='Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.savefig('loss_rnn.png')
# plt.show()