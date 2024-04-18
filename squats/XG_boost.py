import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the data
X = pickle.load(open("squats_interpolated.p", 'rb'))
y = pickle.load(open("labels.p", 'rb'))
X = np.array(X).T
mean_X = np.mean(X, axis=0)
X = X - mean_X
y = np.array(y)

# Perform PCA
pca = PCA(n_components=32)
x_pca = pca.fit_transform(X)
print(x_pca.shape)

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

# Train the classifier
xgb_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = xgb_model.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_xgb.png')
plt.show()

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_kfold = []

for train_index, test_index in kf.split(x_pca):
    X_train, X_test = x_pca[train_index], x_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the classifier
    xgb_model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = xgb_model.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_kfold.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy_kfold = sum(accuracies_kfold) / len(accuracies_kfold)
print("Average accuracy (k-fold):", average_accuracy_kfold)

# Leave-one-out cross-validation
loo = LeaveOneOut()
accuracies_loo = []

for train_index, test_index in loo.split(x_pca):
    X_train, X_test = x_pca[train_index], x_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the classifier
    xgb_model.fit(X_train, y_train)

    # Predict the label for the test sample
    y_pred = xgb_model.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_loo.append(accuracy)

# Calculate the average accuracy across all samples
average_accuracy_loo = sum(accuracies_loo) / len(accuracies_loo)
print("Average accuracy (leave-one-out):", average_accuracy_loo)