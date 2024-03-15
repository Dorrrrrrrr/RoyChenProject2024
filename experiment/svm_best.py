import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

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

# Create a SVM classifier and set the best k for testing
svm_classifier = SVC(kernel='rbf')
# Train the SVM model on the dimensionality-reduced training set
svm_classifier.fit(x_pca_train, y_train_10_11)
# Perform predictions on the dimensionality-reduced test set
y_pred = svm_classifier.predict(x_pca_test)

# ROC is not used for multi-classification problems
# Calculate parameters for ROC curve
# Obtain the decision function results
decision_values = svm_classifier.decision_function(x_pca_test)
# Convert the decision function results to probability values using the sigmoid function
y_scores = 1 / (1 + np.exp(-decision_values))
# Convert labels to binary labels
y_true_binary = [1 if label == 11 else 0 for label in y_test_10_11]
fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)  # Calculate parameters for ROC curve
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_true_binary, y_scores))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('ROC of svm reading and writing')
plt.show()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test_10_11, y_pred)
# Plot the confusion matrix
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('Confusion Matrix of svm reading and writing')
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test_10_11, y_pred))
