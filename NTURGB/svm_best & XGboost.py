import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb

# The 0.90 explained variance ratio calculated by pca.py is 13, but it feels too low, so it's increased to 30
n_low = 30
# Read train_data.npy file
train_data = np.load('../dataset/NTU-RGB-D/xview/train_data.npy')
test_data = np.load('../dataset/NTU-RGB-D/xview/val_data.npy')
# Read train_label.pkl file
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


# Extract two classes for comparison
# Get the training sample indexes with labels 10 and 11
index_train_5 = np.where(y_train == 5)[0]
index_train_6 = np.where(y_train == 6)[0]
index_train_36 = np.where(y_train == 36)[0]
index_train_37 = np.where(y_train == 37)[0] 
index_train_38 = np.where(y_train == 38)[0]
index_train_39 = np.where(y_train == 39)[0]
print(len(index_train_5),len(index_train_6))
# Get the test sample indexes with labels 10 and 11
index_test_5 = np.where(y_test == 5)[0]
index_test_6 = np.where(y_test == 6)[0]
index_test_36 = np.where(y_test == 36)[0]
index_test_37 = np.where(y_test == 37)[0]
index_test_38 = np.where(y_test == 38)[0] 
index_test_39 = np.where(y_test == 39)[0]
print(len(index_test_5),len(index_test_6))
# Select samples with labels 10 and 11 from the training set and test set
X_train_selected = np.concatenate((X_train[index_train_5], X_train[index_train_6],
                                    X_train[index_train_36], X_train[index_train_37],
                                    X_train[index_train_38], X_train[index_train_39]), axis=0)

y_train_selected = np.concatenate((y_train[index_train_5], y_train[index_train_6], 
                                    y_train[index_train_36], y_train[index_train_37],
                                    y_train[index_train_38], y_train[index_train_39]), axis=0)

X_test_selected = np.concatenate((X_test[index_test_5], X_test[index_test_6],
                                   X_test[index_test_36], X_test[index_test_37], 
                                   X_test[index_test_38], X_test[index_test_39]), axis=0)
                                   
y_test_selected = np.concatenate((y_test[index_test_5], y_test[index_test_6],
                                   y_test[index_test_36], y_test[index_test_37],
                                   y_test[index_test_38], y_test[index_test_39]), axis=0)
print(X_train_selected.shape,y_train_selected.shape,X_test_selected.shape,y_test_selected.shape)


# Perform PCA
pca = PCA(n_components=n_low)
# To ensure consistency of data, dimensionality reduction is performed on both the training and test sets together
X = np.concatenate((X_train_selected, X_test_selected), axis=0)
x_pca = pca.fit_transform(X)

# Splitting back according to the original proportion
x_pca_train = x_pca[:X_train_selected.shape[0]]
x_pca_test = x_pca[X_train_selected.shape[0]:]
print(x_pca_train.shape,x_pca_test.shape)



# Map the unique values of y_train_selected and y_test_selected to consecutive integers starting from 0
class_mapping = {cls: i for i, cls in enumerate(np.unique(np.concatenate((y_train_selected, y_test_selected))))}
y_train_mapped = np.array([class_mapping[cls] for cls in y_train_selected])
y_test_mapped = np.array([class_mapping[cls] for cls in y_test_selected])

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier()
# Train the XGBoost model on the dimensionality-reduced training set
xgb_classifier.fit(x_pca_train, y_train_mapped)
# Perform predictions on the dimensionality-reduced test set
y_pred_xgb = xgb_classifier.predict(x_pca_test)
# Calculate prediction accuracy
accuracy_xgb = accuracy_score(y_test_mapped, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)
# XGBoost Accuracy: 0.7906976744186046


# Create a SVM classifier and set the best k for testing
svm_classifier = SVC(kernel='linear')
# Train the SVM model on the dimensionality-reduced training set
svm_classifier.fit(x_pca_train, y_train_selected)
# Perform predictions on the dimensionality-reduced test set
y_pred = svm_classifier.predict(x_pca_test)

# # ROC is not used for multi-classification problems
# # Calculate parameters for ROC curve
# # Obtain the decision function results
# decision_values = svm_classifier.decision_function(x_pca_test)
# # Convert the decision function results to probability values using the sigmoid function
# y_scores = 1 / (1 + np.exp(-decision_values))
# # Convert labels to binary labels
# y_true_binary = [1 if label == 11 else 0 for label in y_test_selected]
# fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)  # Calculate parameters for ROC curve
# # Plot the ROC curve
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_true_binary, y_scores))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.savefig('ROC of svm reading and writing')
# plt.show()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test_selected, y_pred)
# Plot the confusion matrix
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('Confusion Matrix of Best svm')
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test_selected, y_pred))
# 0.76


# # Calculate prediction accuracy
# accuracy = accuracy_score(y_test_selected, y_pred)
# print("Accuracy:", accuracy)

# # Perform K-fold cross-validation
# k = 10
# kf = KFold(n_splits=k)
# accuracy_scores = []
# for train_index, val_index in kf.split(x_pca_train):
#     # Split the data into training and validation sets
#     X_train_fold, X_val_fold = x_pca_train[train_index], x_pca_train[val_index]
#     y_train_fold, y_val_fold = y_train_selected[train_index], y_train_selected[val_index]
    
#     # Create a new SVM classifier for each fold
#     svm_classifier_fold = SVC(kernel='linear')
    
#     # Train the SVM model on the training set
#     svm_classifier_fold.fit(X_train_fold, y_train_fold)
    
#     # Perform predictions on the validation set
#     y_pred_fold = svm_classifier_fold.predict(X_val_fold)
    
#     # Calculate the accuracy score for this fold
#     accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
    
#     # Append the accuracy score to the list
#     accuracy_scores.append(accuracy_fold)

# # Calculate the average accuracy score across all folds
# average_accuracy = np.mean(accuracy_scores)
# print("Average Accuracy:", average_accuracy)


