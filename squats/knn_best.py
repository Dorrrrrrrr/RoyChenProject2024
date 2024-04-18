import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
X = pickle.load(open("squats_interpolated.p", 'rb'))
y = pickle.load(open("labels.p", 'rb'))
X = np.array(X).T
mean_X = np.mean(X, axis=0)
X -= mean_X
y = np.array(y)
pca = PCA(n_components=32)
x_pca = pca.fit_transform(X)

# Prepare to collect accuracies
leave_one_out_accuracies = []
k_fold_accuracies = []
hold_out_accuracies = []

# Define the range of K to test
k_values = range(1, 20)

# Loop over K values
for k in k_values:
    print(f"Testing K={k}")
    
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Perform leave-one-out cross-validation
    loo = LeaveOneOut()
    loo_accuracies = []
    for train_index, test_index in loo.split(x_pca):
        X_train, X_test = x_pca[train_index], x_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        loo_accuracies.append(accuracy_score(y_test, y_pred))
    leave_one_out_accuracies.append(np.mean(loo_accuracies))
    
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_accuracies = []
    for train_index, test_index in kf.split(x_pca):
        X_train, X_test = x_pca[train_index], x_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        kf_accuracies.append(accuracy_score(y_test, y_pred))
    k_fold_accuracies.append(np.mean(kf_accuracies))
    
    # Perform holdout validation
    X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    hold_out_accuracies.append(accuracy_score(y_test, y_pred))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, leave_one_out_accuracies, label='Leave One Out Accuracy')
plt.plot(k_values, k_fold_accuracies, label='5-fold CV Accuracy')
plt.plot(k_values, hold_out_accuracies, label='Hold-out Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN Model Performance on Different Values of K')
plt.legend()
plt.savefig('knn_performance.png')
plt.show()

