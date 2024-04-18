from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X = pickle.load(open("squats_interpolated.p", 'rb'))
y = pickle.load(open("labels.p", 'rb'))
X = np.array(X).T
mean_X = np.mean(X, axis=0)
X = X - mean_X
y = np.array(y)
pca = PCA(n_components=32)
x_pca = pca.fit_transform(X)
print(x_pca.shape)


X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)


accuracy_results = {'linear': {'test': [], '5_fold_cv': [], 'loo': []},
                    'rbf': {'test': [], '5_fold_cv': [], 'loo': []},
                    'poly': {'test': [], '5_fold_cv': [], 'loo': []},
                    'sigmoid': {'test': [], '5_fold_cv': [], 'loo': []}}


kf = KFold(n_splits=5, shuffle=True, random_state=42)


loo = LeaveOneOut()


for kernel_type in ['linear', 'rbf', 'poly', 'sigmoid']:
    print(f"Training with {kernel_type} kernel")

   
    svm_classifier = SVC(kernel=kernel_type)
    
    
    kf_accuracies = []
    for train_index, test_index in kf.split(x_pca):
        X_kf_train, X_kf_test = x_pca[train_index], x_pca[test_index]
        y_kf_train, y_kf_test = y[train_index], y[test_index]
        svm_classifier.fit(X_kf_train, y_kf_train)
        y_kf_pred = svm_classifier.predict(X_kf_test)
        kf_accuracies.append(accuracy_score(y_kf_test, y_kf_pred))
    accuracy_results[kernel_type]['5_fold_cv'] = np.mean(kf_accuracies)
    
   
    loo_accuracies = []
    for train_index, test_index in loo.split(x_pca):
        X_loo_train, X_loo_test = x_pca[train_index], x_pca[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        svm_classifier.fit(X_loo_train, y_loo_train)
        y_loo_pred = svm_classifier.predict(X_loo_test)
        loo_accuracies.append(accuracy_score(y_loo_test, y_loo_pred))
    accuracy_results[kernel_type]['loo'] = np.mean(loo_accuracies)
    
   
    svm_classifier.fit(X_train, y_train)
    y_test_pred = svm_classifier.predict(X_test)
    accuracy_results[kernel_type]['test'] = accuracy_score(y_test, y_test_pred)


kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']


fig, ax = plt.subplots(figsize=(10, 8))


bar_width = 0.2

index = np.arange(len(kernel_types))


colors = ['orange', 'green', 'blue'] 
labels = ['Test Score', '5-fold CV Score', 'LOO Score']  
for idx, kernel in enumerate(kernel_types):
    test_score = accuracy_results[kernel]['test']
    cv_score = accuracy_results[kernel]['5_fold_cv']
    loo_score = accuracy_results[kernel]['loo']
    
    ax.bar(idx - bar_width, test_score, bar_width, color=colors[0])
    ax.bar(idx, cv_score, bar_width, color=colors[1])
    ax.bar(idx + bar_width, loo_score, bar_width, color=colors[2])

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=col, label=label) for col, label in zip(colors, labels)]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))  

plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('SVM Performance')

ax.set_xticks(index)
ax.set_xticklabels(kernel_types)

fig.tight_layout()

plt.savefig('svm_performance.png')
plt.show()





