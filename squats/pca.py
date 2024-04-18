from scipy.interpolate import interp1d
import json
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load data
X = pickle.load(open("squats_interpolated.p", 'rb'))
y = pickle.load(open("labels.p", 'rb'))
X = np.array(X).T
mean_X = np.mean(X, axis=0)
X = X - mean_X
y = np.array(y)
print(X.shape)
print(y.shape)



# Perform PCA
pca = PCA()
x_pca = pca.fit_transform(X)

# Compute explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio = explained_variance_ratio[:40]
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot eigenvalue proportion
fig = plt.figure(figsize=(12, 6))
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(len(cumulative_explained_variance_ratio)), cumulative_explained_variance_ratio, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend(loc='best')
plt.title('Explained Variance Ratio')
plt.show()

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
fig.savefig('explained_variance_ratio.png')
plt.show()
print(index_095)
