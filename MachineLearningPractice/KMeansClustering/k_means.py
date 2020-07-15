"""
Digits recognition demo, using K-means clustering algorithm.
"""
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics


def test_k_means(estimator, name, data, targets):
    estimator.fit(data)
    print("Name : ", name)
    print("Inertia : ", estimator.inertia_)
    print("Homogeneity score : ", metrics.homogeneity_score(targets, estimator.labels_))
    print("Completeness score : ", metrics.completeness_score(targets, estimator.labels_))
    print("V Measure score : ", metrics.v_measure_score(targets, estimator.labels_))
    print("Adjusted rand score : ", metrics.adjusted_rand_score(targets, estimator.labels_))
    print("Adjusted mutual info score : ", metrics.adjusted_mutual_info_score(targets, estimator.labels_))
    print("Silhouette score : ", metrics.silhouette_score(data, estimator.labels_, metric='euclidean'))


# Loading and scaling the data
digits = load_digits()
data = scale(digits.data)

# Calculating targets, number of clusters, N (number of samples), D (dimensionality, or number of features).
Y = digits.target
K = len(np.unique(Y))
N, D = data.shape

# Creating the classifier
classifier = KMeans(n_clusters=K, init="k-means++", n_init=20, max_iter=500)

# Testing the classifier
test_k_means(classifier, "K means demo", data, Y)
