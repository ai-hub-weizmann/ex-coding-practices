# Imports
import numpy as np
from scipy.spatial import KDTree


class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(self.y_train)
        self.n_classes = len(np.unique(self.y_train))

        self.kdtree = KDTree(self.X_train)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        raise NotImplementedError("Method not implemented")

    def predict(self, X):
        raise NotImplementedError("Method not implemented")

    def predict_proba(self, X):
        raise NotImplementedError("Method not implemented")

    def score(self, X, y_true):
        raise NotImplementedError("Method not implemented")
