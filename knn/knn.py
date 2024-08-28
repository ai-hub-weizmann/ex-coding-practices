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
        if X is None:
            X = self.X_train
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        neigh_dist, neigh_ind = self.kdtree.query(X, k=n_neighbors)
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def predict(self, X):
        indices = self.kneighbors(X, return_distance=False)
        neighbors_labels = self.y_train[indices]

        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_labels
        )
        return predictions

    def predict_proba(self, X):
        indices = self.kneighbors(X, return_distance=False)
        neighbors_labels = self.y_train[indices]

        predictions = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes) / len(x),
            axis=1,
            arr=neighbors_labels,
        )
        return predictions

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
