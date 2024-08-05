# Imports
import numpy as np

from ._utils import calc_distance

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    # def fit(self, X, y):
    #     self.X = X
    #     self.y = y

    # def predict(self, X):
    #     y_pred = []
    #     for x in X:
    #         distances = euclidean_distances([x], self.X)[0]
    #         k_indices = np.argsort(distances)[: self.k]
    #         k_nearest_labels = self.y[k_indices]
    #         most_common = np.bincount(k_nearest_labels).argmax()
    #         y_pred.append(most_common)
    #     return np.array(y_pred)

    # def predict_proba(self, X):
    #     y_pred = []
    #     for x in X:
    #         distances = euclidean_distances([x], self.X)[0]
    #         k_indices = np.argsort(distances)[: self.k]
    #         k_nearest_labels = self.y[k_indices]
    #         proba = np.bincount(k_nearest_labels, minlength=self.n_classes) / self.n_neighbors
    #         y_pred.append(proba)
    #     return np.array(y_pred)
