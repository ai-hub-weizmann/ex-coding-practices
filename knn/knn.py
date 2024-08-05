# Imports
import numpy as np

from ._utils import calc_distance


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric="euclidean", p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(self.y_train)
        self.n_classes = len(np.unique(self.y_train))

        self.distances = calc_distance(self.X_train)
        self.neighbors = np.argsort(self.distances, axis=1)[:, : self.n_neighbors]
        self.neighbors_labels = self.y_train[self.neighbors]
        self.neighbors_distances = self.distances[self.neighbors]

    def predict(self, X):
        if X is self.X_train:
            return self.y_train
        else:
            distances = calc_distance(X, self.X_train, metric=self.metric, p=self.p)
            neighbors = np.argsort(distances, axis=1)[:, : self.n_neighbors]
            neighbors_labels = self.y_train[neighbors]

        predictions = np.array(
            [
                np.argmax(np.bincount(neighbor_labels))
                for neighbor_labels in neighbors_labels
            ]
        )

        return self.classes_[predictions]

    def predict_proba(self, X):
        pass

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

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
