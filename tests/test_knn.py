import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn import MyKNeighborsClassifier
from knn import _generate_simple_synthetic_data


class TestKNN(unittest.TestCase):
    def setUp(self):
        # This method is called before each test, cf https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUp
        self.X, self.y = _generate_simple_synthetic_data(n_samples_per_class=20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=42, test_size=0.2
        )
        self.n_neighbors = 7
        self.n_neighbors_test = 5
        self.atol = 1e-8
        self.rtol = 1e-8

        self.neigh = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, algorithm="kd_tree"
        )
        self.neigh.fit(self.X_train, self.y_train)

        self.my_neigh = MyKNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.my_neigh.fit(self.X_train, self.y_train)

    def test_fit(self):
        # Test the kneighbors method
        neigh_dist, neigh_ind = self.neigh.kneighbors(
            self.X_test[0].reshape(1, -1), n_neighbors=self.n_neighbors_test
        )

        my_neigh_dist, my_neigh_ind = self.my_neigh.kneighbors(
            self.X_test[0].reshape(1, -1), n_neighbors=self.n_neighbors_test
        )

        assert np.allclose(
            neigh_dist, my_neigh_dist, atol=self.atol, rtol=self.rtol
        ), f"Distances to {self.n_neighbors_test} nearest neighbors are not equal"
        assert np.allclose(
            neigh_ind, my_neigh_ind
        ), f"Indices of {self.n_neighbors_test} nearest neighbors are not equal"

    def test_predict(self):
        # Test the predict method
        raise NotImplementedError("Test not implemented")

    def test_score(self):
        # Test the score method
        raise NotImplementedError("Test not implemented")
