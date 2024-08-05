# Imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(random_state=42, test_size=20):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=random_state, test_size=test_size
    )
    return iris, X_train, X_test, y_train, y_test
