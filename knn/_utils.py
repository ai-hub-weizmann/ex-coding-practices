# Optional: place here any utility functions that are used in the knn module, as needed.
import numpy as np

np.random.seed(42)


# Generate synthetic data for testing
def _generate_simple_synthetic_data(n_samples_per_class=20):
    # Class 0: Points around (0, 0)
    class_0 = np.random.randn(n_samples_per_class, 2) * 0.5

    # Class 1: Points around (2, 2)
    class_1 = np.random.randn(n_samples_per_class, 2) * 0.5 + [2, 2]

    # Class 2: Points around (-2, 2)
    class_2 = np.random.randn(n_samples_per_class, 2) * 0.5 + [-2, 2]

    # Combine all points and create labels
    X = np.vstack((class_0, class_1, class_2))
    y = np.array(
        [0] * n_samples_per_class
        + [1] * n_samples_per_class
        + [2] * n_samples_per_class
    )

    return X, y
