from sklearn.metrics.pairwise import euclidean_distances


def calc_distance(x0, X):
    return euclidean_distances(x0.reshape(1, -1), X)


# accuracy
