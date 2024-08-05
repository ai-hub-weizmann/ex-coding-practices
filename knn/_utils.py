from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist, pdist, squareform

# def calc_distance(x0, X):
#     return euclidean_distances(x0.reshape(1, -1), X)


def calc_distance(X, metric="euclidean"):
    # return cdist(X, X, metric=metric)
    return squareform(pdist(X, metric=metric))


# score
