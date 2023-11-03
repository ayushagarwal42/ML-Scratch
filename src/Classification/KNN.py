### Knn from scratch
import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


def manhattan_distance(x1, x2):
    distance = np.sum(np.abs(x1 - x2))
    return distance


class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.y_train = None
        self.X_train = None
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = []
        if self.distance_metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]

        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]

        # k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]  # Use iloc to access by index

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
