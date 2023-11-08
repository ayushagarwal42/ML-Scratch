import numpy as np


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps  # The maximum distance between two data points for them to be considered part of the same cluster
        self.min_samples = min_samples  # The minimum number of data points required to form a dense region or cluster.

    def fit(self, X):  # X is a input data
        self.X = X  # the input data is stored as self.X
        self.labels = [0] * len(X)  #  An array of labels is initialized to zeros. These labels will be used to assign each data point to a cluster.
        self.cluster_id = 0

        for idx, point in enumerate(X):
            if self.labels[idx] == 0:
                if self.expand_cluster(point, idx):
                    self.cluster_id += 1

        return self.labels

    def expand_cluster(self, point, idx):
        seeds = self.region_query(point, idx)
        if len(seeds) < self.min_samples:
            self.labels[idx] = -1  # Mark as noise
            return False
        else:
            self.cluster_id += 1
            self.labels[idx] = self.cluster_id
            for seed in seeds:
                self.labels[seed] = self.cluster_id
            while len(seeds) > 0:
                current_point = seeds[0]
                result = self.region_query(self.X[current_point], current_point)
                if len(result) >= self.min_samples:
                    for i in range(len(result)):
                        result_point = result[i]
                        if self.labels[result_point] == 0 or self.labels[result_point] == -1:
                            if self.labels[result_point] == 0:
                                seeds.append(result_point)
                            self.labels[result_point] = self.cluster_id
                seeds = seeds[1:]
            return True

    def region_query(self, point, idx):
        seeds = []
        for i in range(len(self.X)):
            if np.linalg.norm(self.X[idx] - self.X[i]) < self.eps:
                seeds.append(i)
        return seeds
