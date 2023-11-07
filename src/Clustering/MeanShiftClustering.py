import numpy as np


class MeanShift:
    def __init__(self, radius=4.0, max_iters=100):
        self.radius = radius
        self.max_iters = max_iters
        self.centroids = {}

    def fit(self, data):
        centroids = {}  # Initialize an empty dictionary to store centroids
        for i in range(len(data)):
            centroids[i] = data[i]  # Initialize centroids with data points

        for _ in range(self.max_iters):
            new_centroids = {}  # Create a new dictionary for updated centroids

            for i in centroids:
                in_bandwidth = []  # Initialize a list to store data points within the radius of the centroid
                centroid = centroids[i]  # Get the current centroid
                for feature_set in data:
                    if np.linalg.norm(feature_set - centroid) < self.radius:
                        in_bandwidth.append(feature_set)  # Collect data points within the radius

                new_centroid = np.average(in_bandwidth, axis=0)  # Calculate the new centroid as the mean of in-radius points
                new_centroids[i] = new_centroid  # Store the updated centroid in the new dictionary

            optimized = True  # Initialize a flag to check if the algorithm has converged

            for i in centroids:
                if not np.array_equal(new_centroids[i], centroids[i]):
                    optimized = False  # If any centroid has changed, set the optimization flag to False
                    break

            if optimized:
                break  # If the centroids have not changed, the algorithm has converged and we can exit

            centroids = new_centroids  # Update the centroids with the new ones

        self.centroids = centroids  # Store the final centroids in the class variable

    def predict(self, data):
        labels = []  # empty list that will store the assigned cluster labels for each data point in the input data
        for feature_set in data:
            distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
            label = distances.index(min(distances))
            labels.append(label)
        return labels
