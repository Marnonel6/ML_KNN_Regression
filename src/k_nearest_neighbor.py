import numpy as np
from src.distances import euclidean_distances, manhattan_distances, cosine_distances


def find_mode(arr):
    """
    Return the mode (most common element) of `arr`.
    You may use your `numpy_practice` implementation from HW1.
    """
    # print(f"shape - arr = {arr}")
    # print(f"arr = {arr}")

    # most_common_class = np.argmax(np.bincount(arr[:].astype(int)))
    
    uni, uni_c = np.unique(arr, return_counts=True)
    most_common_class = uni[np.argmax(uni_c)] 
    return most_common_class

    #return NotImplementedError

    # print(f"most_common_class = {most_common_class}")

    # return most_common_class

class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator="mode"):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. 

        You should not have to change this __init__ function, but it's
        important to understand how it works.

        Do not import or use these packages: scipy, sklearn, sys, importlib.

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be
                'euclidean,' 'manhattan,' or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels.
            aggregator {str} -- How to aggregate neighbors; either mean or mode.
        """
        self.n_neighbors = n_neighbors

        if aggregator == "mean":
            self.aggregator = np.mean
        elif aggregator == "mode":
            self.aggregator = find_mode
        else:
            raise ValueError(f"Unknown aggregator {aggregator}")

        if distance_measure == "euclidean":
            self.distance = euclidean_distances
        elif distance_measure == "manhattan":
            self.distance = manhattan_distances
        elif distance_measure == "cosine":
            self.distance = cosine_distances
        else:
            raise ValueError(f"Unknown distance {distance_measure}")

    def fit(self, features, targets):
        """
        Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class
        variables that can be accessed in the `predict` function.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples, n_features).
            targets -- Target labels for each data point, shape of (n_samples, 1).
        """
        self.features_train = features
        self.targets_train = targets
        # raise NotImplementedError

    def predict(self, features):
        """
        Use the training data to predict labels on the test features.

        For each test example, find the `self.n_neighbors` closest train
        examples, in terms of the `self.distance` measure. Then, predict the
        test label by using `self.aggregator` among those nearest neighbors.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of
                (n_samples, n_features).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape
                (self.n_samples, 1).
        """

        knn_distance = self.distance(features, self.features_train)

        sort_index = np.zeros((knn_distance.shape[0], knn_distance.shape[1]))
        KNN_index = np.zeros((knn_distance.shape[0], self.n_neighbors))

        for i in range(knn_distance.shape[0]):
            sort_index[i] = np.argsort(knn_distance[i])
            # if self.n_neighbors > knn_distance.shape[1]:
            #     KNN_index[i] = sort_index[i][:]
            # else:
            #     KNN_index[i] = sort_index[i][:self.n_neighbors]
            KNN_index[i] = sort_index[i][:self.n_neighbors]

        # print(f"n_neighbors = {self.n_neighbors}")
        # print(f"temp . shape = {temp.shape}")
        # print(f"temp = {temp}")
        # print(f"KNN . shape = {KNN.shape}")
        # print(f"KNN = {KNN}")

        labels_list = np.zeros((knn_distance.shape[0] ,self.n_neighbors))
        labels = np.zeros((knn_distance.shape[0], 1))


        for s in range(knn_distance.shape[0]):
            for j in range(self.n_neighbors):
                knn_index = int(KNN_index[s,j])
                labels_list[s,j] = self.targets_train[knn_index]
            labels[s] = self.aggregator(labels_list[s])


        # for i in range(knn_distance.shape[0]):
        #     sort_index[i] = np.argsort(knn_distance[i])
        #     knnn_index = int(sort_index[i:])
        #     temp= self.targets_train[knnn_index[i]]
        #     labels[i] = self.aggregator(temp)


        return labels

        #raise NotImplementedError