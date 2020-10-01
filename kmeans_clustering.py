import numpy as np

def distance(p1, p2):
    """
    Returns the Euclidean distance between two points
    """
    return np.sqrt(np.sum((p2 - p1) * (p2 - p1), axis=1))

def range_of(data):
    """
    Returns the minimum and maximum values for each column in the data
    """
    return {"min" : np.array([np.min(data, axis=0)]), "max" : np.array([np.max(data, axis=0)])}

def random_in_range(data):
    """
    Returns a random array with length equal to the number of columns in data between the range of values in each column
    """
    data_range = range_of(data)
    data_min, data_max = data_range["min"], data_range["max"]
    data_diff = data_max - data_min
    return data_min + np.array([np.random.rand(data.shape[1])]) * data_diff

def initialize_random_centroids(data, k):
    """
    Randomly initializes k centroids between the range of the data
    """
    centroids = np.empty((0, data.shape[1]))
    for _ in range(k):
        centroids = np.concatenate([centroids, random_in_range(data)])
    return centroids

def cluster_data(data, centroids):
    """
    Creates clusters from the data and centroids where each data point belongs to the cluster corresponding to the centroid it is closest to
    """
    clusters = {}
    for i in range(1, len(centroids) + 1):
        clusters[i] = np.empty((0, data.shape[1]))
    for point in data:
        nearest_centroid = int(np.argmin(distance(np.array([point]), centroids))) + 1
        clusters[nearest_centroid] = np.concatenate([clusters[nearest_centroid], np.array([point])])
    return clusters

def cost_function(data, centroids, clusters):
    """
    A cost function using the Euclidean distance (L2 norm) of each data point to its closest centroid and averaging it over the number of data points
    """
    total_cost = 0
    n = data.shape[0]
    for centroid_num in clusters:
        d = distance(clusters[centroid_num], centroids[centroid_num - 1])
        total_cost += np.sum(d)
    return total_cost / n

def update_centroids(data, centroids, clusters):
    """
    Moves each centroid to the mean of all data points in its cluster
    """
    change = False
    for centroid_num in clusters:
        if clusters[centroid_num].shape[0] == 0:
            pass
        else:
            points_in_cluster = clusters[centroid_num]
            new_centroid = np.sum(points_in_cluster, axis=0) / points_in_cluster.shape[0]
            if not np.array_equal(centroids[centroid_num - 1], new_centroid):
                change = True
                centroids[centroid_num - 1] = new_centroid
    return centroids, change

def kmeans(data, k):
    """
    Runs the k-means algorithm

    Returns a dictionary containing the list of centroids for each iteration
    """
    centroids = initialize_random_centroids(data, k)
    clusters = cluster_data(data, centroids)
    centroids_iterations = {}

    centroids_changed = True
    i = 0
    iteration_files = []
    while centroids_changed:
        print("Iteration:", i)
        centroids_iterations[i] = centroids.copy() # Lists passed by reference
        clusters = cluster_data(data, centroids)
        centroids, centroids_changed = update_centroids(data, centroids, clusters)
        print(centroids_iterations)
        i += 1
    centroids_iterations[i] = centroids
    print("Cost:", cost_function(data, centroids, clusters))
    print(centroids_iterations)
    return centroids_iterations