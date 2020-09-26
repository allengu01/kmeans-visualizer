import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# READ IMAGE
img = cv2.imread('images/minimalist_landscape3.jpg')
resized_img = cv2.resize(img, (30, 30))

def split_rgb(image):
    b, g, r = cv2.split(image)
    return r, g, b

def flatten(arr):
    n = arr.size
    return arr.reshape([n,])

img_r, img_g, img_b = split_rgb(resized_img)
flatten_img_r, flatten_img_g, flatten_img_b = list(map(flatten, [img_r, img_g, img_b]))
pixels = np.stack([flatten_img_r, flatten_img_g, flatten_img_b], axis=1)

def distance(p1, p2):
    """
    Returns the Euclidean distance
    """
    return np.sqrt(np.sum((p2 - p1) * (p2 - p1), axis=1))

def range_of(data):
    return {"min" : np.array([np.min(data, axis=0)]), "max" : np.array([np.max(data, axis=0)])}

def random_in_range(data):
    data_range = range_of(data)
    data_min, data_max = data_range["min"], data_range["max"]
    data_diff = data_max - data_min
    return data_min + np.array([np.random.rand(data.shape[1])]) * data_diff

def initialize_random_centroids(data, k):
    centroids = np.empty((0, data.shape[1]))
    for _ in range(k):
        centroids = np.concatenate([centroids, random_in_range(data)])
    return centroids

def cluster_data(data, centroids):
    clusters = {}
    for i in range(1, len(centroids) + 1):
        clusters[i] = np.empty((0, data.shape[1]))
    for point in data:
        nearest_centroid = int(np.argmin(distance(np.array([point]), centroids))) + 1
        clusters[nearest_centroid] = np.concatenate([clusters[nearest_centroid], np.array([point])])
    return clusters

def update_centroids(data, centroids, clusters):
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

def k_means(data, k, directory_name):
    centroids = initialize_random_centroids(data, k)
    clusters = cluster_data(data, centroids)

    centroids_changed = True
    i = 1
    while centroids_changed:
        print("Iteration:", i)
        plot_rgb(data, centroids, directory_name + 'iteration' + str(i))
        clusters = cluster_data(data, centroids)
        centroids, centroids_changed = update_centroids(data, centroids, clusters)
        i += 1
    return centroids

def plot_rgb(data, centroids, file_path):
    fig = plt.figure(figsize = (8, 6))
    ax = plt.axes(projection='3d')

    for r, g, b in data:
        ax.scatter3D(r, g, b, facecolor='white', edgecolor=(r / 255, g / 255, b / 255))

    for r, g, b in centroids:
        print(r, g, b)
        ax.scatter3D(r, g, b, s=200, color=(r / 255, g / 255, b / 255))
    
    plt.savefig(file_path, dpi=300)

centroids_result = k_means(pixels, 4, 'figs/')
plot_rgb(pixels, centroids_result, 'figs/final')
