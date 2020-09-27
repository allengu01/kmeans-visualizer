import os, shutil
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.tools as tls
import plotly.express as px
import cv2

# READ IMAGE
img = cv2.imread('images/minimalist_landscape1.jpg')
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

def cost_function(data, centroids, clusters):
    total_cost = 0
    n = data.shape[0]
    for centroid_num in clusters:
        d = distance(clusters[centroid_num], centroids[centroid_num - 1])
        total_cost += np.sum(d)
    return total_cost / n

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
    reset()

    centroids = initialize_random_centroids(data, k)
    clusters = cluster_data(data, centroids)
    plot_rgb(data, centroids, "Initialize Random Centroids", directory_name + 'iteration_start' + '.png')


    centroids_changed = True
    i = 1
    while centroids_changed:
        print("Iteration:", i)
        clusters = cluster_data(data, centroids)
        centroids, centroids_changed = update_centroids(data, centroids, clusters)
        plot_rgb(data, centroids, i, directory_name + 'iteration_' + str(i) + '.png')

        i += 1
    print("Cost:", cost_function(data, centroids, clusters))
    return centroids

def plot_rgb(data, centroids, iteration, file_path, end = False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("Iteration: " + str(iteration))

    ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = 10, 10, 10
    ax.title.set_position([0.85, 1])
    ax.title.set_size(10) 

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    point_opacity = 0.7
    point_edge_color = np.hstack([data/255, np.ones([data.shape[0], 1]) * point_opacity]) # sets color of each point to the pixel value with point_opacity
    ax.scatter3D(r, g, b, edgecolor=point_edge_color, facecolor=np.zeros([data.shape[0], 4]), zorder=1) # facecolor = transparent

    for r, g, b in centroids:
        ax.scatter3D(r, g, b, s=200, facecolor=(r/255, g/255, b/255), edgecolor="black", zorder=2)
    fig.savefig(file_path, dpi=300)

    if end:
        pass
        plt.show()
    else:
        fig.close()

    return fig, ax   

def reset():
    shutil.rmtree("figs")
    os.mkdir('figs')
    
def run_kmeans():
    reset()

    centroids_result = k_means(pixels, 4, 'figs/')
    final_plot, final_ax = plot_rgb(pixels, centroids_result, "Final", 'figs/iteration_final.png', True)

