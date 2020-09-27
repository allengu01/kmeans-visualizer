import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# READ IMAGE
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
    centroids = initialize_random_centroids(data, k)
    clusters = cluster_data(data, centroids)


    centroids_changed = True
    i = 0
    iteration_files = []
    while centroids_changed:
        print("Iteration:", i)
        plot_filename = 'iteration_' + str(i) + '.png'
        plot_rgb(data, centroids, i, directory_name + plot_filename)

        clusters = cluster_data(data, centroids)
        centroids, centroids_changed = update_centroids(data, centroids, clusters)
        iteration_files.append(plot_filename)
        i += 1
    print("Cost:", cost_function(data, centroids, clusters))
    return centroids, iteration_files

def plot_rgb(data, centroids, i, file_path, end = False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Iteration: " + str(i))

    ax.title.set_position([0.8, 0.2])
    ax.title.set_size(20) 
    
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.subplots_adjust(top = 1.1, bottom = -0.1, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0, 0, 0)

    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    point_opacity = 0.7
    point_edge_color = np.hstack([data/255, np.ones([data.shape[0], 1]) * point_opacity]) # sets color of each point to the pixel value with point_opacity
    ax.scatter3D(r, g, b, edgecolor=point_edge_color, facecolor=np.zeros([data.shape[0], 4]), zorder=1) # facecolor = transparent

    for r, g, b in centroids:
        ax.scatter3D(r, g, b, s=200, facecolor=(r/255, g/255, b/255), edgecolor="black", zorder=2)
    fig.savefig(file_path, dpi=100, bbox='tight', pad_inches=0)

    return fig, ax   

def rotate(data, centroids, directory_name):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0, 0, 0)

    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    point_opacity = 0.7
    point_edge_color = np.hstack([data/255, np.ones([data.shape[0], 1]) * point_opacity]) # sets color of each point to the pixel value with point_opacity
    ax.scatter3D(r, g, b, edgecolor=point_edge_color, facecolor=np.zeros([data.shape[0], 4]), zorder=1) # facecolor = transparent

    for r, g, b in centroids:
        ax.scatter3D(r, g, b, s=200, facecolor=(r/255, g/255, b/255), edgecolor="black", zorder=2)
    
    rotate_filenames = []
    for angle in range(0, 360, 3):
        ax.view_init(elev=30, azim=angle)
        cur_filename = "rotate_animation" + str(angle) + ".png"
        fig.savefig(directory_name + cur_filename, dpi=100, pad_inches=0)
        rotate_filenames.append(cur_filename)
    return rotate_filenames
    
def run_kmeans(data, k):
    filenames = {}

    centroids_result, iteration_files = k_means(data, k, 'figs/')
    final_plot, final_ax = plot_rgb(data, centroids_result, "Final", 'figs/iteration_final.png', True)
    iteration_files.append('iteration_final.png')
    filenames['iterate'] = iteration_files

    rotate_files = rotate(data, centroids_result, 'figs/')
    filenames['rotate'] = rotate_files
    plt.close("all")

    return filenames, centroids_result