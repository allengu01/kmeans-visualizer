import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import os


def plot_rgb(data, centroids, title, azim_angle=-60, elev_angle=30):
    """
    Generic 3D scatter plotting function where each point is a pixel value (RGB)
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.text2D(0.7, 0.3, title, transform=ax.transAxes, fontsize=16)
    
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev_angle, azim=azim_angle)
    plt.subplots_adjust(top = 1.1, bottom = -0.1, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0, 0, 0)

    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    point_opacity = 0.7
    point_edge_color = np.hstack([data/255, np.ones([data.shape[0], 1]) * point_opacity]) # sets color of each point to the pixel value with point_opacity
    ax.scatter3D(r, g, b, edgecolor=point_edge_color, facecolor=np.zeros([data.shape[0], 4]), zorder=1) # facecolor = transparent

    for r, g, b in centroids:
        ax.scatter3D(r, g, b, s=200, facecolor=(r/255, g/255, b/255), edgecolor="black", zorder=2)

    return fig

def plot_iterations(data, centroids_dict, directory_name="figs/"):
    """
    Plots each iteration of data/centroids in succession
    """
    iterate_file_names = []
    for i in range(len(centroids_dict)):
        print("Plotting Iteration:", str(i))

        current_file_name = "iteration_" + str(i) + ".png"
        current_file_path = os.path.join(directory_name, current_file_name)
        iterate_file_names.append(current_file_name)
        print(centroids_dict[i])
        fig = plot_rgb(data, centroids_dict[i], "Iteration: " + str(i))
        fig.savefig(current_file_path, dpi=100, pad_inches=0)
    plt.close("all")
    return iterate_file_names

def plot_rotate(data, centroids, directory_name="figs/"):
    """
    Plots different view angles of the 3D plot with constant elevation and equally incremented azimuthal angles between 0 to 360 degrees
    """
    rotate_file_names = []
    for angle in range(0, 360, 3):
        print("Plotting Rotation Angle:", str(angle), "degrees")
        current_file_name = "rotate_animation" + str(angle) + ".png"
        current_file_path = os.path.join(directory_name, current_file_name)
        rotate_file_names.append(current_file_name)
        fig = plot_rgb(data, centroids, "Iteration: Final", azim_angle=angle)
        fig.savefig(current_file_path, dpi=100, pad_inches=0)
    plt.close("all")
    return rotate_file_names

def plot_palette(k, centroids, directory_name = "figs/"):
    """
    Plots a square palette with k colors found by the clustering algorithm
    """
    color_width = 360 // k
    image_array = np.empty([3, 360, 0])
    for centroid in centroids:
        color_block = np.ones([3, 360, color_width]) * centroid[:, np.newaxis, np.newaxis]
        image_array = np.append(image_array, color_block, axis=2)
    palette_file_name = "palette.png"
    palette_file_path = os.path.join(directory_name, palette_file_name)
    palette = Image.fromarray(image_array.transpose(1, 2, 0).astype(np.uint8), 'RGB')
    palette.save(palette_file_path)

    return palette_file_name