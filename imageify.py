from kmeans_clustering import kmeans
from plot_mpl import plot_iterations, plot_rotate, plot_palette
import numpy as np
import os, shutil
import imageio
import cv2
import PIL as Image

def split_rgb(image):
    b, g, r = cv2.split(image)
    return r, g, b

def flatten(arr):
    n = arr.size
    return arr.reshape([n,])

def reset():
    shutil.rmtree("figs")
    os.mkdir("figs")

    shutil.rmtree("output")
    os.mkdir("output")

def run_kmeans(data, k):
    file_names = {}
    centroids_dict = kmeans(data, k)
    final_centroids = centroids_dict[len(centroids_dict)-1]
    iteration_files = plot_iterations(data, centroids_dict)
    file_names['iterate'] = iteration_files

    rotate_files = plot_rotate(data, final_centroids)
    file_names['rotate'] = rotate_files

    palette_file = plot_palette(k, final_centroids)
    file_names['palette'] = palette_file
    return file_names

def main():
    reset()

    image_file = 'starrynight.jpg'
    img = cv2.imread('images/' + image_file)
    resized_img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)
    img_r, img_g, img_b = split_rgb(resized_img)
    flatten_img_r, flatten_img_g, flatten_img_b = list(map(flatten, [img_r, img_g, img_b]))
    pixels = np.stack([flatten_img_r, flatten_img_g, flatten_img_b], axis=1)

    # K-MEANS
    k = 3
    print("Number of Pixels:", pixels.shape[0])
    kmeans_file_names = run_kmeans(pixels, k)

    # ITERATION ANIMATION
    iterate_animation_dir = 'figs'
    images = []
    for file_name in kmeans_file_names["iterate"]:
        assert file_name.endswith('.png') and file_name.startswith('iteration')
        file_path = os.path.join(iterate_animation_dir, file_name)
        images.append(imageio.imread(file_path))
        shutil.copyfile(file_path, "output/" + file_name)
    imageio.mimsave('output/iterate_animation.gif', images, fps=1)

    # ROTATION ANIMATION
    rotate_animation_dir = 'figs'
    images = []
    for file_name in kmeans_file_names["rotate"]:
        assert file_name.endswith('.png') and file_name.startswith('rotate')
        file_path = os.path.join(rotate_animation_dir, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave('output/rotate_animation.gif', images, fps=10)        

    # COLOR PALETTE
    palette_dir = 'figs'
    palette_file_name = kmeans_file_names["palette"]
    palette_file_path = os.path.join(palette_dir, palette_file_name)
    shutil.copyfile(palette_file_path, "output/" + palette_file_name)
main()