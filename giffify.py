from kmeans_mpl import run_kmeans
import numpy as np
import os, shutil
import imageio
import cv2


def split_rgb(image):
    b, g, r = cv2.split(image)
    return r, g, b

def flatten(arr):
    n = arr.size
    return arr.reshape([n,])

def reset():
    shutil.rmtree("figs")
    os.mkdir('figs')

def main():
    reset()

    image_file = 'starrynight.jpg'
    img = cv2.imread('images/' + image_file) # change file here
    resized_img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)
    #resized_img = cv2.resize(img, (60, 80))
    img_r, img_g, img_b = split_rgb(resized_img)
    flatten_img_r, flatten_img_g, flatten_img_b = list(map(flatten, [img_r, img_g, img_b]))
    pixels = np.stack([flatten_img_r, flatten_img_g, flatten_img_b], axis=1)

    # K-MEANS
    k = 3
    print("Number of Pixels:", pixels.shape[0])
    kmeans_filenames = run_kmeans(pixels, k)

    # ITERATION ANIMATION
    iterate_animation_dir = 'figs/'
    images = []
    for filename in kmeans_filenames["iterate"]:
        print(filename)
        assert filename.endswith('.png')
        file_path = os.path.join(iterate_animation_dir, filename)
        images.append(imageio.imread(file_path))
    imageio.mimsave('figs/iterate_animation.gif', images, fps=1)

    # ROTATION ANIMATION
    rotate_animation_dir = 'figs/'
    images = []
    for filename in kmeans_filenames["rotate"]:
        print(filename)
        assert filename.endswith('.png') and filename.startswith('rotate')
        file_path = os.path.join(rotate_animation_dir, filename)
        images.append(imageio.imread(file_path))
        os.remove(file_path)
    imageio.mimsave('figs/rotate_animation.gif', images, fps=10)

main()